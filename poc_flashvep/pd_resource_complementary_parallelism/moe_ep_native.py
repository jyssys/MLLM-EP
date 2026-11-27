"""Native Qwen3 MoE EP2/EP2 PD or one EP2 colocated server with the shared client."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

from . import native


ALL2ALL_BACKENDS = (
    "allgather_reducescatter", "deepep_high_throughput", "deepep_low_latency",
    "flashinfer_nvlink_two_sided",
)


def launch(args, role, start_device, port):
    label = f"{role}0"
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": ",".join(devices[start_device:start_device + 2]),
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(port + 1000),
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "PYTHONHASHSEED": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    backend = args.all2all_backend
    batch_tokens = args.batch_tokens
    if role == "D":
        backend = args.decode_all2all_backend or backend
        batch_tokens = args.decode_batch_tokens or batch_tokens
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--host", "127.0.0.1",
        "--port", str(port), "--model", args.model, "--served-model-name", native.MODEL,
        "--dtype", "bfloat16", "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(batch_tokens), "--max-num-seqs", str(args.max_seqs),
        "--generation-config", "vllm", "--data-parallel-size", "2",
        "--tensor-parallel-size", "1", "--enable-expert-parallel", "--expert-placement-strategy", "linear",
        "--all2all-backend", backend, "--enable-chunked-prefill", "--data-parallel-rpc-port", str(port + 2000),
        "--no-enable-prefix-caching", "--attention-backend", "FLASH_ATTN",
        "--enable-request-id-headers", "--kv-cache-memory-bytes", str(int(args.cache_gib * native.GIB)),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.profile_case:
        command += ["--profiler-config", json.dumps({
            "profiler": "torch", "torch_profiler_dir": str((args.out / "profiles" / label).resolve()),
            "torch_profiler_with_stack": False, "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True,
        })]
    if args.mode == "pd":
        handoff = {"kv_connector": "NixlConnector", "kv_role": "kv_producer" if role == "P" else "kv_consumer"}
        command += ["--kv-transfer-config", json.dumps(handoff)]
    native.write(args.out / f"{label}_command.json", {"argv": command, "physical": env["CUDA_VISIBLE_DEVICES"]})
    with (args.out / f"{label}.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    return label, port, process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, metavar="NAME:PATH:QPS")
    parser.add_argument("--mode", choices=("pd", "dp"), default="pd")
    parser.add_argument("--all2all-backend", choices=ALL2ALL_BACKENDS, default="allgather_reducescatter")
    parser.add_argument("--decode-all2all-backend", choices=ALL2ALL_BACKENDS)
    parser.add_argument("--decode-batch-tokens", type=int)
    parser.add_argument("--cache-gib", type=float, default=24)
    parser.add_argument("--max-model-len", type=int, default=33024)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--port", type=int, default=19740)
    parser.add_argument("--profile-case")
    parser.add_argument("--natural", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    args.apc, args.no_chunk, args.p_tp = False, False, 1
    args.activity_library, args.shape_case = None, None
    args.cross_layer_blocks, args.deflect_at = False, 0
    cases = [native.parse_case(case) for case in args.case]
    if any(qps <= 0 or any(row["output_tokens"] < 16 or row["input_tokens"] + row["output_tokens"] > args.max_model_len
                           for row in rows) for _, _, rows, qps in cases):
        parser.error("Need positive offered QPS, at least 16 fixed output tokens, and valid context")
    placement = (("P", 0), ("D", 2)) if args.mode == "pd" else (("R", 0),)
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 2 * len(placement):
        parser.error("Physical GPU lease must match 2 devices per EP2 group")
    args.out.mkdir(parents=True)
    native.write(args.out / "ep_runtime.json", {"ep_per_group": 2, "tp": 1, "group_mode": args.mode,
                   "scheduler_dp_per_group": 2, "p_or_colocated_backend": args.all2all_backend,
                   "decode_backend": args.decode_all2all_backend or args.all2all_backend,
                   "eager": args.enforce_eager, "apc": False})
    servers = []
    try:
        for index, (role, start_device) in enumerate(placement):
            servers.append(launch(args, role, start_device, args.port + 16 * index))
        asyncio.run(native.run(args, servers, cases))
    finally:
        for _, _, process in servers:
            if process.poll() is None:
                process.terminate()
        for _, _, process in servers:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
