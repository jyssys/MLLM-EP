#!/usr/bin/env python3
"""Fixed four-GPU fleet comparison for native vLLM PCP.

Each API process is an independent serving replica and owns ``pcp`` workers.
Current CUDA PCP explicitly rejects vLLM-managed DP>1, so independent replicas
are the faithful fixed-fleet deployment.  ``device_ids`` selects a disjoint
subset while every process retains CUDA_VISIBLE_DEVICES=4,5,6,7.  Thus the
three configurations consume exactly four physical GPUs:

  PCP1: DP4 x PCP1, PCP2: DP2 x PCP2, PCP4: DP1 x PCP4.

The prompt token IDs and arrival barrier are identical across configurations.
This is a request-level screen; detailed operator attribution is only warranted
if the request-level regret clears the preregistered gate.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import time
import traceback
from pathlib import Path


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def make_prompt_ids(tokenizer, length: int, request_index: int, kind: str = "natural") -> list[int]:
    seed = tokenizer.encode(
        f"Request {request_index}. Analyze this long context carefully and state its main conclusion. ",
        add_special_tokens=False,
    )
    fillers = {
        "natural": "The quick brown fox crosses the field while a systems researcher records every causal dependency. ",
        "code": "def update_state(queue, token): return queue.append((token, token % 4)) # verify invariant\n",
        "math": "For integers x and y, prove that x squared plus two x y plus y squared equals (x+y) squared. ",
        "repetitive": "alpha alpha alpha alpha beta beta gamma delta ",
    }
    filler = tokenizer.encode(fillers[kind], add_special_tokens=False)
    ids = list(seed)
    while len(ids) < length:
        ids.extend(filler)
    return ids[:length]


def worker(rank: int, args: argparse.Namespace, port: int, barrier, queue) -> None:
    out = args.output_dir / f"driver_dp{rank}.json"
    try:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
        for name in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                     "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
            os.environ.pop(name, None)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=1,
            prefill_context_parallel_size=args.pcp,
            device_ids=list(range(rank * args.pcp, (rank + 1) * args.pcp)),
            trust_remote_code=True,
            enforce_eager=True,
            kv_cache_memory_bytes=args.kv_cache_gb << 30,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_concurrency,
            enable_prefix_caching=False,
            enable_expert_parallel=False,
            enable_flashinfer_autotune=False,
            disable_log_stats=False,
        )
        pc = llm.llm_engine.vllm_config.parallel_config
        assert pc.prefill_context_parallel_size == args.pcp
        assert pc.data_parallel_size == 1
        assert pc.tensor_parallel_size == 1
        assert not pc.enable_expert_parallel
        sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
        records = []
        regimes = json.loads(args.regimes)
        for regime in regimes:
            length, concurrency = regime[:2]
            kind = regime[2] if len(regime) > 2 else "natural"
            local_indices = [i for i in range(concurrency) if i % args.dp == rank]
            prompts = [
                {"prompt_token_ids": make_prompt_ids(tokenizer, length, i, kind)}
                for i in local_indices
            ]
            for iteration in range(args.warmups + args.repetitions):
                barrier.wait(timeout=1800)
                started = time.monotonic()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                finished = time.monotonic()
                barrier.wait(timeout=1800)
                rows = []
                for request_index, output in zip(local_indices, outputs, strict=True):
                    metrics = output.metrics
                    assert metrics is not None
                    first = float(metrics.first_token_ts)
                    last = float(metrics.last_token_ts)
                    first_latency = float(metrics.first_token_latency)
                    token_ids = [int(x) for x in output.outputs[0].token_ids]
                    rows.append(
                        {
                            "request_index": request_index,
                            "prompt_tokens": len(output.prompt_token_ids or []),
                            "output_token_ids": token_ids,
                            "ttft_ms": 1000 * first_latency,
                            "e2e_ms": 1000 * (first_latency + last - first),
                            "tpot_ms": 1000 * (last - first) / max(1, len(token_ids) - 1),
                        }
                    )
                records.append(
                    {
                        "pcp": args.pcp,
                        "replicas": args.dp,
                        "rank": rank,
                        "length": length,
                        "concurrency": concurrency,
                        "kind": kind,
                        "iteration": iteration,
                        "warmup": iteration < args.warmups,
                        "driver_wall_ms": 1000 * (finished - started),
                        "requests": rows,
                    }
                )
        write_json(
            out,
            {
                "ok": True,
                "topology": {"tp": 1, "managed_dp": 1, "replicas": args.dp,
                             "pcp": args.pcp, "ep": 1},
                "records": records,
            },
        )
        llm.llm_engine.engine_core.shutdown()
        queue.put((rank, True, ""))
    except BaseException:
        tb = traceback.format_exc()
        write_json(out, {"ok": False, "traceback": tb})
        queue.put((rank, False, tb))
        raise


def aggregate(args: argparse.Namespace) -> None:
    drivers = [json.loads((args.output_dir / f"driver_dp{r}.json").read_text()) for r in range(args.dp)]
    assert all(x["ok"] for x in drivers)
    groups: dict[tuple[int, int, str, int], list[dict]] = {}
    walls: dict[tuple[int, int, str, int], list[float]] = {}
    for driver in drivers:
        for rec in driver["records"]:
            if rec["warmup"]:
                continue
            key = (rec["length"], rec["concurrency"], rec["kind"], rec["iteration"])
            groups.setdefault(key, []).extend(rec["requests"])
            walls.setdefault(key, []).append(rec["driver_wall_ms"])
    iterations = []
    for (length, concurrency, kind, iteration), rows in sorted(groups.items()):
        assert len(rows) == concurrency, (length, concurrency, iteration, len(rows))
        wall_ms = max(walls[(length, concurrency, kind, iteration)])
        ttft = [r["ttft_ms"] for r in rows]
        e2e = [r["e2e_ms"] for r in rows]
        iterations.append(
            {
                "pcp": args.pcp,
                "replicas": args.dp,
                "length": length,
                "concurrency": concurrency,
                "kind": kind,
                "iteration": iteration,
                "ttft_p50_ms": statistics.median(ttft),
                "ttft_p90_ms": percentile(ttft, 0.9),
                "e2e_p50_ms": statistics.median(e2e),
                "e2e_p90_ms": percentile(e2e, 0.9),
                "fleet_wall_ms": wall_ms,
                "prompt_throughput_tok_s": length * concurrency / (wall_ms / 1000),
                "outputs": {str(r["request_index"]): r["output_token_ids"] for r in rows},
            }
        )
    write_json(
        args.output_dir / "summary.json",
        {
            "contract": "FOUR_GPU_FIXED_FLEET_NATIVE_PCP; EXACT_PROMPT_TOKEN_COUNT; EP_OFF",
            "topology": {"tp": 1, "managed_dp": 1, "replicas": args.dp,
                         "pcp": args.pcp, "ep": 1},
            "vllm": "0.26.0+cu129",
            "visible_devices": "4,5,6,7",
            "iterations": iterations,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pcp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--regimes", default="[[8192,4],[16384,4]]")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--kv-cache-gb", type=int, default=8)
    args = parser.parse_args()
    args.dp = 4 // args.pcp
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "args.json", vars(args) | {"output_dir": str(args.output_dir)})
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.dp)
    queue = context.Queue()
    port = free_port()
    processes = [context.Process(target=worker, args=(r, args, port, barrier, queue)) for r in range(args.dp)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(3600)
    codes = [process.exitcode for process in processes]
    if codes != [0] * args.dp:
        raise RuntimeError(f"worker failures: {codes}")
    aggregate(args)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
