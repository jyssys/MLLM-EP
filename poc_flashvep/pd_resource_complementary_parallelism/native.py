"""Native vLLM 1P/1D and DP with real NIXL and fixed Poisson schedules."""
import argparse
import asyncio
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import aiohttp

GIB = 1024**3
MODEL = "study"
SLO = {"ttft_ms": 5000, "tpot_ms": 33, "max_itl_ms": 34}
METRIC = re.compile(r'^(vllm:[\w:]+)(\{[^}]*\})?\s+([\d.eE+-]+)')
WAIT_REASONS = re.compile(r'reason="(capacity|deferred)"')


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def launch(args, role, member, start_device, tp, port):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    label = f"{role}{member}"
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(devices[start_device:start_device + tp]),
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(port + 1000),
                        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1", "PYTHONHASHSEED": "0"}
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--host", "127.0.0.1", "--port", str(port),
        "--model", args.model, "--served-model-name", MODEL, "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len), "--block-size", "16",
        "--max-num-batched-tokens", str(args.batch_tokens),
        "--max-num-seqs", str(args.max_seqs), "--generation-config", "vllm",
        "--tensor-parallel-size", str(tp), "--kv-cache-memory-bytes", str(int(args.cache_gib * GIB)),
        "--enable-prefix-caching" if args.apc else "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill" if args.no_chunk else "--enable-chunked-prefill",
        "--enable-request-id-headers", "--attention-backend", args.attention,
    ]
    if role in ("P", "D"):
        handoff = {"kv_connector": "NixlConnector", "kv_role": "kv_producer" if role == "P" else "kv_consumer"}
        if args.cross_layer_blocks:
            handoff["kv_connector_extra_config"] = {"enable_cross_layers_blocks": True}
        command += ["--kv-transfer-config", json.dumps(handoff)]
    if args.activity_library or args.shape_case or args.cross_layer_blocks:
        env.update(VLLM_SERVER_DEV_MODE="1")
        if args.activity_library:
            env.update(PD_ACTIVITY_LIBRARY=str(args.activity_library.resolve()),
                       PD_ACTIVITY_OUTPUT=str((args.out / "activity" / label).resolve()))
        command += ["--worker-extension-cls",
                    "poc_flashvep.pd_resource_complementary_parallelism.kernel_worker.KernelActivityWorker"]
    if args.profile_case and not args.activity_library:
        command += ["--profiler-config", json.dumps({
            "profiler": "torch", "torch_profiler_dir": str((args.out / "profiles" / label).resolve()),
            "torch_profiler_with_stack": False, "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True,
        })]
    write(args.out / f"{label}_command.json", {"argv": command, "physical": env["CUDA_VISIBLE_DEVICES"]})
    with (args.out / f"{label}.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    return label, port, process


def URL(port, route):
    return f"http://127.0.0.1:{port}/{route}"


async def ready(session, server):
    label, port, process = server
    limit = time.monotonic() + 520
    while time.monotonic() < limit:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited, see its log")
        try:
            async with session.get(URL(port, "health"), timeout=3) as response:
                if response.ok:
                    return
        except (TimeoutError, aiohttp.ClientError):
            pass
        await asyncio.sleep(0.5)
    raise TimeoutError(f"{label} startup timeout")


async def snapshot(session, servers, out, stage):
    for role, port, _ in servers:
        async with session.get(URL(port, "metrics"), timeout=20) as response:
            response.raise_for_status()
            (out / f"{stage}_{role}.prom").write_bytes(await response.read())


async def sample_queues(session, servers, output):
    fields = {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"}
    with output.open("w", buffering=1) as f:
        while True:
            for role, port, _ in servers:
                try:
                    async with session.get(URL(port, "metrics"), timeout=4) as response:
                        data = {}
                        for line in (await response.text()).splitlines():
                            match = METRIC.match(line)
                            if match is None:
                                continue
                            name, labels, value = match.groups()
                            if name == "vllm:num_requests_waiting_by_reason":
                                reason = WAIT_REASONS.search(labels or "")
                                if reason:
                                    key = f"{name}:{reason[1]}"
                                    data[key] = data.get(key, 0) + float(value)
                            elif name in fields:
                                data[name] = data.get(name, 0) + float(value)
                        f.write(json.dumps({"s": time.monotonic(), "rank": role, **data}) + "\n")
                except (TimeoutError, aiohttp.ClientError):
                    continue
            await asyncio.sleep(0.5)


async def reset_profiler(session, servers, event, out, activity):
    reply = {}
    for role, port, _ in servers:
        route = "collective_rpc" if activity else event
        method = "study_activity_start" if event == "start_profile" else "study_activity_stop"
        body = {"method": method} if activity else None
        async with session.post(URL(port, route), json=body, timeout=300) as response:
            if not response.ok:
                raise RuntimeError(f"{role} {event}: {response.status} {await response.text()}")
            if activity:
                reply[role] = await response.json()
    if reply:
        write(out / f"{event}_clock.json", reply)


async def generate(session, args, row, deadline, index, prefill, decode, outstanding):
    await asyncio.sleep(max(0, deadline - time.monotonic()))
    start = time.monotonic()
    p = min(prefill, key=outstanding.get) if prefill else None
    candidates = decode or prefill
    d = min(candidates, key=outstanding.get)
    prompt = row["prompt_token_ids"]
    result = {"id": row["request_id"], "arrival": deadline, "start": start, "input": len(prompt),
              "P_port": p, "D_port": d, "expected_output": row["output_tokens"]}
    tokens, stamp, event_sizes = [], [], []
    base = {"model": MODEL, "prompt": prompt, "temperature": 0.0,
            "max_tokens": row["output_tokens"], "ignore_eos": not args.natural, "stream": True,
            "return_token_ids": True}
    request_id = f"pd-study-{index}"
    deflected = bool(args.mode == "pd" and args.deflect_at and outstanding[p] >= args.deflect_at
                     and outstanding[d] == 0)
    result["execution"] = ("D_local_prefill" if deflected else
                           "PD_handoff" if args.mode == "pd" else "local")
    if args.mode == "prefill":
        # Independent P service reference. This is an internal one-token stage
        # calibration. All end-to-end fixed serving requests use at least 16.
        base["max_tokens"] = 1
    try:
        if args.mode == "pd" and not deflected:
            provisional = dict(base, max_tokens=1, stream=False, ignore_eos=True)
            provisional["kv_transfer_params"] = {
                "do_remote_decode": True, "do_remote_prefill": False,
                "remote_engine_id": None, "remote_block_ids": None, "remote_host": None, "remote_port": None,
            }
            outstanding[p] += 1
            try:
                async with session.post(URL(p, "v1/completions"), json=provisional,
                                        headers={"x-request-id": request_id + ":P"}) as response:
                    warm = await response.json()
                    if not response.ok:
                        raise RuntimeError(f"P {response.status}: {str(warm)[:1200]}")
            finally:
                outstanding[p] -= 1
            result["p_finish"] = time.monotonic()
            base["kv_transfer_params"] = warm["kv_transfer_params"]
            result["serialized_kv_handoff_metadata_bytes"] = len(
                json.dumps(base["kv_transfer_params"], separators=(",", ":")).encode()
            )
            blocks = warm["kv_transfer_params"].get("remote_block_ids")
            if blocks is None:
                raise RuntimeError("No physical NIXL lease from P")
            result["remote_block_groups"] = [len(g) for g in blocks] if blocks and isinstance(blocks[0], list) else [len(blocks)]
            d = min(decode, key=outstanding.get)
            result["D_port"] = d
        result["d_submit"] = time.monotonic()
        outstanding[d] += 1
        try:
            async with session.post(URL(d, "v1/completions"), json=base,
                                    headers={"x-request-id": request_id + ":D"}) as response:
                if not response.ok:
                    raise RuntimeError(f"D {response.status}: {(await response.text())[:1200]}")
                async for raw in response.content:
                    if not raw.startswith(b"data:") or raw.strip() == b"data: [DONE]":
                        continue
                    obj = json.loads(raw[5:])
                    if "error" in obj:
                        raise RuntimeError(str(obj["error"])[:1200])
                    for chunk in obj.get("choices", ()):
                        incoming = chunk.get("token_ids") or []
                        if incoming:
                            now = time.monotonic()
                            tokens.extend(incoming)
                            stamp.extend([now] * len(incoming))
                            event_sizes.append(len(incoming))
        finally:
            outstanding[d] -= 1
        if not stamp or (not args.natural and len(tokens) != base["max_tokens"]):
            raise RuntimeError(f"Expected {base['max_tokens']} output tokens, received {len(tokens)}")
        itl = [1000 * (later - first) for first, later in itertools.pairwise(stamp)]
        ttft = 1000 * (stamp[0] - start)
        tpot = statistics.mean(itl) if itl else 0.0
        result.update(status="OK", ttft_ms=ttft, arrival_ttft_ms=1000 * (stamp[0] - deadline),
                      itl_ms=itl, tpot_ms=tpot, first_token=stamp[0],
                      slo_pass=ttft <= SLO["ttft_ms"] and tpot <= SLO["tpot_ms"] and max(itl, default=0) <= SLO["max_itl_ms"])
    except (TimeoutError, aiohttp.ClientError, RuntimeError, KeyError, ValueError) as error:
        result.update(status="FAILED", error=str(error), slo_pass=False)
    result.update(finish=time.monotonic(), tokens=tokens, token_times=stamp, sse_groups=event_sizes)
    return result


def parse_case(spec):
    name, path, qps = spec.split(":", maxsplit=2)
    source = Path(path)
    trace = json.loads(source.read_text())
    return name, source, trace["requests"], float(qps)


async def shapes(session, servers, action):
    data = {}
    for role, port, _ in servers:
        async with session.post(URL(port, "collective_rpc"), json={"method": "study_shape_" + action}) as response:
            response.raise_for_status()
            data[role] = await response.json()
    return data


async def cohort(session, args, servers, case, suffix="", profiled=False, shape_observed=False):
    name, source, rows, rate = case
    out = args.out / (name + suffix)
    out.mkdir()
    prefill = [p for label, p, _ in servers if label.startswith("P")]
    decode = [p for label, p, _ in servers if not label.startswith("P")]
    profile_servers = [server for server in servers if server[0].startswith("P")
                       or not getattr(args, "profile_prefill_only", False)]
    counts = dict.fromkeys(prefill + decode, 0)
    await snapshot(session, servers, out, "before")
    if profiled:
        await reset_profiler(session, profile_servers, "start_profile", out, args.activity_library)
    if shape_observed:
        await shapes(session, servers, "start")
    from .telemetry import poll_gpu

    sampler = asyncio.create_task(sample_queues(session, servers, out / "queues.jsonl"))
    monitor = asyncio.create_task(poll_gpu(out / "gpu.jsonl"))
    start = time.monotonic() + 0.2
    origin = time.time() - time.monotonic()
    tasks = [asyncio.create_task(generate(session, args, row, start + row["arrival_unit_s"] / rate,
                                         i, prefill, decode, counts)) for i, row in enumerate(rows)]
    results = []
    with (out / "requests.jsonl").open("w", buffering=1) as f:
        for task in asyncio.as_completed(tasks):
            row = await task
            results.append(row)
            f.write(json.dumps(row) + "\n")
    sampler.cancel()
    monitor.cancel()
    await asyncio.gather(sampler, monitor, return_exceptions=True)
    if profiled:
        await reset_profiler(session, profile_servers, "stop_profile", out, args.activity_library)
    if shape_observed:
        write(out / "shapes.json", await shapes(session, servers, "stop"))
    await snapshot(session, servers, out, "after")
    write(out / "run.json", {
        "model": args.model, "mode": args.mode, "case": name, "qps": rate, "count": len(rows),
        "profiled": profiled, "profiler": ("cupti_kernel_only" if args.activity_library else "kineto") if profiled else None,
        "profile_roles": [server[0] for server in profile_servers] if profiled else [],
        "start": start, "clock_epoch_minus_mono": origin,
        "nixl_cross_layer_blocks_requested": args.cross_layer_blocks, "deflect_at": args.deflect_at,
        "deflected_count": sum(r["execution"] == "D_local_prefill" for r in results),
        "deflected_completed_input_tokens": sum(r["input"] for r in results
                                               if r["execution"] == "D_local_prefill" and r["status"] == "OK"),
        "finish": max(row["finish"] for row in results), "apc": args.apc, "chunked": not args.no_chunk,
        "slo": SLO, "natural": args.natural, "trace_sha256": sha256(source.read_bytes()).hexdigest(),
        "cache_gib_per_device": args.cache_gib, "batch_tokens": args.batch_tokens,
        "prefill_workers": len(prefill), "decode_workers": len(decode), "p_tp": args.p_tp,
    })
    print(name + suffix, len(results), sum(row["status"] == "OK" for row in results), flush=True)
    if not any(row["status"] == "OK" for row in results):
        raise RuntimeError(f"{name} all requests failed")


async def run(args, servers, cases):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=310),
                                    connector=aiohttp.TCPConnector(limit=0), trust_env=False) as session:
        await asyncio.gather(*(ready(session, server) for server in servers))
        p = [port for label, port, _ in servers if label.startswith("P")]
        d = [port for label, port, _ in servers if not label.startswith("P")]
        counts = dict.fromkeys(p + d, 0)
        warm = cases[0][2][:min(4, len(cases[0][2]))]
        ready_rows = await asyncio.gather(*(generate(session, args, row, time.monotonic(), i, p, d, counts)
                                            for i, row in enumerate(warm)))
        write(args.out / "warmup.json", ready_rows)
        if not all(item["status"] == "OK" for item in ready_rows):
            raise RuntimeError("Native warmup failed")
        if args.mode == "pd" and (args.cross_layer_blocks or args.activity_library):
            layouts = {}
            for role, port, _ in servers:
                async with session.post(URL(port, "collective_rpc"), json={"method": "study_cache_layout"}) as reply:
                    layouts[role] = {"status": reply.status, "body": await reply.text()}
            write(args.out / "kv_layout.json", layouts)
        for case in cases:
            if case[0] == args.profile_case:
                await cohort(session, args, servers, case, "_clean_before")
                await cohort(session, args, servers, case, "_profile", profiled=True)
                await cohort(session, args, servers, case, "_clean_after")
            elif case[0] == args.shape_case:
                await cohort(session, args, servers, case, "_clean_before")
                await cohort(session, args, servers, case, "_shape", shape_observed=True)
                await cohort(session, args, servers, case, "_clean_after")
            else:
                await cohort(session, args, servers, case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, metavar="NAME:PATH:QPS")
    parser.add_argument("--mode", choices=("pd", "dp", "prefill"), default="pd")
    parser.add_argument("--prefill-workers", type=int, default=1)
    parser.add_argument("--decode-workers", type=int, default=1)
    parser.add_argument("--p-tp", type=int, default=1)
    parser.add_argument("--cache-gib", type=float, default=48)
    parser.add_argument("--max-model-len", type=int, default=33024)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--port", type=int, default=19650)
    parser.add_argument("--attention", default="FLASH_ATTN")
    parser.add_argument("--profile-case")
    parser.add_argument("--profile-prefill-only", action="store_true")
    parser.add_argument("--shape-case")
    parser.add_argument("--activity-library", type=Path)
    parser.add_argument("--apc", action="store_true")
    parser.add_argument("--cross-layer-blocks", action="store_true")
    parser.add_argument("--deflect-at", type=int, default=0,
                        help="When P has this many outstanding requests and D is currently clear")
    parser.add_argument("--no-chunk", action="store_true")
    parser.add_argument("--natural", action="store_true")
    args = parser.parse_args()
    cases = [parse_case(case) for case in args.case]
    if args.activity_library and (not args.profile_case or not args.activity_library.is_file()):
        parser.error("The CUPTI worker requires a built library and one --profile-case")
    if args.profile_prefill_only and (args.mode != "pd" or not args.activity_library):
        parser.error("Prefill-only observation requires PD and the CUDA activity library")
    if args.deflect_at < 0 or (args.deflect_at and args.mode != "pd"):
        parser.error("Deflection requires a nonnegative threshold and PD")
    placements = [
        (role, index, tp) for role, count, tp in (
            (("P", args.prefill_workers, args.p_tp), ("D", args.decode_workers, 1))
            if args.mode == "pd" else (("R", args.decode_workers, 1),)
            if args.mode == "dp" else (("PREF", args.prefill_workers, args.p_tp),))
        for index in range(count)
    ]
    if sum(x[2] for x in placements) > len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
        parser.error("Physical GPU lease is smaller than worker placement")
    if any(row["output_tokens"] < 16 or row["input_tokens"] + row["output_tokens"] > args.max_model_len
           for _, _, rows, qps in cases for row in rows) or any(case[-1] <= 0 for case in cases):
        parser.error("Invalid offered load or model geometry")
    args.out.mkdir(parents=True)
    servers, cursor = [], 0
    try:
        for index, (role, member, tp) in enumerate(placements):
            servers.append(launch(args, role, member, cursor, tp, args.port + index))
            cursor += tp
        asyncio.run(run(args, servers, cases))
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
