"""Run a pinned, native vLLM/NIXL text-only PD cache control on three H200 GPUs."""

import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import urllib.request

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from ..mllm_epd_h200_wide_decode_batch_token_split.host import snapshot
from ..mllm_epd_h200_wide_decode_batch_token_split.online import configure, fingerprint_source
from .native_study import ARRIVAL_SEED, GIB, configure_kv, numeric_metrics


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LONG_BENCH_SHA = "c3472d660ffa61988e04205a1dcbb02b0ff7737ece729a1fdd001cb5ba85b904"
PACKAGE = "poc_flashvep.mllm_epd_memory_asymmetry_discovery.llm_pd_control"
BLOCK, ROW_BYTES = 128, 57344
ROSTER_SEED = 20261127
PROXY_PORT = 18847


def validate_model(model):
    config = rt.read(model / "config.json")
    actual = (config["architectures"], config["hidden_size"], config["num_hidden_layers"],
              config["num_attention_heads"], config["num_key_value_heads"], config["torch_dtype"])
    rt.require(actual == (["Qwen2ForCausalLM"], 3584, 28, 28, 4, "bfloat16"), "Wrong Qwen text checkpoint")
    path = model / "model.safetensors.index.json"
    index = rt.read(path)
    shards = {name: (model / name).stat().st_size for name in sorted(set(index["weight_map"].values()))}
    rt.require(len(shards) == 4 and sum(shards.values()) == 15_231_271_888 and
               index["metadata"]["total_size"] == 15_231_233_024, "Wrong Qwen text checkpoint payload")
    rt.require(rt.sha(path) == "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
               "The pinned Qwen safetensors index changed")
    return {"id": MODEL_ID, "revision": MODEL_REVISION, "geometry": actual, "shards": shards,
            "model_config_sha256": rt.sha(model / "config.json"), "weight_index_sha256": rt.sha(path),
            "tokenizer_sha256": rt.sha(model / "tokenizer.json"),
            "weight_payload_bytes": index["metadata"]["total_size"]}


def controlled_roster(manifest, model, workload):
    from transformers import AutoTokenizer

    rt.require(rt.sha(manifest) == LONG_BENCH_SHA, "Original LongBench cohort changed")
    source = rt.read(manifest)["requests"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    by_task = defaultdict(list)
    for row in source:
        by_task[row["source"]].append(row)
    rt.require(len(by_task) == 4 and all(len(rows) == 32 for rows in by_task.values()),
               "Expected 32 original prompts per LongBench task")
    rng = random.Random(ROSTER_SEED)
    chosen = []
    for items in by_task.values():
        rng.shuffle(items)
        chosen.extend(items if workload == "unique128" else items[:16])
    rng.shuffle(chosen)
    loads = [0, 0]
    assignments = {}
    for row in sorted(chosen, key=lambda item: item["total_input_tokens"], reverse=True):
        shard = loads.index(min(loads))
        assignments[row["request_id"]] = shard
        rounded = (row["total_input_tokens"] + BLOCK - 1) // BLOCK * BLOCK
        loads[shard] += rounded * ROW_BYTES
    cold = []
    for index, row in enumerate(chosen):
        token_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        rt.require(len(token_ids) == row["total_input_tokens"] and
                   hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_SHA256"],
                   "Original Qwen prompt or token geometry changed")
        cold.append({"request_id": f"cold/{index:03d}/{row['request_id']}",
                     "source_request_id": row["request_id"], "prompt": row["prompt"],
                     "prompt_SHA256": row["prompt_SHA256"], "total_input_tokens": len(token_ids),
                     "source": row["source"], "stratum": "cold", "image_count": 0,
                     "campaign_shard": assignments[row["request_id"]]})
    warm = []
    for shard in (0, 1):
        prompt = (f"PD control runtime warm-up shard {shard}. " + "Read this initialization text. " * 25).strip()
        warm.append({"request_id": f"warmup/{shard}", "source": "runtime_warmup", "stratum": "warmup",
                     "image_count": 0, "campaign_shard": shard, "prompt": prompt,
                     "prompt_SHA256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "total_input_tokens": len(tokenizer.encode(prompt, add_special_tokens=False))})
    replay = [] if workload == "unique128" else [
        dict(row, request_id=row["request_id"].replace("cold/", "repeat/", 1), stratum="repeat")
        for row in reversed(cold)]
    return cold + replay, warm, loads


def placement(layout):
    if layout == "2p1d":
        labels = (("prefill0", "prefill"), ("prefill1", "prefill"), ("decode0", "decode"))
    else:
        labels = (("prefill0", "prefill"), ("decode0", "decode"), ("decode1", "decode"))
    return [(label, role, 4 + i, 18840 + i, 19840 + i) for i, (label, role) in enumerate(labels)]


def launch(group, model, layout, devices, args):
    args.apc_roles = ("prefill", "decode")
    roles = placement(layout)
    prefill_gpus = [gpu for _, role, gpu, _, _ in roles if role == "prefill"]
    identities, owners = [], defaultdict(list)
    for label, role, gpu, port, side in roles:
        environment = group.run / label
        for directory in ("state", "raw", "unused_ec", "unused_ec/processed_prompts"):
            (environment / directory).mkdir(parents=True)
        pages_gib = args.p_kv_gib if role == "prefill" else args.d_kv_gib
        nixl = {"kv_connector": "NixlConnector", "kv_role": "kv_producer" if role == "prefill" else "kv_consumer"}
        command = [sys.executable, "-m", f"poc_flashvep.{rt.PACKAGE}.server_entry",
                   "--worker-extension-cls", f"poc_flashvep.{rt.PACKAGE}.worker_observer.CampaignWorkerExtension",
                   "--model", str(model), "--served-model-name", MODEL_ID, "--dtype", "bfloat16",
                   "--port", str(port), "--host", "127.0.0.1", "--max-model-len", "32768",
                   "--max-num-batched-tokens", "8192", "--max-num-seqs", "1024", "--block-size", str(BLOCK),
                   "--kv-cache-memory-bytes", str(int(pages_gib * GIB)), "--no-enable-prefix-caching",
                   "--enable-chunked-prefill", "--enable-prompt-tokens-details", "--enable-request-id-headers",
                   "--generation-config", "vllm", "--seed", "2026112401", "--kv-transfer-config", json.dumps(nixl)]
        configure_kv(command, role, args)
        visible_gpus = (gpu,)
        if role == "prefill" and args.tier == "peer":
            peer_gpu, = (candidate for candidate in prefill_gpus if candidate != gpu)
            visible_gpus = (gpu, peer_gpu)
        process = group.start(label, command, CUDA_VISIBLE_DEVICES=",".join(devices[g] for g in visible_gpus),
                              MLLM_EPD_ROLE=role, MLLM_EPD_TOKEN_SPLIT="0", MLLM_EPD_SESSION_ROOT=str(environment),
                              MLLM_EPD_EC_PATH=str(environment / "unused_ec"),
                              MLLM_EPD_METADATA_PATH=str(environment / "unused_ec/processed_prompts"),
                              VLLM_NIXL_SIDE_CHANNEL_PORT=str(side), VLLM_CACHE_ROOT=str(group.run.parent / "llm_runtime_cache" / role))
        identities.append((label, role, port, process))
        for g in visible_gpus:
            owners[devices[g]].append(process.pid)
    for label, role, port, process in identities:
        group.ready(process, port, label, role=role)
    occupancy = rt.nvidia("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    for uuid, known in owners.items():
        pids = [int(row[1]) for row in occupancy if row[0] == uuid]
        rt.require(pids and all(any(rt.pid_owned(pid, owner, group.env["MLLM_EPD_RUN_NONCE"]) for owner in known)
                               for pid in pids), f"Unowned GPU worker on {uuid}")
    rt.write(group.run / "state/gpu_owners.json", occupancy)
    process = group.start("llm_proxy", [sys.executable, "-m", PACKAGE, "proxy", "--layout", layout,
                                       "--log", str(group.run / "raw/proxy.jsonl")])
    group.ready(process, PROXY_PORT, "llm_proxy")


def serve_proxy(layout, log):
    from aiohttp import ClientSession, ClientTimeout, web

    roles = placement(layout)
    prefill_ports = [port for _, role, _, port, _ in roles if role == "prefill"]
    decode_ports = [port for _, role, _, port, _ in roles if role == "decode"]

    async def lifespan(app):
        async with ClientSession(timeout=ClientTimeout(total=180, sock_read=120), trust_env=False) as session:
            app[session_key] = session
            yield

    async def health(_):
        return web.json_response({"name": "llm_proxy"})

    async def complete(request):
        body = await request.json()
        shard = body.pop("campaign_shard")
        prefill, decode = prefill_ports[shard % len(prefill_ports)], decode_ports[shard % len(decode_ports)]
        request_id = request.headers["x-request-id"]
        prefill_body = dict(body, max_tokens=1, stream=False, return_token_ids=False,
                            kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False,
                                                "remote_engine_id": None, "remote_block_ids": None,
                                                "remote_host": None, "remote_port": None})
        prefill_body.pop("stream_options", None)
        trace = {"request_id": request_id, "prefill_port": prefill, "decode_port": decode, "P_submit": time.monotonic()}
        session = request.app[session_key]
        try:
            async with session.post(f"http://127.0.0.1:{prefill}/v1/completions", json=prefill_body,
                                    headers={"x-request-id": f"{request_id}:P"}) as reply:
                result = await reply.json()
                if reply.status != 200:
                    raise RuntimeError(f"Prefill HTTP {reply.status}: {result}")
            trace["P_ready"] = time.monotonic()
            trace["P_usage"] = result["usage"]
            handoff = result.get("kv_transfer_params")
            if handoff is None or handoff.get("remote_block_ids") is None:
                raise RuntimeError("Prefill did not produce its real NIXL handoff")
            trace["P_kv_blocks"] = handoff["remote_block_ids"]
            body["kv_transfer_params"] = handoff
            trace["D_submit"] = time.monotonic()
            async with session.post(f"http://127.0.0.1:{decode}/v1/completions", json=body,
                                    headers={"x-request-id": f"{request_id}:D"}) as reply:
                if reply.status != 200:
                    raise RuntimeError(f"Decode HTTP {reply.status}: {await reply.text()}")
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                async for block in reply.content.iter_any():
                    trace.setdefault("D_first_sse", time.monotonic())
                    await response.write(block)
                await response.write_eof()
            trace["status"] = "STREAM_COMPLETE"
            return response
        except Exception as error:
            trace["status"] = "FAILED_EXECUTION"
            trace["error"] = repr(error)
            return web.json_response({"error": str(error)}, status=502)
        finally:
            trace["finish"] = time.monotonic()
            with log.open("a") as stream:
                stream.write(json.dumps(trace) + "\n")

    session_key = web.AppKey("session", ClientSession)
    app = web.Application()
    app.cleanup_ctx.append(lifespan)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", complete)
    web.run_app(app, host="127.0.0.1", port=PROXY_PORT)


def metrics_snapshot(layout):
    result = {"monotonic": time.monotonic(), "roles": {}}
    for label, _, _, port, _ in placement(layout):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            body = response.read().decode()
        result["roles"][label] = [line for line in body.splitlines() if line.startswith("vllm:")]
    return result


def measurements(run, rows, warm, args):
    client, metrics = (rt.module(name) for name in ("client", "metrics"))
    native_payload = client.payload
    cache_salt = os.environ["MLLM_EPD_RUN_NONCE"]

    def text_payload(row, _nonce, output_count=None, stream=True):
        output = {"model": MODEL_ID, "prompt": row["prompt"], "campaign_shard": row["campaign_shard"],
                  "add_special_tokens": False, "temperature": 0, "top_p": 1, "return_token_ids": True,
                  "ignore_eos": output_count is not None, "max_tokens": output_count or 2048,
                  "stream": stream, "seed": 2026112401, "cache_salt": cache_salt}
        if stream:
            output["stream_options"] = {"include_usage": True}
        return output

    client.payload = text_payload
    url = f"http://127.0.0.1:{PROXY_PORT}"
    try:
        warm_res = asyncio.run(client.execute(warm, [url], "runtime_warmup", 1, ARRIVAL_SEED, 16, serial=True,
                                              metadata={"evidence": "WARMUP_EXCLUDED"}))
        rt.require(len(warm_res) == len(warm) and all(row["status"] == "MEASURED_ONLINE" for row in warm_res),
                   "Warmup failed")
        before = metrics_snapshot(args.layout)
        response = asyncio.run(client.execute(rows, [url], "measured", args.qps, ARRIVAL_SEED, args.output,
                                              defer_raw=True, metadata={"actual_P_APC": True, "actual_D_APC": True,
                                              "layout": args.layout, "P_prefix_tier": args.tier,
                                              "raw_completion_text_path": "chunks[].data.choices[].text"}))
        after = metrics_snapshot(args.layout)
        rt.write(run / "raw/metrics.json", {"before": before, "after": after})
        delta = {}
        a, b = numeric_metrics(before), numeric_metrics(after)
        for role, counters in b.items():
            delta[role] = {key: val - a[role].get(key, 0) for key, val in counters.items()
                           if key.endswith(("_total", "_sum", "_count")) or "offload" in key and "_created" not in key}
        lengths = {row["request_id"]: row.get("output_tokens", 0) for row in response}
        valid = [row for row in response if row["status"] == "MEASURED_ONLINE"]
        expected_proxy_ids = {row["runtime_request_id"] for row in valid}
        for _ in range(20):
            stage_trace = {row["request_id"]: row for row in
                           map(json.loads, (run / "raw/proxy.jsonl").read_text().splitlines())}
            if expected_proxy_ids <= stage_trace.keys():
                break
            time.sleep(.05)
        rt.require(expected_proxy_ids <= stage_trace.keys(), "Proxy journal did not reach client drain")
        phases = {}
        for phase in ("cold", "repeat"):
            matching = [row for row in valid if row["stratum"] == phase]
            if matching:
                p_elapsed = [(stage_trace[row["runtime_request_id"]]["P_ready"] -
                              stage_trace[row["runtime_request_id"]]["P_submit"]) * 1000 for row in matching]
                phases[phase] = {"complete": len(matching), "TTFT_p95_ms": metrics.percentile([row["TTFT_ms"]
                                 for row in matching], 95), "P_HTTP_p95_ms": metrics.percentile(p_elapsed, 95),
                                 "joint_slo": sum(metrics.gates(row)["JOINT_PASS"] for row in matching)}
        rt.write(run / "analysis/control_metrics.json", {"client": metrics.summarize("measured"),
                 "E2E_p95_ms": metrics.percentile([row["E2E_ms"] for row in valid], 95),
                 "joint_slo_goodput_req_s": sum(metrics.gates(row)["JOINT_PASS"] for row in response) /
                     rt.read(run / "state/measured_complete.json")["duration_seconds"],
                 "phases": phases,
                 "prometheus_deltas": delta,
                 "exact_output_sha256": {row["request_id"]: hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest()
                                           for row in valid}})
        rt.require(args.output is None or set(lengths.values()) == {args.output}, "Fixed-output contract failed")
        rt.require(all(row["status"] == "MEASURED_ONLINE" for row in response), "Incomplete LLM PD cohort")
    finally:
        client.payload = native_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] == "proxy":
        parser.add_argument("mode", choices=("proxy",))
        parser.add_argument("--layout", choices=("2p1d", "1p2d"), required=True)
        parser.add_argument("--log", type=Path, required=True)
        arguments = parser.parse_args()
        serve_proxy(arguments.layout, arguments.log)
        return
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("2p1d", "1p2d"), default="2p1d")
    parser.add_argument("--workload", choices=("reuse64", "unique128"), default="reuse64")
    parser.add_argument("--tier", choices=("none", "cpu", "peer"), default="none")
    parser.add_argument("--tier-gib", type=float, default=16)
    parser.add_argument("--p-kv-gib", type=float, default=42)
    parser.add_argument("--d-kv-gib", type=float, default=42)
    parser.add_argument("--tier-guard-gib", type=float, default=24)
    parser.add_argument("--eviction", choices=("lru", "arc"), default="lru")
    parser.add_argument("--store-threshold", type=int, default=0)
    parser.add_argument("--qps", type=float, default=2.5)
    parser.add_argument("--output", type=int, choices=(16, 64, 128, 256), default=16)
    parser.add_argument("--natural-eos", action="store_true")
    parser.add_argument("--wall-seconds", type=int, default=1000)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if min(args.p_kv_gib, args.d_kv_gib, args.qps, args.wall_seconds) <= 0 or args.tier != "none" and (
            args.tier_gib <= 0 or args.layout != "2p1d"):
        parser.error("Positive capacity/time and a 2P/1D layout for external P tiers are required")
    if args.natural_eos:
        args.output = None
    model, manifest, asset, run = map(Path.resolve, (args.model, args.manifest, args.asset_repo, args.run_dir))
    base = Path(__file__).resolve().parents[2]
    rt.require(run.is_relative_to(base.parent) and not run.is_relative_to(base), "Use a fresh, external run directory")
    model_fingerprint = validate_model(model)
    rows, warm, split_bytes = controlled_roster(manifest, model, args.workload)
    run.mkdir(mode=0o700)
    for directory in ("state", "logs", "raw", "analysis", "data", "configs", "references"):
        (run / directory).mkdir()
    rt.write(run / "data/roster.json", {"source_manifest_sha256": rt.sha(manifest), "roster_seed": ROSTER_SEED,
             "model": model_fingerprint, "cold_prompt_rounded_working_set_bytes_per_2p_shard": split_bytes,
             "rows": [{key: value for key, value in row.items() if key != "prompt"} for row in rows]})
    if args.freeze_only:
        return

    rt.git_check(asset)
    inventory = {int(gpu): (uuid, name) for gpu, uuid, name in rt.nvidia("--query-gpu=index,uuid,name")}
    devices = {gpu: inventory[gpu][0] for gpu in (4, 5, 6)}
    rt.require(all(inventory[gpu][1] == "NVIDIA H200" for gpu in devices), "Expected the registered H200 group")
    deadline = time.time() + args.wall_seconds
    source, _ = configure(base, asset, run, deadline, freeze_auxiliary=True)
    group = rt.OwnedProcesses(run, os.environ.copy(), source, deadline)
    rt.write(run / "state/environment.json", {"runtime": rt.runtime_inventory(), "model": model_fingerprint,
             "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
             "authorized_uuid_by_gpu": devices, "source_sha256": rt.sha(__file__),
             "worker_source_sha256": fingerprint_source(source)})

    def deadline_handler(*_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("LLM PD control cancelled or reached its deadline")

    signal.signal(signal.SIGALRM, deadline_handler)
    signal.signal(signal.SIGTERM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
    try:
        rt.write(run / "state/burners_before.json", burner_status())
        rt.write(run / "state/host_before.json", snapshot(topology=True))
        with experiment_lease(devices, run / "state") as stopped:
            gpu_inventory, occupancy = rt.verify_devices(devices)
            rt.write(run / "state/lease.json", {"stopped_burners": stopped, "gpu_inventory": gpu_inventory,
                                                "other_gpu_processes": occupancy})
            rt.verify_ports()
            try:
                launch(group, model, args.layout, devices, args)
                measurements(run, rows, warm, args)
                rt.write(run / "state/workers_after_drain.json", {label: rt.http_json(port, "campaign/worker_info")[0]
                         for label, _, _, port, _ in placement(args.layout)})
            finally:
                group.close()
                rt.verify_devices(devices)
    except BaseException as error:
        rt.write(run / "state/failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rt.write(run / "state/complete.json", {"status": "COMPLETE", "time": time.time()})


if __name__ == "__main__":
    main()
