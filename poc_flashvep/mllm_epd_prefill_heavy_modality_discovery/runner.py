"""Run native H200 EPD capacity and open-loop workload cases on an owned lease."""

import argparse
import asyncio
from contextlib import asynccontextmanager, contextmanager
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

from ..encoder_assisted_decode_attention.native_profile import sample_gpm
from ..mllm_epd_h200_prefill_heavy_batch_matched import capacity, runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import build_collector, rpc
from ..mllm_epd_h200_wide_decode_batch_token_split.host import snapshot
from ..mllm_epd_h200_wide_decode_batch_token_split.online import configure, fingerprint_source
from ..mllm_epd_memory_asymmetry_discovery.native_study import metrics_snapshot
from ..mllm_specific_disaggregation_discovery.native_epd import prompt_messages

PORTS = {role:port for role, _, port, _ in rt.ROLES}
GPU_MAP = {role:gpu for role, gpu, _, _ in rt.ROLES}
SEED = 2026112809
METRIC = re.compile(r'^(vllm:[^ {]+)(\{[^}]+\})? ([^ ]+)')
APPROVED = {"native", "ondemand", "naive", "naive_isolated", "force", "cost", "cost_blind",
            "text", "probe_e", "probe_d"}


def launch_models(group, ec, devices, media, args):
    servers = rt.module("servers")
    pending = []
    for role, gpu, port, side_port in rt.ROLES:
        encoder = role == "encoder"
        command = servers.command(
            role, port, ec, side_port, observe=True, encoder_native_ops=encoder,
            compiled_encoder=encoder, encoder_sync_scheduling=encoder,
            kv_cache_gib=min(args.kv_gib, 42), mm_cache_gb=0,
            renderer_workers=args.encoder_renderers if encoder else 1,
            ec_transport="cuda_ipc", fixed_reduction=False, deterministic_reductions=False,
        )
        def setting(flag, value):
            command[command.index(flag) + 1] = str(value)

        setting("--allowed-local-media-path", media)
        setting("--max-num-seqs", args.max_seqs)
        setting("--worker-extension-cls", f"{__package__}.observer.DiscoveryWorker")
        if not encoder:
            setting("--max-num-batched-tokens", args.p_budget if role == "prefill" else args.d_budget)
            setting("--kv-cache-memory-bytes", args.kv_gib * 1024**3)
            command.extend(["--attention-backend", "FLASH_ATTN"])
        if args.disable_chunked_mm and role == "decode":
            command.append("--disable-chunked-mm-input")
        command.append("--enable-prompt-tokens-details")
        process = group.start(
            role, command, CUDA_VISIBLE_DEVICES=devices[gpu], VLLM_NIXL_SIDE_CHANNEL_PORT=str(side_port),
            MLLM_EPD_ROLE=role, MLLM_EPD_TOKEN_SPLIT="0",
            MLLM_EPD_IMAGE_READY="1" if args.image_ready and encoder else "0",
            MLLM_EPD_LATE_INLINE_REDIRECT="1" if args.inline_reroute and role == "decode" else "0",
            MLLM_EPD_NIXL_READY="1" if args.nixl_ready and role == "decode" else "0",
            MLLM_EPD_LATE_CORRECTNESS="1" if args.logit_check and role == "decode" else "0",
            VLLM_CACHE_ROOT=str(args.cache_root.resolve() / role), MLLM_EPD_EC_PATH=str(ec),
            MLLM_EPD_METADATA_PATH=str(ec / "processed_prompts"))
        pending.append((role, port, gpu, process))
    for role, port, _, process in pending:
        group.ready(process, port, role, role=role)
    verify_workers(group, devices, pending)
    rt.module("native_common_control").set_common_ipc_policy()


def verify_workers(group, devices, pending):
    apps = rt.nvidia("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    for role, _, gpu, process in pending:
        pids = [int(row[1]) for row in apps if row[0] == devices[gpu]]
        rt.require(pids and all(rt.pid_owned(pid, process.pid, group.env["MLLM_EPD_RUN_NONCE"])
                               for pid in pids), f"Unverified GPU ownership for {role} on {gpu}")
    rt.write(group.run / "state/gpu_owners.json", apps)


def launch_dp(group, ec, devices, media, args):
    servers = rt.module("servers")
    models = []
    for index, (_, gpu, port, side_port) in enumerate(rt.ROLES):
        label = f"replica_{index}"
        command = servers.command("agg", port, ec, side_port, observe=True,
                                  kv_cache_gib=min(args.kv_gib, 42), mm_cache_gb=0, renderer_workers=2)
        for flag, value in (("--allowed-local-media-path", media), ("--max-num-seqs", args.max_seqs),
                            ("--max-num-batched-tokens", args.p_budget),
                            ("--kv-cache-memory-bytes", args.kv_gib * 1024**3)):
            command[command.index(flag) + 1] = str(value)
        command.extend(["--attention-backend", "FLASH_ATTN", "--enable-prompt-tokens-details"])
        process = group.start(label, command, CUDA_VISIBLE_DEVICES=devices[gpu], MLLM_EPD_ROLE="agg",
                              VLLM_CACHE_ROOT=str(args.cache_root.resolve() / "colocated"),
                              MLLM_EPD_EC_PATH=str(ec),
                              MLLM_EPD_METADATA_PATH=str(ec / f"unused_replica_{index}"))
        models.append((label, port, gpu, process))
    for label, port, _, process in models:
        group.ready(process, port, label, role="agg")
    verify_workers(group, devices, models)
    return [f"http://127.0.0.1:{port}" for _, port, _, _ in models]


def select_rows(plan, media):
    owned = {}
    manifests = {}
    for case in plan["cases"]:
        file = Path(case["manifest"]).resolve(strict=True)
        if file in manifests:
            continue
        rows = rt.read(file)["requests"]
        rt.require(rows and len({row["request_id"] for row in rows}) == len(rows), f"Invalid roster {file}")
        for row in rows:
            rt.require(len(row["images"]) == row["image_count"], row["request_id"])
            for item in row["images"]:
                path = Path(item["path"]).resolve(strict=True)
                rt.require(path.is_relative_to(media), f"Media outside frozen workspace: {path}")
                if path not in owned:
                    owned[path] = rt.sha(path)
                rt.require(item.get("SHA256", owned[path]) == owned[path], f"Media hash mismatch: {path}")
        manifests[file] = rows
    return manifests


def roster(case, manifests):
    rows = list(manifests[Path(case["manifest"]).resolve()])
    if case.get("shuffle"):
        random.Random(SEED + 1).shuffle(rows)
    n = case.get("count", len(rows))
    expanded = []
    for i in range(n):
        source = rows[i % len(rows)]
        suffix = f"--cycle{i // len(rows)}" if i >= len(rows) else ""
        expanded.append(dict(source, request_id=source["request_id"] + suffix))
    return expanded


def role_queue():
    data = metrics_snapshot()
    result = {}
    for role, lines in data["roles"].items():
        vals = {}
        for line in lines:
            parsed = METRIC.match(line)
            if not parsed:
                continue
            name, labels, count = parsed.groups()
            if not any(key in name for key in ("num_requests", "kv_cache_usage", "num_preemptions")):
                continue
            if name == "vllm:num_requests_waiting_by_reason":
                category = re.search(r'reason="(capacity|deferred)"', labels or "")
                if category is not None:
                    name += ":" + category[1]
            vals[name] = vals.get(name, 0.0) + float(count)
        result[role] = vals
    return {"monotonic": data["monotonic"], "values": result}


@contextmanager
def sample_scheduler(path, period):
    done = threading.Event()
    rows = []

    def work():
        while not done.is_set():
            try:
                rows.append(role_queue())
            except (OSError, ValueError) as error:
                rows.append({"monotonic": time.monotonic(), "error": str(error)})
            done.wait(period)

    thread = threading.Thread(target=work, name="epd-metrics", daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=12)
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")


@asynccontextmanager
async def profile(tag, enabled, copies, period, p_state_only):
    records = {}
    try:
        if enabled:
            for role, port in PORTS.items():
                records[role] = {"begin": await asyncio.to_thread(rpc, port, "profile/begin", {
                    "name": tag, "include_copies": copies,
                    "state_interval_ms": period if role == "prefill" or not p_state_only else 0,
                    "buffer_gib": 1,
                })}
        yield records
    finally:
        results = await asyncio.gather(*(
            asyncio.to_thread(rpc, PORTS[role], "profile/end", {}) for role in records), return_exceptions=True)
        for role, result in zip(records, results, strict=True):
            if isinstance(result, BaseException):
                records[role]["close_error"] = str(result)
            else:
                records[role]["end"] = result


def launch_routes(group, plan):
    wanted = {case.get("route", "native") for case in plan["cases"] if "stage" not in case}
    routes = {"native": rt.launch_proxy(group, "epd", "epd", 18847, decode_cap="free")}
    if wanted - {"native"}:
        program = f"{__package__}.local_route"
        for index, name in enumerate(sorted(wanted - {"native"})):
            port = 18851 + index
            process = group.start(name, [sys.executable, "-m", program, "--name", name, "--port", str(port),
                                  "--p-budget", str(plan.get("p_service_tok_s", 25000)),
                                  "--d-prefill-speed", str(plan.get("d_prefill_tok_s", 23000)),
                                  "--link-speed", str(plan.get("link_bytes_s", 20 * 1024**3)),
                                  "--typical-visual-ratio", str(plan.get("typical_visual_ratio", .75))])
            group.ready(process, port, name)
            routes[name] = f"http://127.0.0.1:{port}"
    return routes


async def text_prefill(run, rows, name, client):
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=360), trust_env=False) as session:
        async def send(index, row):
            body = client.payload(row, f"{name}_{index}", 1, stream=False)
            start = time.monotonic()
            result = await capacity.json_reply(session.post(
                f'http://127.0.0.1:{PORTS["prefill"]}/v1/chat/completions', json=body,
                headers={"x-request-id": f"{name}_{index}:P"}), name)
            tokens = result["usage"]["prompt_tokens"]
            rt.require(tokens == row["total_input_tokens"], "Text control token contract changed")
            return {"request_id": row["request_id"], "submit": start, "finish": time.monotonic(),
                    "input_tokens": tokens, "visual_tokens": 0, "image_count": 0}

        results = await capacity.collect_all(*(send(i, row) for i, row in enumerate(rows)))
    summary = capacity.cohort_metrics(results, "prefill")
    summary.update(stage="prefill", evidence="ISOLATED_GPU_MEASUREMENT",
                   scope="Full text prefill and one internal token, no multimodal or KV export.")
    rt.write(run/"raw"/f"{name}_rows.json", results)
    rt.write(run/"analysis"/f"{name}.json", summary)
    return summary


async def survey(group, plan, manifests, routes, args):
    client = rt.module("client")
    metrics = rt.module("metrics")
    owner = rt.module("ipc_ec_control")
    client.messages = prompt_messages
    run = group.run
    warm = next((row for rows in manifests.values() for row in rows if row["images"]), None)
    if warm is None:
        warm = manifests[next(iter(manifests))][0]
    await client.execute([warm], [routes["native" if warm["images"] else "text"]], "warmup", 1, SEED, 16,
                         serial=True, unique_media_occurrence=True)
    rt.require(rt.read(run / "state/warmup_complete.json")["passed"], "Warmup failed")
    owner.ownership_empty("warmup", "cuda_ipc")
    references = {}
    inline = True
    for case in plan["cases"]:
        rows = roster(case, manifests)
        name = case["tag"]
        route = case.get("route", "native")
        if "teacher_probe_index" in case:
            from .logit_pair import run_pair

            result_path = await asyncio.to_thread(run_pair, run, Path(case["manifest"]),
                                                  case["teacher_probe_index"], name)
            print(name, rt.read(result_path)["counts"], flush=True)
            continue
        if "stage" in case:
            stage = case["stage"]
            rt.write(run / "configs" / f"{name}_workload.json", {
                "cohort_id": case["cohort_id"], "stage": stage, "p_budget": args.p_budget,
                "input_tokens": [row["total_input_tokens"] for row in rows],
                "output": case.get("output"), "ids": [row["request_id"] for row in rows]})
            replay = {row["request_id"]: case.get("output", 16) for row in rows} if stage == "decode" else None
            with sample_gpm(run/"raw"/f"{name}_gpm.json", GPU_MAP, args.gpm_period):
                if stage == "prefill" and all(not row["images"] for row in rows):
                    obj = await text_prefill(run, rows, name, client)
                else:
                    obj = await capacity.measure_stages(run, rows, stage, replay=replay,
                                                        batch_one=case.get("batch_one", False), tag=name)
            if stage == "prefill":
                references[case["cohort_id"]] = obj["requests_per_second"]
            print(name, obj["requests_per_second"], obj["tokens_per_second"], flush=True)
            continue

        use_inline = route == "native" or args.inline_reroute
        if inline != use_inline:
            result, = await asyncio.to_thread(rpc, PORTS["encoder"], "ec_ipc/inline_policy",
                                              {"enabled": use_inline, "consumer_role": "prefill"})
            rt.require(result["inline_role"] == ("prefill" if use_inline else None), "EC policy mismatch")
            inline = use_inline
        arms = ("_clean_before", "_profile", "_clean_after") if case.get("profile") else ("",)
        qps = case.get("qps")
        arrival_trace = rt.read(Path(case["arrival_trace"])) if case.get("arrival_trace") else None
        if arrival_trace is not None:
            qps = arrival_trace["qps"]
        if qps is None:
            qps = round(case["qps_multiplier"] * references[case["cohort_id"]], 3)
            rt.require(qps > 0, f"Invalid calibrated load for {name}")
        if case.get("prime"):
            prime = name + "_prime"
            await client.execute(rows, [routes[route]], prime, qps, SEED, case.get("output"),
                                 defer_raw=True, unique_media_occurrence=True)
            rt.require(rt.read(run/"state"/f"{prime}_complete.json")["passed"], f"{prime} failed")
            owner.ownership_empty(prime, "cuda_ipc")
        for suffix in arms:
            tag = name + suffix
            observed = suffix == "_profile"
            setup = {
                "tag": tag, "group": name, "source": case.get("source", "natural"),
                "cohort_id": case["cohort_id"], "qps": qps, "output": case.get("output"),
                "count": len(rows), "route": route, "observer": "cupti" if observed else "clean",
                "manifest_sha256": rt.sha(case["manifest"]), "copies": bool(observed and case.get("copies")),
                "p_budget": args.p_budget, "d_budget": args.d_budget, "cache_gib": args.kv_gib,
                "ec_policy": "inline" if use_inline else "on_demand",
                "arrival": arrival_trace.get("kind", "observed_e_ready") if arrival_trace else "poisson",
            }
            with (run/"configs/cohorts.jsonl").open("a") as stream:
                stream.write(json.dumps(setup) + "\n")
            before = await asyncio.to_thread(metrics_snapshot)
            async with profile(tag, observed, setup["copies"], args.state_period, args.p_state_only) as receipts:
                with sample_scheduler(run/"raw"/f"{tag}_scheduler.jsonl", args.queue_period), \
                        sample_gpm(run/"raw"/f"{tag}_gpm.json", GPU_MAP, args.gpm_period):
                    metadata = {"method": "NativeStrong" if route == "native" else route,
                                "same_budget_GPU": 3, **setup}
                    if arrival_trace is None:
                        result = await client.execute(
                            rows, [routes[route]], tag, qps, SEED, case.get("output"),
                            defer_raw=True, unique_media_occurrence=True, metadata=metadata)
                    else:
                        from .replay import execute

                        result = await execute(client, rows, routes[route], tag, arrival_trace,
                                               case.get("output"), metadata)
            if observed:
                rt.write(run/"raw"/f"{tag}_profile_receipts.json", receipts)
            summary = metrics.summarize(tag)
            print(tag, len(result), summary.get("gate_attainment"), summary.get("TTFT_p95_ms"), flush=True)
            owner.ownership_empty(tag, "cuda_ipc")
            after = await asyncio.to_thread(metrics_snapshot)
            rt.write(run/"raw"/f"{tag}_prometheus.json", {"before": before, "after": after})
            if args.nixl_ready:
                worker, = await asyncio.to_thread(rt.http_json, PORTS["decode"], "campaign/worker_info", 30)
                rt.write(run/"state"/f"{tag}_nixl_ready.json", worker.get("native_nixl_readiness", {}))


async def survey_dp(group, plan, manifests, urls, args):
    client = rt.module("client")
    metrics = rt.module("metrics")
    client.messages = prompt_messages
    run = group.run
    warm = next(row for rows in manifests.values() for row in rows if row["images"])
    for i, url in enumerate(urls):
        await client.execute([warm], [url], f"warmup_replica_{i}", 1, SEED, 16, serial=True)
    for case in plan["cases"]:
        rows = roster(case, manifests)
        tag = case["tag"]
        setup = {
            "tag": tag, "group": case.get("group", tag), "source": case.get("source", "natural"),
            "cohort_id": case["cohort_id"], "qps": case["qps"], "output": case.get("output"),
            "count": len(rows), "route": "dp3", "observer": "clean", "arrival": "poisson",
            "manifest_sha256": rt.sha(case["manifest"]), "p_budget": args.p_budget,
            "d_budget": args.p_budget, "cache_gib": args.kv_gib, "assignment": "round_robin",
        }
        with (run/"configs/cohorts.jsonl").open("a") as stream:
            stream.write(json.dumps(setup) + "\n")
        devices = {f"replica_{i}": gpu for i, (_, gpu, _, _) in enumerate(rt.ROLES)}
        with sample_gpm(run/"raw"/f"{tag}_gpm.json", devices, args.gpm_period):
            result = await client.execute(rows, urls, tag, case["qps"], SEED, case.get("output"),
                                          defer_raw=True, unique_media_occurrence=True, metadata=setup)
        summary = metrics.summarize(tag)
        print(tag, len(result), summary.get("gate_attainment"), summary.get("TTFT_p95_ms"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--asset-repo", type=Path, default=Path("../MLLM-EP-mllm-epd"))
    parser.add_argument("--receipt", type=Path,
        default=Path("../mllm_epd_autosetup/download_ready_1795600931728841878.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("../h200_memory_asymmetry_runs/epd_runtime_cache"))
    parser.add_argument("--p-budget", type=int, default=8192)
    parser.add_argument("--d-budget", type=int, default=8192)
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--kv-gib", type=int, choices=(42, 48), default=42)
    parser.add_argument("--encoder-renderers", type=int, default=2)
    parser.add_argument("--wall-seconds", type=int, default=2700)
    parser.add_argument("--state-period", type=int, default=100)
    parser.add_argument("--p-state-only", action="store_true", help="Sample scheduler state on P only")
    parser.add_argument("--gpm-period", type=float, default=.25)
    parser.add_argument("--queue-period", type=float, default=.22)
    parser.add_argument("--disable-chunked-mm", action="store_true")
    parser.add_argument("--image-ready", action="store_true")
    parser.add_argument("--nixl-ready", action="store_true")
    parser.add_argument("--logit-check", action="store_true")
    parser.add_argument("--inline-reroute", action="store_true",
                        help="Allow the single D recipient to claim P-inline descriptors")
    parser.add_argument("--layout", choices=("epd", "dp3"), default="epd")
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    media = base.parent
    run, assets, receipt = map(Path.resolve, (args.run_dir, args.asset_repo, args.receipt))
    rt.require(run.is_relative_to(media) and not run.is_relative_to(base), "Use a fresh external run root")
    plan = rt.read(args.plan)
    rt.require(plan.get("cases") and all(case.get("route", "native") in APPROVED for case in plan["cases"]),
               "Nonempty plan with supported routes required")
    if args.layout == "dp3":
        rt.require(all("qps" in case and not any(case.get(flag) for flag in ("stage", "arrival_trace", "profile"))
                       for case in plan["cases"]), "DP3 requires explicit clean open-loop request cases")
    teacher = ["teacher_probe_index" in case for case in plan["cases"]]
    rt.require(not any(teacher) or (all(teacher) and args.logit_check and args.layout == "epd"),
               "Use an independent logit-check EPD epoch for a teacher-forced probe")
    rt.git_check(assets)
    model, _ = rt.verify_inputs(assets, receipt)
    manifests = select_rows(plan, media)
    available = {int(gpu):(uuid, kind) for gpu, uuid, kind in rt.nvidia("--query-gpu=index,uuid,name")}
    devices = {gpu: available[gpu][0] for gpu in GPU_MAP.values()}
    rt.require(all(available[gpu][1] == "NVIDIA H200" for gpu in devices), "Expected H200s")
    deadline = time.time() + args.wall_seconds
    run.mkdir(mode=0o700)
    for entry in ("data", "state", "logs", "raw", "analysis", "configs", "references"):
        (run / entry).mkdir()
    rt.write(run/"configs/frozen_plan.json", plan)
    rt.write(run/"state/model_download_complete.json", {"directory": str(model), "revision": rt.MODEL_REVISION})
    source, settings = configure(base, assets, run, deadline, freeze_auxiliary=True)
    rt.write(run/"state/environment.json", {
        "runtime": rt.runtime_inventory(), "worker_source": fingerprint_source(source),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base, text=True).strip(),
        "runner_sha256": rt.sha(__file__), "manifest_sha256": {str(f): rt.sha(f) for f in manifests},
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "authorized_uuid_by_gpu": devices,
    })
    if any(case.get("profile") for case in plan["cases"]):
        build_collector(assets, run, Path(sys.executable).parent.parent)
    ec = Path("/dev/shm") / f"epd_modality_{settings['MLLM_EPD_RUN_NONCE']}"
    group = rt.OwnedProcesses(run, os.environ.copy(), source, deadline)

    def cancel(*_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("Experiment stopped or exceeded its authorized window")

    signal.signal(signal.SIGALRM, cancel)
    signal.signal(signal.SIGTERM, cancel)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
    try:
        rt.write(run/"state/burners_before.json", burner_status())
        rt.write(run/"state/host_before.json", snapshot(topology=True))
        with experiment_lease(devices, run/"state") as stopped:
            gpu_inventory, occupancy = rt.verify_devices(devices)
            rt.write(run/"state/lease.json", {"stopped": stopped, "inventory": gpu_inventory, "remaining": occupancy})
            rt.verify_ports()
            ec.mkdir(mode=0o700)
            try:
                if args.layout == "dp3":
                    urls = launch_dp(group, ec, devices, media, args)
                    asyncio.run(survey_dp(group, plan, manifests, urls, args))
                else:
                    launch_models(group, ec, devices, media, args)
                    if any(case.get("route") in ("probe_e", "probe_d") for case in plan["cases"]):
                        rt.require(not any(case.get("profile") for case in plan["cases"]),
                                   "The one-role CUPTI trace is not a device-wide E co-residency trace")
                        from .assist import launch_helper

                        launch_helper(group, ec, devices[GPU_MAP["encoder"]], media, args)
                    routes = launch_routes(group, plan)
                    asyncio.run(survey(group, plan, manifests, routes, args))
            finally:
                group.close()
                shutil.rmtree(ec)
                rt.verify_devices(devices)
    except BaseException as ex:
        rt.write(run/"state/failure.json", {"type": type(ex).__name__, "error": str(ex)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rt.write(run/"state/complete.json", {"status":"COMPLETE", "time":time.time()})


if __name__ == "__main__":
    main()
