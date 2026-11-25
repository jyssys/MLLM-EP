"""Finite-cohort H200 EPD execution using the verified handoff serving stack."""
import argparse
import asyncio
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import signal
import subprocess
import sys
import time

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from .host import sample_nvml, snapshot

TOKEN_FRACTIONS = {"static5": .05, "static10": .10, "static15": .15, "static20": .20, "idle10": .10}
MODES = ("native-strong", "native-common", *TOKEN_FRACTIONS)
SEED = 2026112402
ORDER_SEED = 2026112404


def configure(worktree, asset_repo, run, deadline):
    source = rt.prepare_overlay(asset_repo, run, research=worktree)
    runtime = Path(sys.executable).parent.parent
    compilers = runtime / 'bin'
    nvidia = runtime / 'lib/python3.14/site-packages/nvidia'
    headers = ' '.join(f"-I{nvidia / name / 'include'}" for name in ('curand', 'cublas', 'cusparse', 'cusolver'))
    settings = dict(
        PATH=str(compilers) + os.pathsep + os.environ["PATH"], PYTHONPATH=str(source),
        PYTHONDONTWRITEBYTECODE='1', CC=str(compilers / 'x86_64-conda-linux-gnu-cc'),
        CXX=str(compilers / 'x86_64-conda-linux-gnu-c++'),
        CUDA_HOME=str(runtime / 'targets/x86_64-linux'), FLASHINFER_NVCC=str(compilers / 'nvcc'),
        NVCC_CCBIN=str(compilers / 'x86_64-conda-linux-gnu-c++'),
        FLASHINFER_WORKSPACE_BASE=str(run.parent / 'flashinfer_workspace'),
        FLASHINFER_EXTRA_CFLAGS=headers, FLASHINFER_EXTRA_CUDAFLAGS=headers,
        MLLM_EPD_PINNED_REPO=str(asset_repo), MLLM_EPD_SESSION_ROOT=str(run),
        MLLM_EPD_SESSION_PYTHON=sys.executable, MLLM_EPD_SESSION_DEADLINE=str(deadline),
        MLLM_EPD_RUN_NONCE=secrets.token_hex(12), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VLLM_BATCH_INVARIANT='0', MLLM_EPD_EC_TRANSPORT='cuda_ipc', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', UCX_NET_DEVICES='all',
        VLLM_ALLOW_LONG_MAX_MODEL_LEN='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
    )
    for key in ('PYTHONHOME', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                'VLLM_USE_V2_MODEL_RUNNER', 'MLLM_EPD_API_SWITCH_INTERVAL'):
        os.environ.pop(key, None)
    os.environ.update(settings)
    sys.path.insert(0, str(source))
    sys.modules["poc_flashvep"].__path__.insert(0, str(source / "poc_flashvep"))
    sys.dont_write_bytecode = True
    return source, settings


def fingerprint_source(source):
    package = source / "poc_flashvep" / rt.PACKAGE
    return {file.name: rt.sha(file) for file in sorted(package.glob("*.py"))}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--decode-cap", choices=("1","2","4","8","16","32","64","free"), default="free")
    parser.add_argument("--all-burners-off", action="store_true")
    parser.add_argument("--wide-capacity", action="store_true")
    parser.add_argument("--capacity-caps", choices=("1", "2", "4", "8", "16", "32", "64", "free"),
                        nargs="+", help="Repeat selected D-only arms after an incomplete full sweep")
    parser.add_argument("--sanity", choices=("stock", "native"))
    parser.add_argument("--observe-batch", action="store_true")
    parser.add_argument("--screen", action="store_true",
                        help="Keep clean latency and one separate diagnostic batch-counter cohort per load")
    parser.add_argument("--defer-client-raw", action="store_true",
                        help="Flush full client traces after drain using the pinned deferred writer")
    parser.add_argument("--profile", action="store_true",
                        help="One QPS point with clean/profile/clean kernel-only CUPTI collection")
    parser.add_argument("--short-roi", action="store_true",
                        help="Begin just before the fixed central arrival window and flush after drain")
    parser.add_argument("--profile-state-ms", type=int, default=50)
    parser.add_argument("--profile-decode-buffer-gib", type=int, choices=(1, 2), default=1)
    parser.add_argument("--qps", type=float, nargs="+", default=[])
    parser.add_argument("--subset", choices=("all", "discovery", "heldout"), default="all")
    parser.add_argument("--repeat-main100", type=int, choices=(1, 3), default=1,
                        help="Separate sustained 300-arrival diagnostic with fresh media UUIDs")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--capacity-stage", choices=("encoder", "prefill", "decode-b1", "decode-free"))
    parser.add_argument("--capacity-p-profile", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--paired-start-id", help="Pair only independent arms assigned to the same block")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--wall-seconds", type=int, default=3600)
    return parser.parse_args()


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_replay(path, rows):
    from poc_flashvep.mllm_epd_prefill_heavy_token_split import client
    if path.suffix == ".csv":
        with path.open() as stream:
            records = list(csv.DictReader(stream))
        counts = {row["request_id"]: int(row["natural_output_tokens"]) for row in records}
        rt.require(len(records) == len(counts), "Duplicate replay request ID")
    else:
        counts = client.load_replay_counts(path)
    rt.require(set(counts) == {row["request_id"] for row in rows} and sum(counts.values()) == 5269,
               "Replay must match all Main100 requests and 5269 original tokens")
    return counts


def measured_cohorts(args, run, url, manifests):
    from poc_flashvep.mllm_epd_prefill_heavy_token_split import client, metrics
    from poc_flashvep.mllm_epd_prefill_heavy_token_split.ipc_ec_control import ownership_empty
    from poc_flashvep.mllm_epd_prefill_heavy_token_split.llm_probe_control import rpc

    warm_rows = rt.read(manifests[rt.CALIBRATION])["requests"][:1]
    asyncio.run(client.execute(warm_rows, [url], "warmup", 1, args.seed, 16, serial=True,
                               metadata={"evidence": "WARMUP_EXCLUDED"}))
    rt.require(rt.read(run / "state/warmup_complete.json")["passed"], "Online warmup failed")
    ownership_empty("warmup", "cuda_ipc")
    manifest = rt.read(manifests[rt.MAIN100])
    rows = manifest["requests"]
    replay = load_replay(args.replay, rows) if args.replay else None
    if args.subset != "all":
        selected = set(manifest[args.subset + "_ids"])
        rows = [row for row in rows if row["request_id"] in selected]
        rt.require(len(rows) == 50 and len(selected) == 50, "Frozen subset is incomplete")
    random.Random(ORDER_SEED).shuffle(rows)
    rows *= args.repeat_main100

    def measure(tag, qps, observe_batch=False, timing_hook=None):
        if observe_batch:
            rpc(18842, "batch_counter/begin", {"name": tag, "decode_only": True})
        try:
            telemetry = sample_nvml(run / "raw" / f"{tag}_nvml.json", {role: gpu for role, gpu, _, _ in rt.ROLES})
            with telemetry if args.screen else nullcontext():
                results = asyncio.run(client.execute(rows, [url], tag, qps, args.seed, replay,
                                                     defer_raw=args.defer_client_raw,
                                                     unique_media_occurrence=args.repeat_main100 > 1,
                                                     timing_hook=timing_hook,
                                                     metadata={"order_seed": ORDER_SEED, "method": args.mode,
                                                               "batch_one": args.decode_cap == "1", "decode_cap": args.decode_cap,
                                                               "Main100_cycles": args.repeat_main100, "subset": args.subset,
                                                               "paired_start_id": args.paired_start_id,
                                                               "nvml_sampling": args.screen,
                                                               "manifest_SHA256": rt.sha(manifests[rt.MAIN100])}))
        finally:
            if observe_batch:
                batch = rpc(18842, "batch_counter/end", {})
                rt.write(run / "analysis" / f"{tag}_batch.json", batch)
                if args.decode_cap != "free":
                    receipt, = batch
                    maxima = receipt["decode_maxima"]
                    rt.require(receipt["valid"] and 0 < maxima["scheduled_requests"] <= int(args.decode_cap)
                               and 0 < maxima["prepared_decode_requests"] <= int(args.decode_cap),
                               "Actual decode batch exceeds admission cap or was not observed")
        ownership_empty(tag, "cuda_ipc")
        result = metrics.summarize(tag)
        print("METRICS", json.dumps(result), flush=True)
        rt.require(len(results) == len(rows) and result["primary_execution_valid"], f"{tag} incomplete")
        return result

    for qps in args.qps:
        tag = f"{args.mode.replace('-', '_')}_{'free' if args.decode_cap == 'free' else 'b' + args.decode_cap}_{qps:g}".replace(".", "p")
        if args.repeat_main100 > 1:
            tag += f"_sustained_r{args.repeat_main100}"
        if args.screen:
            measure(tag, qps)
            measure(tag + "_observed", qps, observe_batch=True)
            continue
        if args.profile or args.observe_batch:
            priming = tag + "_priming"
            asyncio.run(client.execute(rows, [url], priming, qps, args.seed, replay,
                                       defer_raw=args.defer_client_raw,
                                       unique_media_occurrence=args.repeat_main100 > 1,
                                       metadata={"evidence": "WARMUP_EXCLUDED", "order_seed": ORDER_SEED}))
            rt.require(rt.read(run / "state" / f"{priming}_complete.json")["passed"], "Main100 priming failed")
            ownership_empty(priming, "cuda_ipc")
        if args.observe_batch:
            from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import COMPARISON_METRICS

            before = measure(tag + "_batch_clean_before", qps)
            observed = measure(tag + "_observed", qps, observe_batch=True)
            after = measure(tag + "_batch_clean_after", qps)
            changes = {}
            for key in COMPARISON_METRICS:
                center = (before[key] + after[key]) / 2
                changes[key] = {"observed_relative_deviation": abs(observed[key] / center - 1),
                                "clean_relative_spread": abs(after[key] - before[key]) / center}
            passed = all(max(value.values()) <= .05 for value in changes.values())
            rt.write(run / "analysis" / f"{tag}_batch_qualification.json",
                     {"passed": passed, "threshold": .05, "metrics": changes,
                      "evidence": "TRACE_ANALYSIS" if passed else "DIAGNOSTIC"})
            continue
        if not args.profile:
            measure(tag, qps)
            continue
        from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import cpc_gate, profile_session, summarize_trace

        before = measure(tag + "_clean_before", qps)
        rt.write(run / "state" / f"{tag}_host_before_profile.json", snapshot())
        profile = tag + "_profile"

        async def instrumented():
            ports = {role: port for role, _, port, _ in rt.ROLES}
            async with profile_session(ports, profile, state_interval_ms=args.profile_state_ms,
                                       decode_buffer_gib=args.profile_decode_buffer_gib) as receipts:
                observed = await asyncio.to_thread(measure, profile, qps)
            return observed, receipts

        if args.short_roi:
            from .profile import short_roi_hook
            hook, receipts = short_roi_hook(profile, state_interval_ms=args.profile_state_ms,
                                            decode_buffer_gib=args.profile_decode_buffer_gib)
            observed = measure(profile, qps, timing_hook=hook)
        else:
            observed, receipts = asyncio.run(instrumented())
        after = measure(tag + "_clean_after", qps)
        rt.write(run / "state" / f"{tag}_host_after_profile.json", snapshot())
        gate = cpc_gate(before, observed, after, receipts)
        gate["collector_receipts"] = receipts
        rt.write(run / "analysis" / f"{tag}_profile_gate.json", gate)
        client_data = _rows(run / "raw" / f"{profile}_requests.jsonl")
        request_ids = {row["runtime_request_id"] for row in client_data}
        proxy_data = [row for row in _rows(run / "raw/proxy_epd.jsonl") if row["runtime_request_id"] in request_ids]
        trace = summarize_trace(run / "raw/profiles" / profile, client_data, proxy_data)
        rt.write(run / "analysis" / f"{tag}_trace.json", {"evidence": gate["evidence"], **trace})


def calibration_sanity(url, manifests, seed):
    from poc_flashvep.mllm_epd_prefill_heavy_token_split import client, metrics

    rows = rt.read(manifests[rt.CALIBRATION])["requests"]
    asyncio.run(client.execute(rows[:1], [url], "sanity_warmup", 1, seed, None, serial=True))
    result = asyncio.run(client.execute(rows, [url], "calibration32_sanity", 1, seed, None, serial=True,
                                        metadata={"scope": "Natural EOS serial identical Calibration32"}))
    summary = metrics.summarize("calibration32_sanity")
    rt.require(len(result) == 32 and summary["primary_execution_valid"], "Calibration32 sanity incomplete")
    return summary


def run_stock_sanity(group, expected, media, manifests, seed):
    servers = rt.module("servers")
    role, gpu, _, _ = rt.ROLES[1]
    port = 18847
    command = servers.command("agg", port, Path("/dev/shm"), 19847)
    command[command.index("--allowed-local-media-path") + 1] = str(media)
    process = group.start("stock_sanity", command, CUDA_VISIBLE_DEVICES=expected[gpu],
                          VLLM_CACHE_ROOT=str(group.run / "runtime_cache/stock"))
    while time.time() < group.deadline:
        rt.require(process.poll() is None, "Stock vLLM exited during startup")
        try:
            if rt.http_json(port, "v1/models"):
                break
        except OSError:
            time.sleep(1)
    else:
        raise TimeoutError("Stock vLLM startup expired")
    gpu_processes = [int(row[1]) for row in rt.nvidia("--query-compute-apps=gpu_uuid,pid")
                     if row[0] == expected[gpu]]
    rt.require(gpu_processes and all(rt.pid_owned(pid, process.pid, group.env["MLLM_EPD_RUN_NONCE"])
                                     for pid in gpu_processes), "Stock worker ownership failed")
    calibration_sanity(f"http://127.0.0.1:{port}", manifests, seed)


def main():
    args = parse_args()
    worktree = Path(__file__).resolve().parents[2]
    asset_repo, receipt, run = (path.resolve() for path in (args.asset_repo, args.receipt, args.run_dir))
    rt.require(not run.is_relative_to(asset_repo) and not run.is_relative_to(worktree), "Run directory must be external")
    rt.require(args.wall_seconds > 0 and all(qps > 0 for qps in args.qps)
               and len(set(args.qps)) == len(args.qps), "Invalid budget or QPS points")
    rt.require(args.qps or args.regression or args.capacity or args.sanity or args.wide_capacity, "Select an experiment")
    rt.require(not args.capacity or args.mode == "native-common" and args.replay,
               "Stage capacity requires NativeCommon and original replay counts")
    rt.require(not args.capacity_stage or args.capacity, "Stage selection needs --capacity")
    rt.require(not args.capacity_p_profile or args.capacity and args.capacity_stage == "prefill",
               "Isolated P profiling is a separate stage run")
    rt.require(not args.regression or (args.mode == "native-common" or args.mode.startswith("static")),
               "Regression covers NativeCommon and static Token Split controls")
    rt.require(not args.profile or len(args.qps) == 1 and not args.observe_batch,
               "A clean/profile/clean comparison requires one load and no batch observer")
    rt.require(1 <= args.profile_state_ms <= 1000, "Positive sampled observer interval required")
    rt.require(not args.short_roi or args.profile, "Short ROI requires --profile")
    rt.require(args.repeat_main100 == 1 or args.subset == "all", "Sustained run needs all Main100")
    rt.require(not args.wide_capacity or args.mode == "native-common" and args.replay, "Wide capacity requires NativeCommon replay")
    rt.require(not args.capacity_caps or args.wide_capacity, "--capacity-caps needs --wide-capacity")
    rt.require(not args.sanity or not (args.qps or args.capacity or args.wide_capacity or args.regression),
               "Calibration sanity is a separate worker epoch")
    rt.git_check(asset_repo)
    fingerprint = rt.runtime_inventory()
    model, manifests = rt.verify_inputs(asset_repo, receipt)
    available = {int(gpu): (uuid, name) for gpu, uuid, name in rt.nvidia("--query-gpu=index,uuid,name")}
    expected = {gpu: available[gpu][0] for _, gpu, _, _ in rt.ROLES}
    rt.require(all(available[gpu][1] == "NVIDIA H200" for gpu in expected), "H200 group changed")
    deadline = time.time() + args.wall_seconds
    run.mkdir(mode=0o700)
    for directory in ("state", "logs", "raw", "analysis", "data", "configs", "references"):
        (run / directory).mkdir()
    if args.profile or args.capacity_p_profile:
        from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import build_collector
        build_collector(asset_repo, run, Path(sys.executable).parent.parent)
    rt.write(run / "state/model_download_complete.json", {"directory": str(model), "revision": rt.MODEL_REVISION})
    shutil.copyfile(manifests[rt.CALIBRATION], run / "data" / rt.CALIBRATION)
    if args.regression:
        shutil.copytree(asset_repo / rt.REPORT / "references/quality_sources", run / "references/quality_sources")
    source, settings = configure(worktree, asset_repo, run, deadline)
    rt.write(run / "state/environment.json", {
        "runtime": fingerprint, "source_SHA256": fingerprint_source(source), "receipt_SHA256": rt.sha(receipt),
        "runner_SHA256": {file.name: rt.sha(file) for file in sorted(Path(__file__).parent.glob("*.py"))},
        "git_head": subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip(),
        "authorized_uuid_by_gpu": expected, "settings": settings, "argv": sys.argv, "deadline": deadline,
    })
    ec = Path("/dev/shm") / f"h200_epd_{settings['MLLM_EPD_RUN_NONCE']}"
    group = rt.OwnedProcesses(run, os.environ.copy(), source, deadline)

    def timeout(*_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("H200 run cancelled or expired")

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
    try:
        rt.write(run / "state/burners_before.json", burner_status())
        rt.write(run / "state/host_before_lease.json", snapshot(topology=True))
        scope = {gpu: info[0] for gpu, info in available.items()} if args.all_burners_off else expected
        with experiment_lease(scope, run / "state") as stopped:
            gpus, occupancy = rt.verify_devices(expected)
            rt.write(run / "state/lease.json", {"stopped_burners": stopped, "gpu_inventory": gpus,
                                                 "foreign_gpu_occupancy": occupancy})
            rt.write(run / "state/host_in_lease.json", snapshot())
            rt.verify_ports()
            ec.mkdir(mode=0o700)
            try:
                if args.sanity == "stock":
                    run_stock_sanity(group, expected, asset_repo / rt.REPORT / "media", manifests, args.seed)
                    rt.write(run / "state/complete.json", {"status": "COMPLETE", "time": time.time()})
                    return
                token = args.mode in TOKEN_FRACTIONS
                fraction = TOKEN_FRACTIONS.get(args.mode, 0.)
                rt.launch_models(group, args.mode, ec, expected, asset_repo / rt.REPORT / "media",
                                 token=token, common=args.mode != "native-strong",
                                 prefill_kv_ttl=180 if args.wide_capacity else None)
                url = rt.launch_proxy(group, "epd", "epd", 18847, token=token,
                                      decode_cap=args.decode_cap, idle=args.mode == "idle10", fraction=fraction)
                if args.regression:
                    from ..mllm_epd_h200_prefill_heavy_batch_matched.regression import native_checks, token_checks
                    teacher = rt.launch_proxy(group, "teacher", "e_pd", 18848)
                    calibration = rt.read(manifests[rt.CALIBRATION])["requests"]
                    result = token_checks(calibration, teacher, url, "epd", fraction) if token else native_checks(
                        calibration, teacher, url)
                    rt.write(run / "analysis/regression.json", {"evidence": "CORRECTNESS_PASS", "result": result})
                if args.sanity == "native":
                    calibration_sanity(url, manifests, args.seed)
                if args.wide_capacity:
                    from .decode_capacity import CAPS, measure_decode_sweep
                    rows = rt.read(manifests[rt.MAIN100])["requests"]
                    random.Random(ORDER_SEED).shuffle(rows)
                    asyncio.run(measure_decode_sweep(run, rows, load_replay(args.replay, rows),
                                                     caps=args.capacity_caps or CAPS, lease_seconds=180))
                if args.qps:
                    measured_cohorts(args, run, url, manifests)
                if args.capacity:
                    from ..mllm_epd_h200_prefill_heavy_batch_matched.capacity import measure_stages
                    rows = rt.read(manifests[rt.MAIN100])["requests"]
                    random.Random(ORDER_SEED).shuffle(rows)
                    counts = load_replay(args.replay, rows)
                    stages = {"encoder": ("encoder", False), "prefill": ("prefill", False),
                              "decode-b1": ("decode", True), "decode-free": ("decode", False)}
                    for stage, batch_one in (stages[name] for name in ([args.capacity_stage]
                                                                       if args.capacity_stage else stages)):
                        asyncio.run(measure_stages(run, rows, stage, replay=counts, batch_one=batch_one,
                                                   profile=args.capacity_p_profile))
            finally:
                group.close()
                rt.verify_devices(expected)
                shutil.rmtree(ec)
    except BaseException as exc:
        rt.write(run / "state/failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rt.write(run / "state/complete.json", {"status": "COMPLETE", "time": time.time()})


if __name__ == "__main__":
    main()
