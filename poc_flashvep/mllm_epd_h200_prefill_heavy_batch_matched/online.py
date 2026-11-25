"""Finite-cohort H200 EPD execution using the verified handoff serving stack."""
import argparse
import asyncio
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

from . import runtime as rt
from .resources import burner_status, experiment_lease

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
    parser.add_argument("--batch-one", action="store_true")
    parser.add_argument("--observe-batch", action="store_true")
    parser.add_argument("--defer-client-raw", action="store_true",
                        help="Flush full client traces after drain using the pinned deferred writer")
    parser.add_argument("--profile", action="store_true",
                        help="One QPS point with clean/profile/clean kernel-only CUPTI collection")
    parser.add_argument("--profile-state-ms", type=int, default=50)
    parser.add_argument("--profile-decode-buffer-gib", type=int, choices=(1, 2), default=1)
    parser.add_argument("--qps", type=float, nargs="+", default=[])
    parser.add_argument("--repeat-main100", type=int, choices=(1, 3), default=1,
                        help="Separate sustained 300-arrival diagnostic with fresh media UUIDs")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--capacity-stage", choices=("encoder", "prefill", "decode-b1", "decode-free"))
    parser.add_argument("--capacity-p-profile", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
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
    rows = rt.read(manifests[rt.MAIN100])["requests"]
    random.Random(ORDER_SEED).shuffle(rows)
    replay = load_replay(args.replay, rows) if args.replay else None
    rows *= args.repeat_main100

    def measure(tag, qps, observe_batch=False):
        if observe_batch:
            rpc(18842, "batch_counter/begin", {"name": tag, "decode_only": True})
        try:
            results = asyncio.run(client.execute(rows, [url], tag, qps, args.seed, replay,
                                                 defer_raw=args.defer_client_raw,
                                                 unique_media_occurrence=args.repeat_main100 > 1,
                                                 metadata={"order_seed": ORDER_SEED, "method": args.mode,
                                                           "batch_one": args.batch_one,
                                                           "Main100_cycles": args.repeat_main100,
                                                           "manifest_SHA256": rt.sha(manifests[rt.MAIN100])}))
        finally:
            if observe_batch:
                batch = rpc(18842, "batch_counter/end", {})
                rt.write(run / "analysis" / f"{tag}_batch.json", batch)
                if args.batch_one:
                    receipt, = batch
                    maxima = receipt["decode_maxima"]
                    rt.require(receipt["valid"] and maxima["scheduled_requests"] == 1
                               and maxima["prepared_decode_requests"] == 1, "Actual D batch1 gate failed")
        ownership_empty(tag, "cuda_ipc")
        result = metrics.summarize(tag)
        print("METRICS", json.dumps(result), flush=True)
        rt.require(len(results) == len(rows) and result["primary_execution_valid"], f"{tag} incomplete")
        return result

    for qps in args.qps:
        tag = f"{args.mode.replace('-', '_')}_{'b1' if args.batch_one else 'free'}_{qps:g}".replace(".", "p")
        if args.repeat_main100 > 1:
            tag += f"_sustained_r{args.repeat_main100}"
        if args.profile or args.observe_batch:
            priming = tag + "_priming"
            asyncio.run(client.execute(rows, [url], priming, qps, args.seed, replay,
                                       defer_raw=args.defer_client_raw,
                                       unique_media_occurrence=args.repeat_main100 > 1,
                                       metadata={"evidence": "WARMUP_EXCLUDED", "order_seed": ORDER_SEED}))
            rt.require(rt.read(run / "state" / f"{priming}_complete.json")["passed"], "Main100 priming failed")
            ownership_empty(priming, "cuda_ipc")
        if args.observe_batch:
            from .trace import COMPARISON_METRICS

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
        from .trace import cpc_gate, profile_session, summarize_trace

        before = measure(tag + "_clean_before", qps)
        profile = tag + "_profile"

        async def instrumented():
            ports = {role: port for role, _, port, _ in rt.ROLES}
            async with profile_session(ports, profile, state_interval_ms=args.profile_state_ms,
                                       decode_buffer_gib=args.profile_decode_buffer_gib) as receipts:
                observed = await asyncio.to_thread(measure, profile, qps)
            return observed, receipts

        observed, receipts = asyncio.run(instrumented())
        after = measure(tag + "_clean_after", qps)
        gate = cpc_gate(before, observed, after, receipts)
        gate["collector_receipts"] = receipts
        rt.write(run / "analysis" / f"{tag}_profile_gate.json", gate)
        client_data = _rows(run / "raw" / f"{profile}_requests.jsonl")
        request_ids = {row["runtime_request_id"] for row in client_data}
        proxy_data = [row for row in _rows(run / "raw/proxy_epd.jsonl") if row["runtime_request_id"] in request_ids]
        trace = summarize_trace(run / "raw/profiles" / profile, client_data, proxy_data)
        rt.write(run / "analysis" / f"{tag}_trace.json", {"evidence": gate["evidence"], **trace})


def main():
    args = parse_args()
    worktree = Path(__file__).resolve().parents[2]
    asset_repo, receipt, run = (path.resolve() for path in (args.asset_repo, args.receipt, args.run_dir))
    rt.require(not run.is_relative_to(asset_repo) and not run.is_relative_to(worktree), "Run directory must be external")
    rt.require(args.wall_seconds > 0 and all(qps > 0 for qps in args.qps)
               and len(set(args.qps)) == len(args.qps), "Invalid budget or QPS points")
    rt.require(args.qps or args.regression or args.capacity, "Select an experiment")
    rt.require(not args.capacity or args.mode == "native-common" and args.replay,
               "Stage capacity requires NativeCommon and original replay counts")
    rt.require(not args.capacity_stage or args.capacity, "Stage selection needs --capacity")
    rt.require(not args.capacity_p_profile or args.capacity and args.capacity_stage == "prefill",
               "Isolated P profiling is a separate stage run")
    rt.require(not args.regression or args.batch_one and (args.mode == "native-common" or args.mode.startswith("static")),
               "Regression covers the B1 NativeCommon and static Token Split controls")
    rt.require(not args.profile or len(args.qps) == 1 and not args.observe_batch,
               "A clean/profile/clean comparison requires one load and no batch observer")
    rt.require(1 <= args.profile_state_ms <= 1000, "Positive sampled observer interval required")
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
        from .trace import build_collector
        build_collector(asset_repo, run, Path(sys.executable).parent.parent)
    rt.write(run / "state/model_download_complete.json", {"directory": str(model), "revision": rt.MODEL_REVISION})
    shutil.copyfile(manifests[rt.CALIBRATION], run / "data" / rt.CALIBRATION)
    if args.regression:
        shutil.copytree(asset_repo / rt.REPORT / "references/quality_sources", run / "references/quality_sources")
    source, settings = configure(worktree, asset_repo, run, deadline)
    rt.write(run / "state/environment.json", {
        "runtime": fingerprint, "source_SHA256": fingerprint_source(source), "receipt_SHA256": rt.sha(receipt),
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
        with experiment_lease(expected, run / "state") as stopped:
            gpus, occupancy = rt.verify_devices(expected)
            rt.write(run / "state/lease.json", {"stopped_burners": stopped, "gpu_inventory": gpus,
                                                 "foreign_gpu_occupancy": occupancy})
            rt.verify_ports()
            ec.mkdir(mode=0o700)
            try:
                token = args.mode in TOKEN_FRACTIONS
                fraction = TOKEN_FRACTIONS.get(args.mode, 0.)
                rt.launch_models(group, args.mode, ec, expected, asset_repo / rt.REPORT / "media",
                                 token=token, common=args.mode != "native-strong")
                url = rt.launch_proxy(group, "epd", "epd", 18847, token=token,
                                      batch_one=args.batch_one, idle=args.mode == "idle10", fraction=fraction)
                if args.regression:
                    from .regression import native_checks, token_checks
                    teacher = rt.launch_proxy(group, "teacher", "e_pd", 18848)
                    calibration = rt.read(manifests[rt.CALIBRATION])["requests"]
                    result = token_checks(calibration, teacher, url, "epd", fraction) if token else native_checks(
                        calibration, teacher, url)
                    rt.write(run / "analysis/regression.json", {"evidence": "CORRECTNESS_PASS", "result": result})
                if args.qps:
                    measured_cohorts(args, run, url, manifests)
                if args.capacity:
                    from .capacity import measure_stages
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
