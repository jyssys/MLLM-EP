"""Own, run and clean up an isolated Common H200 suffix-calibration epoch."""
import argparse
import asyncio
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from ..mllm_epd_h200_wide_decode_batch_token_split.host import snapshot
from ..mllm_epd_h200_wide_decode_batch_token_split.online import configure, fingerprint_source
from .calibration import INLINE_REDIRECT, ON_DEMAND, measure


INVARIANCE_ROLES = {
    "bi0": (), "bi1": ("prefill", "decode"), "bi1-p": ("prefill",), "bi1-d": ("decode",),
    "bi1-qsplit4": ("prefill", "decode"), "eager": (),
}


def run_calibration(args, *, measure_epoch=measure, decode_environment=None, tag="suffix-calibration",
                    freeze_auxiliary=False, numeric_regime="bi0", preissue_crossing=False):
    worktree = Path(__file__).resolve().parents[2]
    asset_repo, receipt, run = (path.resolve() for path in (args.asset_repo, args.receipt, args.run_dir))
    rt.require(not run.is_relative_to(asset_repo) and not run.is_relative_to(worktree), "Run directory must be external")
    rt.require(args.repeats > 0 and args.wall_seconds > 0, "Repetitions and wall budget must be positive")
    rt.require(numeric_regime in INVARIANCE_ROLES, "Unknown isolated calibration numerical regime")
    rt.require(not preissue_crossing or args.inline_redirect, "Crossing preissue requires inline redirect")
    rt.git_check(asset_repo)
    fingerprint = rt.runtime_inventory()
    transport = INLINE_REDIRECT if args.inline_redirect else ON_DEMAND
    model, manifests = rt.verify_inputs(asset_repo, receipt)
    available = {int(gpu): (uuid, name) for gpu, uuid, name in rt.nvidia("--query-gpu=index,uuid,name")}
    expected = {gpu: available[gpu][0] for _, gpu, _, _ in rt.ROLES}
    rt.require(all(available[gpu][1] == "NVIDIA H200" for gpu in expected), "GPU type mismatch")
    deadline = time.time() + args.wall_seconds
    run.mkdir(mode=0o700)
    for directory in ("state", "logs", "raw", "analysis", "data", "configs", "references"):
        (run / directory).mkdir()
    rt.write(run / "state/model_download_complete.json", {"directory": str(model), "revision": rt.MODEL_REVISION})
    shutil.copyfile(manifests[rt.CALIBRATION], run / "data" / rt.CALIBRATION)
    source, settings = configure(worktree, asset_repo, run, deadline, freeze_auxiliary=freeze_auxiliary)
    decode_env = dict(decode_environment or {})
    if args.inline_redirect:
        decode_env["MLLM_EPD_LATE_INLINE_REDIRECT"] = "1"
    role_environment = {"decode": decode_env}
    for role in INVARIANCE_ROLES[numeric_regime]:
        role_environment.setdefault(role, {})["VLLM_BATCH_INVARIANT"] = "1"
    if numeric_regime == "bi1-qsplit4":
        for role in INVARIANCE_ROLES[numeric_regime]:
            role_environment[role]["MLLM_EPD_BI1_D_SPLITS"] = "4"
    rt.write(run / "state/environment.json", {
        "runtime": fingerprint, "transport": transport, "receipt_sha256": rt.sha(receipt),
        "manifest_sha256": {name: rt.sha(path) for name, path in manifests.items()},
        "source_sha256": fingerprint_source(source),
        "runner_sha256": {file.name: rt.sha(file) for file in sorted(Path(__file__).parent.glob("*.py"))},
        "decode_environment": decode_environment or {}, "numeric_regime": numeric_regime,
        "role_environment": role_environment, "eager_pd": numeric_regime == "eager",
        "crossing_preissue_envelope": {"max_suffix_fraction": .35, "max_suffix_tokens": 0}
        if preissue_crossing else None,
        "git_head": subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip(),
        "authorized_uuid_by_gpu": expected, "settings": settings, "argv": sys.argv, "deadline": deadline,
    })
    ec = Path("/dev/shm") / f"h200_epd_{settings['MLLM_EPD_RUN_NONCE']}"
    workers = os.environ.copy()
    if preissue_crossing:
        from ..mllm_epd_prefill_heavy_token_split.crossing_ipc import MAX_FRACTION_ENV, MAX_TOKENS_ENV, PREISSUE_ENV

        workers.update({PREISSUE_ENV: "1", MAX_FRACTION_ENV: ".35", MAX_TOKENS_ENV: "0"})
    group = rt.OwnedProcesses(run, workers, source, deadline)

    def timeout(*_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("Calibration cancelled or exceeded the wall budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
    try:
        rt.write(run / "state/burners_before.json", burner_status())
        rt.write(run / "state/host_before_lease.json", snapshot(topology=True))
        scope = {gpu: info[0] for gpu, info in available.items()} if args.all_burners_off else expected
        with experiment_lease(scope, run / "state") as stopped:
            devices, occupancy = rt.verify_devices(expected)
            rt.write(run / "state/lease.json", {"stopped_burners": stopped, "gpu_inventory": devices,
                                                 "foreign_gpu_occupancy": occupancy})
            rt.write(run / "state/host_in_lease.json", snapshot())
            rt.verify_ports()
            ec.mkdir(mode=0o700)
            try:
                rt.launch_models(group, tag, ec, expected, asset_repo / rt.REPORT / "media",
                                 token=True, common=True, role_environment=role_environment,
                                 eager_pd=numeric_regime == "eager")
                if numeric_regime == "eager":
                    observed = {}
                    for role, port in (("encoder", 18840), ("prefill", 18841), ("decode", 18842)):
                        worker, = rt.http_json(port, "campaign/worker_info", timeout=30)
                        numerics = worker["numerics"]
                        rt.require(not numerics["batch_invariant"]
                                   and numerics["enforce_eager"] == (role != "encoder"),
                                   f"Worker did not activate the configured eager BI0 regime: {role}")
                        observed[role] = numerics
                    rt.write(run / "state/eager_execution.json", observed)
                asyncio.run(measure_epoch(run, manifests[rt.CALIBRATION], manifests[rt.MAIN100], ec,
                                          device_indices=tuple(gpu for _, gpu, _, _ in rt.ROLES),
                                          repeats=args.repeats, inline_redirect=args.inline_redirect))
                if numeric_regime == "bi1-qsplit4":
                    observed = {}
                    for role, port in (("prefill", 18841), ("decode", 18842)):
                        worker, = rt.http_json(port, "campaign/worker_info", timeout=30)
                        numerics = worker["numerics"]
                        fixed = numerics["fixed_single_query_FA3"]
                        rt.require(numerics["batch_invariant"] and fixed and fixed["requested_splits"] == 4,
                                   f"Worker did not activate the configured numerical regime: {role}")
                        observed[role] = fixed
                    rt.write(run / "state/fixed_decode_attention.json", observed)
                if preissue_crossing:
                    registry, = rt.http_json(18840, "campaign/ec_ipc/state", timeout=30)
                    rt.write(run / "state/crossing_registry_closeout.json", registry)
                    rt.require(registry["live_images"] == registry["outstanding_exports"] == 0
                               and registry["crossing_issued"] == registry["crossing_acknowledged"]
                               + registry["crossing_revoked"], "Crossing descriptor closeout is incomplete")
            finally:
                group.close()
                rt.verify_devices(expected)
                shutil.rmtree(ec)
    except BaseException as error:
        rt.write(run / "state/failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rt.write(run / "state/complete.json", {"status": "COMPLETE", "time": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--wall-seconds", type=int, default=2700)
    parser.add_argument("--all-burners-off", action="store_true")
    parser.add_argument("--inline-redirect", action="store_true",
                        help="Use the shared P-inline descriptor route with D reassignment")
    parser.add_argument("--numeric-regime", choices=INVARIANCE_ROLES, default="bi0")
    args = parser.parse_args()
    run_calibration(args, numeric_regime=args.numeric_regime)


if __name__ == "__main__":
    main()
