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


def run_calibration(args):
    worktree = Path(__file__).resolve().parents[2]
    asset_repo, receipt, run = (path.resolve() for path in (args.asset_repo, args.receipt, args.run_dir))
    rt.require(not run.is_relative_to(asset_repo) and not run.is_relative_to(worktree), "Run directory must be external")
    rt.require(args.repeats > 0 and args.wall_seconds > 0, "Repetitions and wall budget must be positive")
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
    source, settings = configure(worktree, asset_repo, run, deadline)
    rt.write(run / "state/environment.json", {
        "runtime": fingerprint, "transport": transport, "receipt_sha256": rt.sha(receipt),
        "manifest_sha256": {name: rt.sha(path) for name, path in manifests.items()},
        "source_sha256": fingerprint_source(source),
        "runner_sha256": {file.name: rt.sha(file) for file in sorted(Path(__file__).parent.glob("*.py"))},
        "git_head": subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip(),
        "authorized_uuid_by_gpu": expected, "settings": settings, "argv": sys.argv, "deadline": deadline,
    })
    ec = Path("/dev/shm") / f"h200_epd_{settings['MLLM_EPD_RUN_NONCE']}"
    group = rt.OwnedProcesses(run, os.environ.copy(), source, deadline)

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
                role_environment = {"decode": {"MLLM_EPD_LATE_INLINE_REDIRECT": "1"}} if args.inline_redirect else None
                rt.launch_models(group, "suffix-calibration", ec, expected, asset_repo / rt.REPORT / "media",
                                 token=True, common=True, role_environment=role_environment)
                asyncio.run(measure(run, manifests[rt.CALIBRATION], manifests[rt.MAIN100], ec,
                                    device_indices=tuple(gpu for _, gpu, _, _ in rt.ROLES), repeats=args.repeats,
                                    inline_redirect=args.inline_redirect))
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
    run_calibration(parser.parse_args())


if __name__ == "__main__":
    main()
