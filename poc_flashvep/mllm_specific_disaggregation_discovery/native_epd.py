"""Run the frozen image-only cohort through the existing BI0 NativeStrong EPD."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from ..mllm_epd_h200_wide_decode_batch_token_split.host import sample_nvml, snapshot
from ..mllm_epd_h200_wide_decode_batch_token_split.online import configure, fingerprint_source

ORDER_SEED = 2026112701
ARRIVAL_SEED = 2026112702
SCREEN_LOADS = (0.75, 2.5, 5.0)
ROSTER = {"Main100": 100,
          "MMMU": 40, "MathVista": 30, "MMReason": 30}


def prompt_messages(row, nonce):
    def image_message(index):
        image = row["images"][index]
        return {
            "type": "image_url",
            "image_url": {"url": Path(image["path"]).as_uri()},
            "uuid": hashlib.sha256((nonce + row["request_id"] + str(index)).encode()).hexdigest(),
        }

    if "content_order" in row:
        content = [
            image_message(item["image_index"]) if item["type"] == "image"
            else {"type": "text", "text": item["text"]}
            for item in row["content_order"]
        ]
    else:
        content = [image_message(index) for index in range(len(row["images"]))]
        content.append({"type": "text", "text": row["question"]})
    return [{"role": "user", "content": content}]


def validate_media(rows, root):
    seen = set()
    for row in rows:
        request_id = row["request_id"]
        rt.require(request_id not in seen, f"Duplicate request {request_id}")
        seen.add(request_id)
        rt.require(len(row["images"]) == row["image_count"] > 0, f"Invalid images for {request_id}")
        for image in row["images"]:
            path = Path(image["path"]).resolve(strict=True)
            rt.require(path.is_relative_to(root), f"Image outside allowed workspace for {request_id}")
            if "SHA256" in image:
                rt.require(rt.sha(path) == image["SHA256"], f"Frozen image changed for {request_id}")
        if "content_order" in row:
            indices = [part["image_index"] for part in row["content_order"] if part["type"] == "image"]
            rt.require(sorted(indices) == list(range(row["image_count"])), f"Invalid prompt images for {request_id}")


def screen_rows(rows, count):
    """Choose a nested, dataset-stratified Discovery roster independently of performance."""
    groups = {}
    for row in rows:
        groups.setdefault(row["source"], []).append(row)
    nominations = []
    for source, group in groups.items():
        for rank, row in enumerate(group):
            nominations.append(((rank + 0.5) / len(group), source, row))
    selected = {row["request_id"] for _, _, row in sorted(nominations)[:count]}
    return [row for row in rows if row["request_id"] in selected]


def choose_medium(results):
    """Freeze the highest screened point retaining 80% offered rate and 80% TTFT pass."""
    suitable = [qps for qps, metrics in results if metrics["primary_execution_valid"]
                and metrics["completed_requests_per_second"] >= .8 * qps
                and metrics["gate_attainment"]["TTFT_PASS"] >= .8]
    if not suitable:
        return results[0][0]
    maximum = results[-1][0]
    if len(suitable) == len(results):
        return results[-2][0]
    return min(maximum, max(suitable))


def nixl_metrics():
    results = {}
    for role, _, port, _ in rt.ROLES:
        if role == "encoder":
            continue
        url = f"http://127.0.0.1:{port}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                lines = response.read().decode("utf-8").splitlines()
            results[role] = [line for line in lines if "vllm:nixl_" in line]
        except (OSError, urllib.error.URLError) as error:
            results[role] = {"status": "NOT_MEASURED", "error": str(error)}
    return results


def run_cohorts(run, manifest, url, *, screen_count, screening_only=False, full_qps=None):
    client = rt.module("client")
    metrics_module = rt.module("metrics")
    ownership = rt.module("ipc_ec_control")
    client.messages = prompt_messages
    original = manifest["requests"]
    rows = original.copy()
    random.Random(ORDER_SEED).shuffle(rows)
    discovery = set(manifest["discovery_ids"])
    tuning = screen_rows([row for row in rows if row["request_id"] in discovery], screen_count)
    rt.require(len(tuning) == screen_count, "Incomplete screening roster")
    # Warm only with a request in the Discovery200 contract.
    asyncio.run(client.execute(tuning[:1], [url], "warmup", 1, ARRIVAL_SEED, 16, serial=True))
    rt.require(rt.read(run / "state/warmup_complete.json")["passed"], "Warmup failed")
    ownership.ownership_empty("warmup", "cuda_ipc")

    def execute(label, cohort, qps):
        counters_before = nixl_metrics()
        with sample_nvml(run / "raw" / f"{label}_nvml.json", {role: gpu for role, gpu, _, _ in rt.ROLES}):
            asyncio.run(client.execute(
                cohort, [url], label, qps, ARRIVAL_SEED, None, defer_raw=True,
                metadata={"method": "NativeStrong-Free-BI0", "order_seed": ORDER_SEED,
                          "protocol": "Discovery200", "roster_scope": "Discovery" if label.startswith("screen")
                          else "Discovery200 with frozen Internal Validation50"},
            ))
        ownership.ownership_empty(label, "cuda_ipc")
        # Outside the clean timed cohort. Histograms are engine-level telemetry,
        # so a diagnostic snapshot reports any unsettled count explicitly.
        rt.write(run / "raw" / f"{label}_nixl_prometheus.json",
                 {"before": counters_before, "after": nixl_metrics()})
        result = metrics_module.summarize(label)
        print("COHORT", label, json.dumps(result), flush=True)
        return result

    results = [(qps, execute(f"screen_{index + 1}", tuning, qps))
               for index, qps in enumerate(SCREEN_LOADS)]
    rt.require(all(result["primary_execution_valid"] for _, result in results),
               "Load screening registered failed requests. See raw request logs.")
    selected_qps = choose_medium(results)
    qps = full_qps if full_qps is not None else selected_qps
    rt.write(run / "configs/frozen_native_load.json", {
        "offered_QPS": qps, "screened_QPS": SCREEN_LOADS, "automatic_QPS": selected_qps,
        "explicit_complementary_load": full_qps is not None,
        "selection": "Highest screened point with >=80% historical TTFT pass and >=80% offered completion rate. "
                     "If all pass, retain the middle point. If none pass, retain the first point.",
        "scope": "Finite cohort. Load and historical SLO freeze before candidate inspection.",
        "screen_requests": len(tuning),
    })
    if not screening_only:
        result = execute("discovery200_native", rows, qps)
        rt.require(result["primary_execution_valid"], "Discovery200 registered failed requests. See raw request logs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--screen-count", type=int, default=42)
    parser.add_argument("--screening-only", action="store_true")
    parser.add_argument("--full-qps", type=float, choices=SCREEN_LOADS,
                        help="Measure a complementary full cohort at this registered screening load")
    parser.add_argument("--wall-seconds", type=int, default=3000)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    media_root = base.parent
    asset_repo, receipt, manifest_file, run = (
        path.resolve() for path in (args.asset_repo, args.receipt, args.manifest, args.run_dir)
    )
    rt.require(run.is_relative_to(media_root) and not run.is_relative_to(base),
               "Use a separate run directory in the shared H200 workspace")
    manifest = rt.read(manifest_file)
    rt.require(Counter(row["study_source"] for row in manifest["requests"]) == ROSTER
               and len(set(manifest["discovery_ids"])) == 150,
               "Full frozen Discovery200 manifest required")
    validate_media(manifest["requests"], media_root)
    rt.git_check(asset_repo)
    fingerprint = rt.runtime_inventory()
    model, _ = rt.verify_inputs(asset_repo, receipt)
    available = {int(index): (uuid, name) for index, uuid, name in
                 rt.nvidia("--query-gpu=index,uuid,name")}
    expected = {gpu: available[gpu][0] for _, gpu, _, _ in rt.ROLES}
    rt.require(all(available[gpu][1] == "NVIDIA H200" for gpu in expected), "Unexpected EPD GPU model")
    deadline = time.time() + args.wall_seconds

    run.mkdir(mode=0o700)
    for directory in ("state", "logs", "raw", "analysis", "data", "configs", "references"):
        (run / directory).mkdir()
    shutil.copyfile(manifest_file, run / "data/discovery200.json")
    rt.write(run / "state/model_download_complete.json", {"directory": str(model), "revision": rt.MODEL_REVISION})
    source, settings = configure(base, asset_repo, run, deadline, freeze_auxiliary=True)
    rt.write(run / "state/environment.json", {
        "runtime": fingerprint, "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "receipt_SHA256": rt.sha(receipt), "manifest_SHA256": rt.sha(manifest_file),
        "source_SHA256": fingerprint_source(source), "runner_SHA256": rt.sha(__file__),
        "authorized_uuid_by_gpu": expected, "args": {key: str(value) for key, value in vars(args).items()},
    })
    ec = Path("/dev/shm") / f"h200_discovery_{settings['MLLM_EPD_RUN_NONCE']}"
    group = rt.OwnedProcesses(run, os.environ.copy(), source, deadline)

    def cancel(*_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("Native EPD deadline or cancellation")

    signal.signal(signal.SIGALRM, cancel)
    signal.signal(signal.SIGTERM, cancel)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.time()))
    try:
        rt.write(run / "state/burners_before.json", burner_status())
        rt.write(run / "state/host_before.json", snapshot(topology=True))
        with experiment_lease(expected, run / "state") as stopped:
            inventory, occupancy = rt.verify_devices(expected)
            rt.write(run / "state/lease.json", {"stopped_burners": stopped, "gpu_inventory": inventory,
                                                "other_gpu_processes": occupancy})
            rt.verify_ports()
            ec.mkdir(mode=0o700)
            try:
                rt.launch_models(group, "discovery_native", ec, expected, media_root, common=False)
                url = rt.launch_proxy(group, "epd", "epd", 18847, decode_cap="free")
                run_cohorts(run, manifest, url, screen_count=args.screen_count,
                            screening_only=args.screening_only, full_qps=args.full_qps)
            finally:
                group.close()
                shutil.rmtree(ec)
                rt.verify_devices(expected)
    except BaseException as error:
        rt.write(run / "state/failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rt.write(run / "state/complete.json", {"status": "COMPLETE", "time": time.time()})


if __name__ == "__main__":
    main()
