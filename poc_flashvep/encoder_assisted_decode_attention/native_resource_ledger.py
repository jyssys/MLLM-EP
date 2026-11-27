"""Index the Native EPD resource measurements and preserve their input receipts."""

import argparse
import hashlib
import json
from pathlib import Path
import re


RUNS = {
    "native_scan": ("FAILED_STARTUP", ()),
    "native_scan_retry": ("MEASURED_ONLINE", (
        "native_strong_free_1", "native_strong_free_2p5", "native_strong_free_5",
    )),
    "native_profile_q2p5": ("DIAGNOSTIC", tuple(
        f"native_strong_free_2p5_sustained_r2_{arm}"
        for arm in ("clean_before", "profile", "clean_after")
    )),
    "native_capacity_retry": ("MEASURED_ONLINE_CORESIDENCY", (
        "native_strong_free_2p5", "encoder_arena_64gib",
    )),
}
EXTENDED_RUNS = {
    "native_prefill_capacity": ("MEASURED_ONLINE_CORESIDENCY", (
        "native_strong_free_2p5", "prefill_arena_80gib", "prefill_arena_64gib",
    )),
    "native_encoder_capacity80": ("MEASURED_ONLINE_CORESIDENCY", (
        "native_strong_free_2p5", "encoder_arena_80gib",
    )),
    "native_profile_low_observer": RUNS["native_profile_q2p5"],
}
ROLE_BY_GPU = {"4": "encoder", "5": "prefill", "6": "decode"}
GPU_FIELDS = ("index", "uuid", "name", "driver", "memory_MiB", "pci_bus_id")
RUNTIME_FLAGS = (
    "VLLM_BATCH_INVARIANT", "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "MLLM_EPD_EC_TRANSPORT",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)


def read(path):
    return json.loads(path.read_text())


def receipt(root, path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest}


def fingerprint(run, env, lease):
    host = read(run / "state/host_before_lease.json")
    topology = re.sub(r"\x1b\[[\d;]*m", "", host["topology"]["stdout"])
    links = {}
    for line in topology.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("GPU") or fields[0][3:] not in ROLE_BY_GPU:
            continue
        role = ROLE_BY_GPU[fields[0][3:]]
        links[role] = {
            "gpu_links": {role: fields[int(index) + 1] for index, role in ROLE_BY_GPU.items()},
            "cpu_affinity": fields[-3], "numa_node": int(fields[-2]),
        }
    devices = {
        ROLE_BY_GPU[fields[0]]: dict(zip(GPU_FIELDS, fields, strict=True))
        for fields in lease["gpu_inventory"] if fields[0] in ROLE_BY_GPU
    }
    assert len(links) == len(devices) == 3
    runtime = env["runtime"]
    return {"python": runtime["python"], "packages": runtime["packages"],
            "vllm_wheel_sha256": runtime["vllm_wheel_sha256"],
            "base_git_sha": env["git_head"], "gpu_by_role": devices, "topology": links,
            "runtime_flags": {k: env["settings"][k] for k in RUNTIME_FLAGS}}


def cohort(root, run, tag):
    config = read(run / f"configs/{tag}.json")
    workload = config["workload"]
    complete = read(run / f"state/{tag}_complete.json")
    measured = receipt(root, run / f"raw/{tag}_requests.jsonl")
    trace = receipt(root, run / f"configs/{tag}_arrival.json")
    assert measured["sha256"] == complete["raw_SHA256"]
    assert trace["sha256"] == config["arrival_SHA256"]
    return {"tag": tag, "offered_qps": config["QPS"], "arrival_seed": config["seed"],
            "output_mode": config["output_mode"], "planned": complete["planned_count"],
            "completed": complete["completed"], "duration_seconds": complete["duration_seconds"],
            "workload": {k: workload[k] for k in (
                "manifest_SHA256", "order_seed", "subset", "same_encoder_resident_GiB", "same_prefill_resident_GiB",
            ) if k in workload},
            "arrival": trace, "raw_requests": measured,
            "metrics": receipt(root, run / f"analysis/{tag}_metrics.json")}


def build(root, include_extended=False):
    summary = {"schema_version": 1, "raw_root_example": "h200_eada_runs", "runs": {}}
    runs = RUNS | EXTENDED_RUNS if include_extended else RUNS
    for label, (evidence, tags) in runs.items():
        run = root / label
        env = read(run / "state/environment.json")
        lease = read(run / "state/lease.json")
        common = fingerprint(run, env, lease)
        if "hardware_and_runtime" not in summary:
            summary["hardware_and_runtime"] = common
        terminal_path = run / "state/complete.json"
        if not terminal_path.exists():
            terminal_path = run / "state/failure.json"
        model = read(run / "state/model_download_complete.json")
        observer = run / "state/native_observer.json"
        measured = [tag for tag in tags if (run / f"analysis/{tag}_metrics.json").is_file()]
        entry = {"evidence": evidence, "terminal": read(terminal_path),
                 "model_revision": model["revision"], "asset_receipt_sha256": env["receipt_SHA256"],
                 "runner_source_sha256": env["runner_SHA256"]["online.py"],
                 "burners_stopped_count": len(lease["stopped_burners"]),
                 "outside_lease_gpu_processes": len(lease["foreign_gpu_occupancy"]),
                 "files": [receipt(root, f) for f in (
                     run / "state/environment.json", run / "state/lease.json", terminal_path,
                 )], "cohorts": [cohort(root, run, tag) for tag in measured]}
        if common != summary["hardware_and_runtime"]:
            entry["hardware_and_runtime_changed_fields"] = {
                field: value for field, value in common.items() if summary["hardware_and_runtime"][field] != value
            }
        if observer.exists():
            entry["native_observer"] = receipt(root, observer)
        if label in ("native_profile_q2p5", "native_profile_low_observer"):
            profile = tags[1]
            gate = run / f"analysis/{profile}_gate.json"
            if gate.exists():
                entry["observer_gate"] = receipt(root, gate)
                if label == "native_profile_low_observer" and read(gate)["passed"]:
                    entry["evidence"] = "TRACE_ANALYSIS"
            entry["profiler_inputs"] = [receipt(root, path)
                for path in sorted((run / f"raw/profiles/{profile}").glob("*/*")) if path.is_file()]
            counters = [run / f"raw/{profile}_{suffix}.json" for suffix in ("gpm", "nixl_before", "nixl_after")]
            entry["profiler_inputs"].extend(receipt(root, path) for path in counters if path.exists())
        if label == "native_capacity_retry":
            entry["capacity_inputs"] = [receipt(root, run / path) for path in (
                "state/worker_after_full_load.json", "state/encoder_arena_96gib_failure.json",
                "state/encoder_arena_64gib_ready.json", "state/encoder_arena_64gib_encoder_memory.json",
                "raw/native_strong_free_2p5_gpm.json", "raw/encoder_arena_64gib_gpm.json",
            )]
        if label in ("native_prefill_capacity", "native_encoder_capacity80"):
            paths = [run / "state/worker_after_full_load.json"]
            for tag in tags[1:]:
                role = tag.split("_", 1)[0]
                paths.extend(run / f"state/{tag}_{suffix}.json" for suffix in ("ready", "failure", f"{role}_memory"))
                paths.append(run / f"logs/{tag}.log")
            paths.extend(run / f"raw/{tag}_gpm.json" for tag in measured)
            entry["capacity_inputs"] = [receipt(root, path) for path in paths if path.exists()]
            if "COMPLETE" != entry["terminal"].get("status"):
                entry["evidence"] = "FAILED_CAPACITY_RUN"
        summary["runs"][label] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--include-extended", action="store_true")
    args = parser.parse_args()
    result = build(args.raw_root, args.include_extended)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
