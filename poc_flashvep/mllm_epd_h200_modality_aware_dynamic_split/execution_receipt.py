"""Collect the campaign commands, run state and releasable resource receipts."""

import argparse
import hashlib
import json
import re
import shlex
from datetime import datetime
from pathlib import Path


RUN_LINE = re.compile(r"(START|END) (\S+) (?:(?:rc=(\d+)) )?(\S+)")
ENGINE_VALUES = (
    "--dtype",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-seqs",
    "--kv-cache-memory-bytes",
    "--gpu-memory-utilization",
    "--block-size",
    "--mm-processor-cache-gb",
    "--renderer-num-workers",
    "--compilation-config",
    "--attention-backend",
    "--worker-extension-cls",
)
ENGINE_FLAGS = (
    "--no-enable-prefix-caching",
    "--enable-chunked-prefill",
    "--mm-encoder-only",
    "--no-async-scheduling",
    "--enforce-eager",
)
EXECUTION_VALUES = (
    "--asset-repo",
    "--receipt",
    "--run-dir",
    "--wall-seconds",
    "--mode",
    "--decode-cap",
    "--seed",
    "--profile-state-ms",
    "--profile-decode-buffer-gib",
    "--paired-start-id",
    "--repeats",
    "--cost-model",
    "--replay",
    "--idle-budget-ms",
    "--safety-ms",
    "--subset",
    "--repeat-main100",
)
EXECUTION_MULTI_VALUES = ("--qps", "--ratios")
EXECUTION_FLAGS = (
    "--screen",
    "--profile",
    "--defer-client-raw",
    "--correctness",
    "--serial-encode-control",
    "--natural-boundaries",
    "--natural-hold",
    "--inline-redirect",
    "--observed-cut-fixtures",
    "--lead-boundary-control",
    "--shadow",
    "--all-burners-off",
)


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def sha256(path):
    if path.is_file():
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return None


def selected(data, keys):
    return {key: data[key] for key in keys if key in data}


def allowed_args(argv, values, flags, many=()):
    output = []
    omitted = []
    pos = 0
    while pos < len(argv):
        arg = argv[pos]
        pos += 1
        if arg in many:
            output.append(arg)
            while pos < len(argv) and not argv[pos].startswith("--"):
                output.append(argv[pos])
                pos += 1
        if arg in values:
            output.extend((arg, argv[pos]))
            pos += 1
        elif arg in flags:
            output.append(arg)
        elif arg not in many and arg.startswith("--"):
            omitted.append(arg.split("=", 1)[0])
    return output, omitted


def phase_runs(runs):
    phases = {}
    events = {}
    for path in sorted(runs.glob("phase_*.log")):
        ids = []
        for line in path.read_text().splitlines():
            match = RUN_LINE.fullmatch(line)
            if match:
                action, run_id, rc, at = match.groups()
                if run_id not in events:
                    events[run_id] = {"phase": path.stem}
                event = events[run_id]
                if action == "START":
                    event["start"] = at
                    ids.append(run_id)
                else:
                    event["end"] = at
                    if rc is not None:
                        event["rc"] = int(rc)
        phases[path.stem] = {
            "run_ids": ids,
            "log_sha256": sha256(path),
            "shell_sha256": sha256(path.with_suffix(".sh")),
        }
    return phases, events


def runtime_config(state):
    roles = {}
    for path in sorted(state.glob("owned_*.json")):
        role = path.stem.rsplit("_", 1)[-1]
        if role in ("encoder", "prefill", "decode"):
            owned = read_json(path)
            command = owned["command"]
            argv, _ = allowed_args(command, ENGINE_VALUES, ENGINE_FLAGS)
            worker = read_json(state / path.name.replace("owned_", "worker_", 1)) or {}
            roles[path.stem.removeprefix("owned_")] = {
                "args": argv,
                "gpu_uuid": owned["environment"].get("CUDA_VISIBLE_DEVICES"),
                **selected(worker, ("model_runner", "model_type", "scheduler", "device_properties")),
            }
            config = roles[path.stem.removeprefix("owned_")]
            for prefix in ("kv", "ec"):
                option = f"--{prefix}-transfer-config"
                if option in command:
                    payload = json.loads(command[command.index(option) + 1])
                    config[f"{prefix}_transfer"] = selected(
                        payload, (f"{prefix}_connector", f"{prefix}_role", f"{prefix}_connector_module_path")
                    )
            log = state.parent / "logs" / f"{path.stem.removeprefix('owned_')}.log"
            if log.is_file():
                data = log.read_text()
                patterns = {
                    "attention_backends": r"Using (?:backend AttentionBackendEnum\.)?([A-Z_]+) (?:vit )?attention",
                    "compiler_modes": r"<CompilationMode\.([A-Z_]+):",
                    "cuda_graph_modes": r"<CUDAGraphMode\.([A-Z_]+):",
                }
                config["log_sha256"] = sha256(log)
                config.update({name: sorted(set(re.findall(pattern, data))) for name, pattern in patterns.items()})
    return roles


def run_receipt(run, event):
    state = run / "state"
    path = state / "environment.json"
    at = {**event, "status": "IN_PROGRESS"}
    if "end" in event and "start" in event:
        elapsed = datetime.fromisoformat(event["end"]) - datetime.fromisoformat(event["start"])
        status = {0: "COMPLETED", None: "END_RECORDED_RC_UNAVAILABLE"}.get(event.get("rc"), "FAILED_EXECUTION")
        at.update(status=status, elapsed_seconds=elapsed.total_seconds())
    env = read_json(path)
    if env is None:
        return at
    argv = env["argv"]
    entry = Path(argv[0])
    entry_module = ".".join(("poc_flashvep", entry.parent.name, entry.stem))
    args, omitted = allowed_args(argv[1:], EXECUTION_VALUES, EXECUTION_FLAGS, EXECUTION_MULTI_VALUES)
    runtime = env["runtime"]
    command = shlex.join([runtime["executable"], "-u", "-m", entry_module, *args])
    state_close = read_json(state / "complete.json")
    if state_close and "end" not in event:
        at["status"] = "RUNNER_COMPLETE_EXIT_UNVERIFIED"
    out = {
        **at,
        "command": command,
        "omitted_option_names": omitted,
        "runtime": runtime,
        "environment_sha256": sha256(path),
        "base_commit": env["git_head"],
        "approved_devices": env["authorized_uuid_by_gpu"],
        "download_receipt_sha256": env.get("receipt_SHA256", env.get("receipt_sha256")),
        "runner_SHA256": env.get("runner_SHA256", env.get("runner_sha256")),
        "calibration_transport": env.get("transport"),
        "cost_model_SHA256": env.get("cost_model_SHA256"),
        "engine_config": runtime_config(state),
        "state_close": state_close,
        "output_log_sha256": sha256(run.with_suffix(".log")),
        "burner_gpu_indexes_before": [row[0] for row in (read_json(state / "burners_before.json") or [])],
    }
    lease = read_json(state / "lease.json") or {}
    out["lease"] = {
        "prior_burners_stopped_count": len(lease.get("stopped_burners", [])),
        "outside_lease_compute_process_count": len(lease.get("foreign_gpu_occupancy", [])),
    }
    result = {}
    for path in sorted((run / "configs").glob("*_arrival.json")):
        data = read_json(path)
        if isinstance(data, dict) and "trace" in data:
            result[path.stem.removesuffix("_arrival")] = {
                "sha256": sha256(path),
                "seed": data["seed"],
                "QPS": data["QPS"],
                "request_count": len(data["trace"]),
            }
    out["arrivals"] = result
    return out


def hardware(runs):
    for name in ("host_before_lease.json", "host.json"):
        for path in sorted(runs.glob(f"*/state/{name}")):
            data = read_json(path)
            if data and all(key in data for key in ("driver", "gpu", "cpu", "topology")):
                driver_log = data["driver"]["stdout"]
                hosts = {key: data[key]["stdout"] for key in ("gpu", "cpu", "topology")}
                hosts["topology"] = re.sub(r"\x1b\[[\d;]*m", "", hosts["topology"])
                hosts["driver"] = {}
                for name, label in (("driver_version", "Driver Version"), ("driver_cuda_max", "CUDA Version")):
                    version = re.search(label + r":\s*([\d.]+)", driver_log)
                    hosts["driver"][name] = version.group(1) if version else None
                disk = data.get("disk", {}).get("stdout", "").splitlines()
                hosts["disk"] = [dict(zip(("size", "used", "available", "capacity_percent", "mount"),
                                          line.split(maxsplit=5)[1:], strict=True))
                                 for line in disk[1:] if len(line.split(maxsplit=5)) == 6]
                hosts["ram"] = next(line for line in data["memory"].splitlines() if line.startswith("MemTotal:"))
                return {"time_unix": data["time_unix"], "run": path.parent.parent.name, **hosts}
    return None


def manifest_hashes(receipt, asset_repo):
    manifests = []
    for row in receipt["relocated_manifests"]:
        original = asset_repo / row["original_path"]
        relocated = Path(row["relocated_path"])
        manifests.append({
            **row,
            "original_verified": sha256(original) == row["original_SHA256"],
            "relocated_verified": sha256(relocated) == row["relocated_SHA256"],
        })
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--download-receipt", required=True, type=Path)
    parser.add_argument("--asset-repo", required=True, type=Path)
    parser.add_argument("--resource-closeout", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    phases, events = phase_runs(options.runs)
    for path in sorted(options.runs.glob("*/state/environment.json")):
        events.setdefault(path.parent.parent.name, {})
    asset_receipt = read_json(options.download_receipt)
    summary = {
        "base_asset_model_revision": asset_receipt["model_revision"],
        "manifest_hashes": manifest_hashes(asset_receipt, options.asset_repo),
        "phase_scripts": phases,
        "runs": {name: run_receipt(options.runs / name, event) for name, event in events.items()},
        "hardware": hardware(options.runs),
        "resource_closeout": read_json(options.resource_closeout) if options.resource_closeout else None,
        "scope": "Process ownership and authentication data omitted. A completed command may have failed correctness or evidence qualification.",
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
