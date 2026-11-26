"""Index public experiment metadata and hash the retained local evidence."""

import argparse
import json
from pathlib import Path
import shlex

from ..mllm_epd_h200_modality_aware_dynamic_split import execution_receipt as prior


RUN_FLAGS = (
    *prior.EXECUTION_FLAGS, "--stock-cut-trace", "--observe-batch", "--native-aa",
    "--slice-aware-transfer", "--adaptive-budget", "--strict-only", "--cost-only",
    "--decode-identities", "--record-causal-progress",
    "--preissue-crossing-ec", "--require-ready-transport",
)
RUN_VALUES = (
    *prior.EXECUTION_VALUES, "--numeric-regime", "--causal-model", "--fixed-d-graph-size",
    "--gain-estimate",
    "--preissue-max-suffix-fraction", "--preissue-max-suffix-tokens",
)
RUN_MULTI_VALUES = (
    *prior.EXECUTION_MULTI_VALUES, "--ratio-gates", "--recovery-fixtures",
    "--qualified-ratios", "--qualified-token-cuts",
)


def retained_evidence(run):
    for folder in ("configs", "analysis", "raw", "data"):
        for path in sorted((run / folder).rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".bin", ".npz", ".pt"}:
                yield str(path.relative_to(run)), {
                    "bytes": path.stat().st_size,
                    "sha256": prior.sha256(path),
                }


def receipt(run, event, hash_raw):
    result = prior.run_receipt(run, event)
    state = run / "state"
    environment = prior.read_json(state / "environment.json")
    if environment:
        entry = Path(environment["argv"][0])
        module = ".".join(("poc_flashvep", entry.parent.name, entry.stem))
        args, omitted = prior.allowed_args(
            environment["argv"][1:], RUN_VALUES, RUN_FLAGS, RUN_MULTI_VALUES,
        )
        result["command"] = shlex.join([environment["runtime"]["executable"], "-u", "-m", module, *args])
        result["omitted_option_names"] = omitted
        result["numeric_regime"] = environment.get("numeric_regime")
        for field in ("recovery_source_SHA256", "causal_model_SHA256", "crossing_preissue_envelope"):
            if field in environment:
                result[field] = environment[field]
        result["worker_numerics"] = {
            worker.stem: (prior.read_json(worker) or {}).get("numerics")
            for worker in sorted(state.glob("worker_*.json"))
        }
    close = prior.read_json(state / "complete.json")
    failure = prior.read_json(state / "failure.json")
    if failure:
        result.update(status="FAILED_EXECUTION", failure=failure)
    elif close:
        result.update(status="COMPLETED", state_close=close)
    if hash_raw and (close or failure):
        result["retained_local_evidence"] = dict(retained_evidence(run))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--asset-repo", required=True, type=Path)
    parser.add_argument("--download-receipt", required=True, type=Path)
    parser.add_argument("--resource-closeout", type=Path)
    parser.add_argument("--hash-raw", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    phases, events = prior.phase_runs(args.runs)
    for state in args.runs.glob("*/state"):
        events.setdefault(state.parent.name, {})
    assets = prior.read_json(args.download_receipt)
    results = {
        "base_asset_model_revision": assets["model_revision"],
        "manifest_hashes": prior.manifest_hashes(assets, args.asset_repo),
        "hardware": prior.hardware(args.runs),
        "phase_scripts": phases,
        "runs": {name: receipt(args.runs / name, event, args.hash_raw) for name, event in sorted(events.items())},
        "resource_closeout": prior.read_json(args.resource_closeout) if args.resource_closeout else None,
        "scope": "Command completion is recorded independently of correctness and evidence qualification. "
                 "Retained tensor and profile artifacts stay on the experiment server.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
