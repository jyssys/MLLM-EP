"""Index a small, explicit review bundle; never copy weights or profiler binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    study = Path(__file__).resolve().parents[1]
    result = args.results.resolve()
    assert result.parent == repo / "poc_flashvep/deepep_revalidation/results"
    names = {
        "manifest.json", "plan.json", "analysis.json", "summary.json",
        "existing_policy_envelope.json", "existing_knob_slo_envelope.json",
        "finite_plan_envelope.json", "restart_aggregate.csv", "paired_results.csv",
        "paired_comparisons.csv", "request_run_summary.csv", "correctness_screen.json",
        "observer_restart_summary.json", "observer_restart_summary.csv",
        "matched_clean_summary.json", "matched_clean_bounds.csv",
        "matched_clean_composition_pairs.csv", "matched_clean_requests.csv",
        "decode_components.csv", "decode_component_aggregate.csv", "decode_component_summary.json",
    }
    selected = set()
    for area in ("fastpp_runs", "layered_runs", "nanoflow_runs", "vl_transfer"):
        for path in (result / area).rglob("*"):
            if (path.is_file() and (path.name in names or path.name.endswith((".overlap.json", ".overlap.csv")))
                    and path.stat().st_size < 4_000_000):
                selected.add(path)
    for path in (result / "cpu_analysis").glob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".md"} and path.stat().st_size < 4_000_000:
            selected.add(path)
    for path in (result / "request_traces").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl"} and path.stat().st_size < 4_000_000:
            selected.add(path)
    records = [{"path": str(p.relative_to(repo)), "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(selected)]
    target = study / "COMMON/EVIDENCE_BUNDLE.json"
    target.write_text(json.dumps({
        "scope": "selected small evidence; complete raw data remains local",
        "excluded": "weights, tensors, source checkouts, builds, raw stage streams and Nsight binaries",
        "results_root": str(result), "files": records,
        "bytes": sum(r["bytes"] for r in records)}, indent=2))
    if args.stage:
        # Each path is enumerated under the validated result root, no broad add.
        paths = [r["path"] for r in records] + [str(target.relative_to(repo))]
        for offset in range(0, len(paths), 50):
            subprocess.run(["git", "add", "--", *paths[offset:offset+50]], cwd=repo, check=True)
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records),
                      "index": str(target), "staged": args.stage}))


if __name__ == "__main__":
    main()
