"""Phase 0: read historical evidence without modifying a historical artifact."""

import csv
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from .common import ROOT, OLD, REPO, HISTORY, digest, write


def readcsv(path):
    return list(csv.DictReader(path.open()))


def main():
    names = (
        "analysis.md",
        "benchmark_summary.md",
        "correctness.md",
        "profiling.md",
        "results_raw.csv",
        "paired_comparisons.csv",
        "ITL_stall_counts.csv",
        "experiment_matrix.csv",
        "natural_output_parity.csv",
        "workload_manifest.json",
        "failures.md",
    )
    evidence = []
    for name in names:
        file = OLD / name
        committed = subprocess.check_output(
            ["git", "show", f"{HISTORY}:{file.relative_to(REPO)}"], cwd=REPO
        )
        assert hashlib.sha256(committed).hexdigest() == digest(file), name
        evidence.append(
            dict(
                path=str(file.relative_to(REPO)),
                sha256=digest(file),
                bytes=file.stat().st_size,
            )
        )
    archives = json.loads((OLD / "archive_manifest.json").read_text())
    for row in archives:
        data = gzip.decompress((OLD / row["archive"]).read_bytes())
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert digest(OLD / row["source"]) == row["sha256"]
    rows = readcsv(OLD / "results_raw.csv")
    mismatch = [
        r
        for r in readcsv(OLD / "natural_output_parity.csv")
        if r["run"].startswith("fa4_confirm_warm_s")
        and r["mode"] == "grouped"
        and int(r["trial"]) >= 2
        and r["tokens_exact"] != "True"
    ]
    write(
        ROOT / "baseline_audit.json",
        dict(
            historical_commit=HISTORY,
            historical_verdict="CONDITIONAL_FIXED_QUANTUM_ONLY_PRESERVED",
            verified_files=evidence,
            verified_archives=len(archives),
            serving_rows=len(rows),
            run_counts=dict(Counter(r["run"] for r in rows)),
            warm_confirmation_rows=sum(
                r["run"].startswith("fa4_confirm_warm_s") and r["warmup"] == "False"
                for r in rows
            ),
            unresolved_batched_mismatches=mismatch,
            old_index_fingerprint="7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9",
        ),
    )
    write(
        ROOT / "research_state.json",
        dict(
            status="PHASE0_AUDIT_COMPLETE",
            next="Calibrate no-interference decode and prefill. Freeze SLOs before new comparisons.",
            GPU_execution="Four independent single-GPU workers on physical4–7 per latest user instruction. No TP/PP or cluster scaling claim.",
        ),
    )
    print(
        "AUDIT",
        len(evidence),
        "commit-matched files",
        len(archives),
        "verified archives",
        len(mismatch),
        "preserved mismatches",
    )


if __name__ == "__main__":
    main()
