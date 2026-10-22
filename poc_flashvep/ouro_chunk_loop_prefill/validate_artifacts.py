"""CPU-only checks for committed evidence and explicit negative-result limits."""

import csv
import gzip
import hashlib
import json
from .common import ROOT, write


def rows(name):
    return list(csv.DictReader((ROOT / name).open()))


def main():
    required = (
        "analysis.md",
        "benchmark_summary.md",
        "correctness.md",
        "implementation_scope.md",
        "experiment_matrix.csv",
        "profiling.md",
        "reproduce.md",
        "failures.md",
    )
    assert all((ROOT / name).is_file() for name in required)
    archives = json.loads((ROOT / "archive_manifest.json").read_text())
    for row in archives:
        data = gzip.decompress((ROOT / row["archive"]).read_bytes())
        digest = hashlib.sha256(data).hexdigest()
        assert digest == row["sha256"]
        assert digest == hashlib.sha256((ROOT / row["source"]).read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "workload_manifest.json").read_text())
    original_hash = manifest.pop("sha256")
    assert (
        original_hash
        == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    )
    assert manifest["SLO"] == dict(TTFT_s=5.0, request_max_ITL_s=0.100)
    probes = rows("correctness_probes.csv")
    comparisons = [r for r in probes if r["tokens_exact"]]
    assert len(comparisons) == 31
    for r in comparisons:
        assert all(
            r[field] == "True"
            for field in ("tokens_exact", "hidden_exact", "logits_exact", "KV_exact")
        )
        assert all(
            float(r[field]) == 0
            for field in ("hidden_rel_l2", "logits_rel_l2", "KV_rel_l2")
        )
    work = rows("full_R4_work_validation.csv")
    assert all(
        r["PASS"] == "True" and r["all_generated_depth4"] == "True" for r in work
    )
    confirmation = rows("warm_confirmation_summary.csv")
    assert len(confirmation) == 10
    assert all(
        int(r["cohorts"]) == 10 and int(r["process_starts"]) == 3 for r in confirmation
    )
    contrast = rows("paired_comparisons.csv")
    assert len(contrast) == 56
    validity = rows("interference_validity.csv")
    assert all(
        int(r["actual_background_at_first_long_quantum"]) > 0
        for r in validity
        if int(r["N"]) > 1
    )
    counts = json.loads((ROOT / "diagnostic_summary.json").read_text())
    assert counts["warm_confirmation_requests"] == 600
    assert counts["warm_confirmation_truncated"] == 600
    assert counts["grouped_vs_split_exact_outputs"] == 112
    write(
        ROOT / "artifact_validation.json",
        dict(
            status="PASS_ARTIFACT_INTEGRITY_NOT_QUALITY_CERTIFICATION",
            archive_files=len(archives),
            archive_source_bytes=sum(r["bytes"] for r in archives),
            archive_compressed_bytes=sum(r["compressed_bytes"] for r in archives),
            required_documents=list(required),
            numerical_comparisons=len(comparisons),
            issued_work_checks=len(work),
            confirmation_cohorts_per_policy=10,
            confirmation_process_starts=3,
            paired_contrasts=len(contrast),
            quality="NOT_RUN_COMPLETE_BENCHMARK_ACCURACY",
            batched_output_parity="112/120 Grouped vs Split, not universal parity",
            verdict="CONDITIONAL_FIXED_QUANTUM_ONLY",
            adaptive_and_multiGPU="NOT_RUN_GATE_NOT_PASSED",
        ),
    )
    print(
        "VALIDATED",
        len(archives),
        "archives",
        len(comparisons),
        "numerical comparisons",
        len(work),
        "work checks",
    )


if __name__ == "__main__":
    main()
