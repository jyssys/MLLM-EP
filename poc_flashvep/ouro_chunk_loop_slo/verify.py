"""Audit completed evidence without treating a negative finding as a test failure."""

import csv
import json

from .common import ROOT, UUIDS, write


def rows(name):
    return list(csv.DictReader((ROOT / name).open()))


def main():
    checks = []
    required = (
        "fixed_grid_manifest.json",
        "repeated_shortlist_manifest.json",
        "numerical_manifest.json",
        "heldout_poisson_manifest.json",
        "heldout_burst_manifest.json",
        "separate_profiling_manifest.json",
        "quantum_midpoints_manifest.json",
        "final_controls_manifest.json",
        "solo_policy_verification_manifest.json",
    )
    seen = set()
    profiler_failures = []
    for name in required:
        manifest = json.loads((ROOT / name).read_text())
        for job in manifest["jobs"]:
            if job["id"] in seen:
                continue
            seen.add(job["id"])
            folder = ROOT / "raw" / job["id"]
            complete = (folder / "completed.json").exists()
            failed = (folder / "failure.json").exists()
            if job["kind"] == "nsys" and failed:
                profiler_failures.append(job["id"])
                continue
            assert complete and not failed, (
                job["id"],
                "Missing or failed required job",
            )
            identity = json.loads((folder / "identity.json").read_text())
            assert identity["physical_GPU"] in UUIDS
            assert identity["UUID"] == UUIDS[identity["physical_GPU"]]
            if job["kind"] in ("grid", "traffic"):
                for mode, chunk in job["cells"]:
                    assert (folder / f"{mode}_C{chunk}_complete.json").exists()
            checks.append(
                dict(
                    job=job["id"],
                    status="EXPECTED_EVIDENCE_PRESENT",
                    GPU=identity["physical_GPU"],
                )
            )
    counts = json.loads((ROOT / "analysis_counts.json").read_text())
    assert counts["fixed_cohorts"] == 316
    assert counts["serving_runs"] == 176
    numerical = rows("numerical_probe_summary.csv")
    assert len(numerical) == 48 and all(r["PASS"] == "True" for r in numerical)
    work = rows("request_work_accounting.csv")
    complete_work = [r for r in work if r["status"] == "COMPLETE"]
    assert all(r["prompt_R4_PASS"] == "True" for r in complete_work)
    assert all(
        r["decode_token_passes_submitted"]
        == r["minimum_decode_passes_for_delivered_outputs"]
        for r in complete_work
    )
    result = dict(
        status="EVIDENCE_COMPLETENESS_PASS",
        expected_job_checks=checks,
        optional_profiler_failures=profiler_failures,
        fixed_cohorts=counts["fixed_cohorts"],
        serving_policy_runs=counts["serving_runs"],
        numerical_exact_comparisons=len(numerical),
        complete_request_work_checks=len(complete_work),
        limitation="This audit checks evidence completeness and work invariants. It does not assert universal output parity, benchmark accuracy or sustainable capacity.",
    )
    write(ROOT / "evidence_verification.json", result)
    print("EVIDENCE_COMPLETENESS_PASS", len(checks), len(complete_work))


if __name__ == "__main__":
    main()
