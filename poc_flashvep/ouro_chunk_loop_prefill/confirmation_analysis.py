"""Paired cohort bootstrap for the corrected warm confirmation only."""

import csv
import numpy as np
from .common import ROOT, write
from .analyze import csvfile


def bootstrap(a, b, blocks, seed=20261022):
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    deltas = []
    gains = []
    for _ in range(5000):
        selected = []
        for block in rng.choice(unique, len(unique), replace=True):
            idx = np.flatnonzero(blocks == block)
            selected.extend(rng.choice(idx, len(idx), replace=True))
        aa, bb = a[selected].mean(), b[selected].mean()
        deltas.append(bb - aa)
        if aa:
            gains.append(100 * (1 - bb / aa))
    return (
        np.percentile(deltas, [2.5, 97.5]).tolist(),
        np.percentile(gains, [2.5, 97.5]).tolist() if gains else [None, None],
    )


def main():
    records = [
        r
        for r in csv.DictReader((ROOT / "results_raw.csv").open())
        if r["run"].startswith("fa4_confirm_warm_s") and r["warmup"] == "False"
    ]
    summary = []
    comparisons = []
    for p, n in [(2048, 4), (8192, 8)]:
        rows = [r for r in records if int(r["P"]) == p and int(r["N"]) == n]
        for mode, c in [
            ("atomic", 1024),
            ("split", 1024),
            ("mixed", 128),
            ("mixed", 1024),
            ("grouped", 1024),
        ]:
            r = [x for x in rows if x["mode"] == mode and int(x["C"]) == c]
            if not r:
                continue
            out = dict(
                P=p,
                N=n,
                mode=mode,
                C=c,
                cohorts=len(r),
                process_starts=len({x["run"] for x in r}),
                evidence="LIVE_MEASURED",
            )
            for k in [
                "long_TTFT_s",
                "ITL_p95_s",
                "ITL_p99_s",
                "ITL_max_s",
                "client_cohort_makespan_s",
                "E2E_p50_s",
                "E2E_p95_s",
                "TPOT_p50_s",
                "cohort_SLO_qualified_requests_s",
                "completed_requests_s",
                "output_tokens_s",
                "peak_HBM_bytes",
                "activation_bytes",
            ]:
                out["mean_cohort_" + k] = float(np.mean([float(x[k]) for x in r]))
            summary.append(out)
        candidate = {
            int(r["trial"]): r
            for r in rows
            if r["mode"] == "grouped" and r["C"] == "1024"
        }
        for mode, c in [
            ("atomic", 1024),
            ("split", 1024),
            ("mixed", 128),
            ("mixed", 1024),
        ]:
            baseline = {
                int(r["trial"]): r
                for r in rows
                if r["mode"] == mode and int(r["C"]) == c
            }
            trials = sorted(candidate.keys() & baseline.keys())
            if len(trials) < 3:
                continue
            blocks = np.array(
                [int(candidate[t]["run"].rsplit("s", 1)[-1]) for t in trials]
            )
            for field in [
                "long_TTFT_s",
                "ITL_p95_s",
                "ITL_p99_s",
                "ITL_max_s",
                "client_cohort_makespan_s",
                "cohort_SLO_qualified_requests_s",
                "output_tokens_s",
            ]:
                a = np.array([float(baseline[t][field]) for t in trials])
                b = np.array([float(candidate[t][field]) for t in trials])
                delta, gain = bootstrap(a, b, blocks)
                rate = field.endswith("requests_s") or field == "output_tokens_s"
                comparisons.append(
                    dict(
                        P=p,
                        N=n,
                        baseline=mode,
                        baseline_C=c,
                        candidate="grouped",
                        candidate_C=1024,
                        metric="mean_cohort_" + field,
                        paired_cohorts=len(trials),
                        baseline_mean=a.mean(),
                        candidate_mean=b.mean(),
                        speedup=(b.mean() / a.mean() if rate else a.mean() / b.mean())
                        if a.mean()
                        else None,
                        improvement_percent=(
                            100 * (b.mean() / a.mean() - 1)
                            if rate
                            else 100 * (1 - b.mean() / a.mean())
                        )
                        if a.mean()
                        else None,
                        CI95_delta_low=delta[0],
                        CI95_delta_high=delta[1],
                        CI95_improvement_low=(-gain[1] if rate else gain[0])
                        if gain[0] is not None
                        else None,
                        CI95_improvement_high=(-gain[0] if rate else gain[1])
                        if gain[1] is not None
                        else None,
                        evidence="LIVE_MEASURED_COHORT_BOOTSTRAP",
                    )
                )
    csvfile("warm_confirmation_summary.csv", summary)
    csvfile("paired_comparisons.csv", comparisons)
    write(
        ROOT / "confirmation_summary.json",
        dict(
            summary=summary,
            comparisons=comparisons,
            bootstrap="5000 draws, paired cohorts, process-start block resampling. Cohort quantiles are not pooled token quantiles.",
        ),
    )
    print("CONFIRMATION", len(records), "measured cells", len(comparisons), "contrasts")


if __name__ == "__main__":
    main()
