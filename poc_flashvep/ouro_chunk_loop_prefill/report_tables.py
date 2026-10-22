"""Generate evidence tables from completed measurements, never estimated timings."""

import csv
import json
from collections import Counter
from .common import ROOT, write
from .analyze import csvfile, scan


def main():
    records = list(scan())
    probes = []
    for file in sorted((ROOT / "raw").glob("*/P*.json")):
        d = json.loads(file.read_text())
        if "mode" not in d:
            continue
        p = d.get("parity", {})
        probes.append(
            dict(
                run=file.parent.name,
                P=d["prompt_tokens"],
                mode=d["mode"],
                C=d["chunk_tokens"],
                output_tokens=len(d["tokens"]),
                backend=d.get("backend", "NOT_RECORDED_INITIAL_SMOKE"),
                all_depth4=all(x == 4 for x in d["depths"]),
                tokens_exact=p.get("tokens"),
                hidden_exact=p.get("hidden", {}).get("exact"),
                hidden_rel_l2=p.get("hidden", {}).get("rel_l2"),
                logits_exact=p.get("logits", {}).get("exact"),
                logits_rel_l2=p.get("logits", {}).get("rel_l2"),
                KV_exact=p.get("kv_all_exact"),
                KV_rel_l2=p.get("kv_max_rel_l2"),
            )
        )
    csvfile("correctness_probes.csv", probes)
    work = []
    matrix = [
        dict(
            run=r["run"],
            phase="P0" if r["run"] == "baseline_gpu4_v1" else "P1",
            mode=r["mode"],
            C=r["C"],
            P_bin=r["P"],
            GPU=4,
            backend=r["backend"],
            evidence="LIVE_MEASURED_NUMERICAL_PROBE",
            status="COMPLETED",
        )
        for r in probes
    ]
    captures = []
    for file, d in records:
        family = file.parent.name
        status = (
            "WARMUP"
            if d.get("warmup")
            else "INVALID_COLD_ORDER"
            if family == "fa4_confirm_s0"
            else "COMPLETED"
        )
        matrix.append(
            dict(
                run=family,
                file=file.name,
                phase="P2",
                mode=d["mode"],
                C=d["chunk_tokens"],
                G=d["loop_group"],
                P_bin=d["prompt_bin"],
                concurrency=d["concurrency"],
                GPU=4,
                backend=d.get("backend", "triton"),
                trial=d["trial"],
                evidence="LIVE_MEASURED",
                status=status,
            )
        )
        if family == "stock_calibration_v1":
            continue  # Older trace used a default group label.
        prompt_work = Counter()
        first_long = None
        for q in d["quantums"]:
            for r in q["spans"]:
                if r["phase"] in ("prefill", "prefilling_recurrent"):
                    prompt_work[r["id"]] += r["tokens"] * r["group"]
                if r["id"].startswith("long_") and first_long is None:
                    first_long = q["host_begin"]
        for r in d["requests"]:
            expected = 4 * r["prompt_tokens"]
            if d["quantums"]:
                work.append(
                    dict(
                        run=family,
                        file=file.name,
                        id=r["id"],
                        expected_prompt_token_loops=expected,
                        issued_prompt_token_loops=prompt_work[r["id"]],
                        PASS=prompt_work[r["id"]] == expected,
                        all_generated_depth4=all(x == 4 for x in r["depths"]),
                        evidence="LIVE_SCHEDULER_DESCRIPTORS_PLUS_SEPARATE_GPU_KV_PROBES",
                    )
                )
        if first_long is not None:
            captures.append(
                dict(
                    run=family,
                    file=file.name,
                    mode=d["mode"],
                    C=d["chunk_tokens"],
                    P=d["prompt_bin"],
                    N=d["concurrency"],
                    actual_background_at_first_long_quantum=sum(
                        r["background"] and r["token_times"][-1] >= first_long
                        for r in d["requests"]
                    ),
                    API_submission_background=d["live_decode_at_long_admission"],
                )
            )
    matrix.extend(
        [
            dict(
                run="conditional",
                phase="P3",
                mode=m,
                evidence="NOT_RUN",
                status="NOT_RUN_GATE",
                GPU=4,
            )
            for m in ("adaptive_C_only", "adaptive_G_only", "adaptive_joint")
        ]
    )
    matrix.extend(
        [
            dict(
                run="conditional",
                phase="P4",
                mode="DP" + str(n),
                evidence="NOT_RUN",
                status="NOT_RUN_GATE",
                GPU="4,5" if n == 2 else "4,5,6,7",
            )
            for n in (2, 4)
        ]
    )
    matrix.append(
        dict(
            run="optional",
            phase="P2",
            mode="literal_whole_prompt_G_only",
            evidence="NOT_RUN",
            status="NOT_RUN_COMMON_TOKEN_BUDGET1024",
            GPU=4,
        )
    )
    matrix.extend(
        dict(
            run="optional_extension",
            phase="P2",
            mode=mode,
            GPU=4,
            evidence="NOT_RUN",
            status="NOT_RUN_NO_CONTINUATION_GATE",
        )
        for mode in (
            "steady_open_loop_Poisson",
            "P16K_C16_extension",
            "C2048_extension",
            "G3_robustness",
            "complete_GSM8K64_MATH50_quality_anchor",
        )
    )
    csvfile("experiment_matrix.csv", matrix)
    csvfile("full_R4_work_validation.csv", work)
    csvfile("interference_validity.csv", captures)
    measured = (
        list(csv.DictReader((ROOT / "warm_confirmation_summary.csv").open()))
        if (ROOT / "warm_confirmation_summary.csv").exists()
        else []
    )
    lines = [
        "# Benchmark summary",
        "",
        "All values below come from physical GPU4. No model-parallel or DP scaling result is included.",
        "The main confirmation uses FA4, resident engines, equal calibration warm-up, and frozen natural generation.",
        "The output cap is 128 tokens. EOS is enabled. These are bounded serving requests, not full reasoning accuracy tests.",
        "",
        "## Corrected warm confirmation",
        "",
        "Each cell summarizes paired cohorts across three process starts. The table reports means of cohort statistics.",
        "It does not label a mean of cohort p95 values as a pooled-token p95.",
        "",
        "| Prompt bin / concurrency | Policy | C / G | Cohorts | Long TTFT ms | ITL p95 ms | ITL p99 ms | Full cohort s | Output tok/s | SLO-qualified req/s | Peak allocated GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in measured:
        g = 4 if r["mode"] == "atomic" else 2 if r["mode"] == "grouped" else 1
        lines.append(
            "| "
            + f"{r['P']} / {r['N']} | {r['mode']} | {r['C']} / {g} | {r['cohorts']} | "
            + " | ".join(
                f"{float(r[k]) * s:.3f}"
                for k, s in [
                    ("mean_cohort_long_TTFT_s", 1000),
                    ("mean_cohort_ITL_p95_s", 1000),
                    ("mean_cohort_ITL_p99_s", 1000),
                    ("mean_cohort_client_cohort_makespan_s", 1),
                    ("mean_cohort_output_tokens_s", 1),
                    ("mean_cohort_cohort_SLO_qualified_requests_s", 1),
                    ("mean_cohort_peak_HBM_bytes", 1 / 2**30),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Full-cohort and throughput ratios can retain new-shape JIT effects. They are not equal-work architecture speedups.",
        "SLO-qualified req/s uses a drained burst cohort. It is not steady-state open-loop goodput.",
        "Thresholds are TTFT <= 5 s and maximum observed request ITL <= 100 ms. Zero values remain zero.",
        "",
        "## Paired contrasts",
        "",
        "Positive improvement means lower latency or higher rate. Speedup below 1.000× means a slowdown.",
        "The 95% intervals resample paired cohorts with process-start blocks. They never resample tokens independently.",
        "",
        "| Prompt / concurrency | Baseline | Metric | G2 speedup | Improvement % [95% CI] |",
        "| --- | --- | --- | ---: | --- |",
    ]
    if (ROOT / "paired_comparisons.csv").exists():
        for r in csv.DictReader((ROOT / "paired_comparisons.csv").open()):
            if r["baseline"] not in ("atomic", "mixed") or r["baseline_C"] != "1024":
                continue
            if r["metric"] not in (
                "mean_cohort_long_TTFT_s",
                "mean_cohort_ITL_p95_s",
                "mean_cohort_ITL_p99_s",
                "mean_cohort_output_tokens_s",
            ):
                continue
            speed = f"{float(r['speedup']):.3f}×" if r["speedup"] else "N/A"
            gain = (
                f"{float(r['improvement_percent']):.2f} [{float(r['CI95_improvement_low']):.2f}, {float(r['CI95_improvement_high']):.2f}]"
                if r["improvement_percent"]
                else "N/A"
            )
            lines.append(
                f"| {r['P']} / {r['N']} | {r['baseline']} C1024 | {r['metric'].replace('mean_cohort_', '')} | {speed} | {gain} |"
            )
    lines += [
        "",
        "## Complete tables and limitations",
        "",
        "- `results_raw.csv` contains every completed natural-serving cell, including calibration, warm-up, and rejected cold-order data.",
        "- `paired_comparisons.csv` contains all registered baselines and confidence intervals.",
        "- `experiment_matrix.csv` marks conditional Adaptive and DP work as NOT_RUN.",
        "- `natural_output_parity.csv` preserves batch-dependent output differences.",
        "- `full_R4_work_validation.csv` checks issued prompt work against four times the actual prompt length.",
        "- No full GSM8K/MATH accuracy, HTTP serving, steady-state Poisson goodput, or adaptive-policy superiority is claimed.",
        "",
    ]
    (ROOT / "benchmark_summary.md").write_text("\n".join(lines))
    write(
        ROOT / "evidence_counts.json",
        dict(
            natural_cells=len(records),
            matrix_rows=len(matrix),
            probes=len(probes),
            prompt_work_checks=len(work),
            prompt_work_failures=sum(not r["PASS"] for r in work),
            invalid_interference_cases=sum(
                r["N"] > 1 and r["actual_background_at_first_long_quantum"] == 0
                for r in captures
            ),
            caveat="Counts include separately labeled warm-up and diagnostic rows. Not a benchmark accuracy denominator.",
        ),
    )


if __name__ == "__main__":
    main()
