"""Render reviewed metric definitions with automatically transcribed result tables."""

import csv

from .common import ROOT


def read(name):
    path = ROOT / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def number(value, digits=2):
    return f"{float(value):.{digits}f}" if value not in (None, "") else "NOT_AVAILABLE"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main():
    source = read("slo_attainment_by_rate.csv")
    selected = [
        r for r in source if r["SLO"] == "moderate" and float(r["TTFT_limit_s"]) == 5.0
    ]
    lines = [
        "# Natural serving result tables",
        "",
        "Evidence: LIVE_MEASURED, in-process open-loop natural generation.",
        "Each row combines common-seed independent engine blocks for one policy and rate.",
        "All offered requests remain in the denominator.",
        "The joint rule requires a natural stop, TTFT ≤5 s, and every observed TBT ≤271.893227 ms.",
        "A capped response cannot pass the joint rule.",
        "The TBT-only column checks observed intervals. It does not certify response completion.",
        "Intervals use 5,000 crossed engine-block/source-family bootstrap draws with seed 20261023.",
        "These finite windows do not qualify production capacity or request-tail P99.",
        "Natural output differences prevent an equal-output-work speedup interpretation.",
        "",
    ]
    for pattern in ("poisson", "burst"):
        rows = sorted(
            [r for r in selected if r["pattern"] == pattern],
            key=lambda r: (float(r["rate"]), r["policy"]),
        )
        lines += [f"## {pattern.capitalize()} attainment", ""]
        values = []
        for r in rows:
            values.append(
                [
                    number(r["rate"], 3),
                    r["policy"],
                    r["offered"],
                    f"{r['TTFT_pass']}/{r['offered']}",
                    f"{r['TBT_all_token_pass']}/{r['offered']}",
                    f"{r['joint_pass']}/{r['offered']}",
                    number(100 * float(r["joint_pass_fraction"])),
                    f"[{number(100 * float(r['joint_pass_CI95_low']))}, {number(100 * float(r['joint_pass_CI95_high']))}]",
                    f"{r['natural_complete']}/{r['offered']}",
                    r["capped"],
                    f"{r['errors']}/{r['timeouts']}",
                    f"{r['stable_blocks']}/{r['process_blocks']}",
                ]
            )
        lines += [
            table(
                [
                    "Configured req/s",
                    "Policy",
                    "Offered N",
                    "TTFT pass",
                    "Observed TBT pass",
                    "Joint pass",
                    "Joint %",
                    "95% CI, %",
                    "Natural complete",
                    "Caps",
                    "Errors/timeouts",
                    "Stability-screen blocks",
                ],
                values,
            ),
            "",
            f"## {pattern.capitalize()} latency and goodput",
            "",
        ]
        lines += [
            table(
                [
                    "Configured req/s",
                    "Policy",
                    "TTFT P50/P95, s",
                    "Pooled TBT P95/P99, ms",
                    "E2E P50/P95, s",
                    "Mean output tokens",
                    "Declared-drain SLO req/s",
                ],
                [
                    [
                        number(r["rate"], 3),
                        r["policy"],
                        f"{number(r['TTFT_p50_s'])}/{number(r['TTFT_p95_s'])}",
                        f"{number(r['pooled_ITL_p95_ms'])}/{number(r['pooled_ITL_p99_ms'])}",
                        f"{number(r['E2E_p50_s'])}/{number(r['E2E_p95_s'])}",
                        number(r["mean_output_tokens"]),
                        number(r["SLO_goodput_offered_window_req_s"], 4),
                    ]
                    for r in rows
                ],
            ),
            "",
        ]
    lines += [
        "## Reading the tables",
        "",
        "A failed stability screen can reflect queued growth or an unfinished in-service tail.",
        "The CSV also records queued-growth blocks. A failed screen alone does not prove asymptotic instability.",
        "Configured Poisson rate differs from the realized offered count divided by the arrival window.",
        "The CSV preserves both rates.",
        "A completed cap and a complete natural response are different outcomes.",
        "The complete SLO grid is in `slo_attainment_by_rate.csv`.",
        "Paired attainment differences are in `serving_paired_differences.csv`.",
        "These are measured outcomes, not a simulated scheduler or a live adaptive result.",
        "",
    ]
    (ROOT / "serving_tables.md").write_text("\n".join(lines))

    quantum = [
        r
        for r in read("long_quantum_response_surface.csv")
        if r["kind"] == "PREFILL_CORE"
    ]
    profile = [
        "# Separate heavy quantum tables",
        "",
        "Evidence: HEAVY_MEASURED, not clean serving latency.",
        "The stream interval includes staging, stream waits, and host launch gaps.",
        "It is not pure GPU compute or removable E2E time.",
        "Each shape/policy normally has one profile process. Submission counts are not independent samples.",
        "These rows select the new long prompt only. Background priming quanta remain in the full CSV.",
        "Full-C rows exclude the short final prompt chunk when the observed row count permits that filter.",
        "Mixed can combine decode and prefill rows. It is not a pure-G control.",
        "",
        table(
            [
                "P/N",
                "Policy",
                "C",
                "G",
                "GPU",
                "Submissions",
                "Mean/P95 stream ms",
                "Full-C mean ms",
                "Mean prompt/decode rows",
            ],
            [
                [
                    f"{r['P']}/{r['N']}",
                    r["mode"],
                    r["C"],
                    r["G"],
                    r["GPUs"],
                    r["submissions"],
                    f"{number(r['mean_stream_elapsed_ms'])}/{number(r['p95_stream_elapsed_ms'])}",
                    number(r["full_C_mean_stream_elapsed_ms"]),
                    f"{number(r['actual_prefill_rows_mean'])}/{number(r['actual_decode_rows_mean'])}",
                ]
                for r in quantum
            ],
        ),
        "",
        "The long-prompt subset is `long_quantum_response_surface.csv`.",
        "Prelude, background priming, and decode/coda records remain separate in `quantum_response_surface.csv`.",
        "No held-out online quantum-cost predictor is validated by this table.",
        "",
    ]
    (ROOT / "profiling_tables.md").write_text("\n".join(profile))
    print("REPORT_TABLES", len(selected), len(quantum))


if __name__ == "__main__":
    main()
