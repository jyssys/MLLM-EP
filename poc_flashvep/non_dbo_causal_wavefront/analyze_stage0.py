"""Apply the preregistered A-vs-S zero-contention W gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(root: Path, variant: str):
    directory = root / variant
    driver = []
    for rank in range(2):
        payload = json.loads((directory / f"driver.dp_rank{rank}.json").read_text())
        if not payload["ok"]:
            raise RuntimeError(payload)
        driver.extend(payload["records"])
    forwards, stages, proofs = [], [], []
    for rank in range(4):
        payload = json.loads((directory / "raw" / f"rank{rank}.json").read_text())
        if payload["visible_devices"] != "1,2,3,4" or payload["dbo_configured"]:
            raise RuntimeError(payload)
        proofs.append(payload)
        forwards.extend({**row, "ep_rank": rank} for row in payload["forward_records"])
        stages.extend({**row, "ep_rank": rank} for row in payload["stage_records"])
    return pd.DataFrame(driver), pd.DataFrame(forwards), pd.DataFrame(stages), proofs


def _latencies(forwards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, local in forwards.groupby(
        ["wave", "request_id", "phase", "iteration"], sort=True
    ):
        values = []
        for _, rank_rows in local.groupby("ep_rank"):
            values.append(float(rank_rows.end_ms.max() - rank_rows.start_ms.min()))
        rows.append(
            dict(zip(("wave", "request_id", "phase", "iteration"), keys, strict=True))
            | {"latency_ms": max(values)}
        )
    return pd.DataFrame(rows)


def _critical_stage(stages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["request_id", "stage", "layer", "segment"]
    for values, local in stages.groupby(keys):
        spans = [
            float(rank_rows.end_ms.max() - rank_rows.start_ms.min())
            for _, rank_rows in local.groupby("ep_rank")
        ]
        rows.append(dict(zip(keys, values, strict=True)) | {"duration_ms": max(spans)})
    return pd.DataFrame(rows)


def _oracle(stage: pd.DataFrame, request_id: str) -> float:
    local = stage[stage.request_id == request_id]
    finish_tail_moe = 0.0
    finish_prefix_moe = 0.0
    for layer in range(48):

        def duration(segment: str, name: str) -> float:
            row = local[
                (local.layer == layer)
                & (local.segment == segment)
                & (local.stage == name)
            ]
            if len(row) != 1:
                raise AssertionError((request_id, layer, segment, name, len(row)))
            return float(row.iloc[0].duration_ms)

        prefix_attn_end = finish_prefix_moe + duration("prefix", "attention")
        tail_attn_end = max(prefix_attn_end, finish_tail_moe) + duration(
            "tail", "attention"
        )
        prefix_moe_start = max(prefix_attn_end, finish_tail_moe)
        finish_prefix_moe = prefix_moe_start + duration("prefix", "moe_total")
        finish_tail_moe = max(tail_attn_end, finish_prefix_moe) + duration(
            "tail", "moe_total"
        )
    return max(finish_prefix_moe, finish_tail_moe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    drivers, forwards, stages, proofs = {}, {}, {}, {}
    for variant in ("A", "S"):
        drivers[variant], forwards[variant], stages[variant], proofs[variant] = _load(
            args.result_dir, variant
        )
    latency = {variant: _latencies(forwards[variant]) for variant in ("A", "S")}
    measured = pd.concat(
        [
            table[table.phase == "measured"].assign(variant=variant)
            for variant, table in latency.items()
        ],
        ignore_index=True,
    )
    stage = {variant: _critical_stage(stages[variant]) for variant in ("A", "S")}
    requests = ("coins", "histology", "method")
    per_request = []
    for request_id in requests:
        a = float(
            measured[
                (measured.variant == "A") & (measured.request_id == request_id)
            ].latency_ms.median()
        )
        s = float(
            measured[
                (measured.variant == "S") & (measured.request_id == request_id)
            ].latency_ms.median()
        )
        oracle = _oracle(stage["S"], request_id)
        per_request.append(
            {
                "request_id": request_id,
                "A_ms": a,
                "S_ms": s,
                "split_overhead": s / a,
                "W_zero_contention_oracle_ms": oracle,
                "A_to_oracle_speedup": a / oracle,
            }
        )
    frame = pd.DataFrame(per_request)
    median_headroom = float(frame.A_to_oracle_speedup.median())
    status = "PASS" if median_headroom >= 1.10 else "NO-GO"
    correctness = {}
    for variant in ("A", "S"):
        rows = drivers[variant]
        correctness[variant] = all(
            len({tuple(value) for value in group.output_tokens}) == 1
            for _, group in rows[rows.phase == "correctness"].groupby("wave")
        )
    summary = {
        "STAGE0_STATUS": status,
        "median_A_ms": float(measured[measured.variant == "A"].latency_ms.median()),
        "median_S_ms": float(measured[measured.variant == "S"].latency_ms.median()),
        "median_split_overhead": float(
            measured[measured.variant == "S"].latency_ms.median()
            / measured[measured.variant == "A"].latency_ms.median()
        ),
        "median_A_to_W_oracle_speedup": median_headroom,
        "per_request": per_request,
        "correctness_dp_repeatability": correctness,
        "dbo_all_variants": False,
        "W_executed": False,
    }
    (args.result_dir / "stage0_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    frame.to_csv(args.result_dir / "stage0_per_request.csv", index=False)
    figure = args.result_dir / "figures"
    figure.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(frame))
    ax.bar(x - 0.25, frame.A_ms, width=0.25, label="A stock")
    ax.bar(x, frame.S_ms, width=0.25, label="S sequential split")
    ax.bar(
        x + 0.25,
        frame.W_zero_contention_oracle_ms,
        width=0.25,
        label="W zero-contention oracle",
    )
    ax.set_xticks(x, frame.request_id)
    ax.set_ylabel("Prefill latency / critical-path estimate (ms)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure / "plot1_stage0_split_oracle.png", dpi=180)
    plt.close(fig)

    rows = [
        "# Non-DBO Causal Stage-Wavefront",
        "",
        f"`NON_DBO_CAUSAL_WAVEFRONT: {'NOT-YET' if status == 'PASS' else 'NO-GO'}`",
        "",
        "## Stage 0 gate",
        "",
        f"- A median: {summary['median_A_ms']:.4f} ms.",
        f"- S median: {summary['median_S_ms']:.4f} ms.",
        f"- Split overhead S/A: {summary['median_split_overhead']:.4f}×.",
        f"- Median zero-contention A→W oracle speedup: {median_headroom:.4f}×.",
        f"- Gate: **{status}** (W requires >=1.10×).",
        "- DBO is disabled in both variants; S uses one host owner thread and one compute stream.",
        "",
        "| Request | A ms | S ms | S/A | W oracle ms | A/W oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in per_request:
        rows.append(
            f"| {item['request_id']} | {item['A_ms']:.4f} | {item['S_ms']:.4f} | "
            f"{item['split_overhead']:.4f}× | {item['W_zero_contention_oracle_ms']:.4f} | "
            f"{item['A_to_oracle_speedup']:.4f}× |"
        )
    rows.extend(
        [
            "",
            "## Decision",
            "",
            (
                "Stage 0 passes; implement and measure W without changing the preregistered schedule."
                if status == "PASS"
                else "Even the zero-contention stage-wavefront oracle cannot provide 1.10× median headroom over stock. Per the preregistered stop condition, W is not implemented or executed."
            ),
            "",
            f"- Correctness DP repeatability A/S: {correctness['A']}/{correctness['S']}.",
            f"- Result directory: `{args.result_dir}`.",
        ]
    )
    args.report.write_text("\n".join(rows) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
