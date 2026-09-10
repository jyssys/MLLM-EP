#!/usr/bin/env python3
"""Analyze the matched N0/N1/N2 physical-split diagnostic.

N0 and N2 reuse the previously validated two-microbatch wrapper.  N1 uses the
same requests but executes the two exact fragments serially.  CUDA times are
joined only within a rank/device; a logical invocation takes the slowest rank.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICIES = ("N0_stock", "N1_sequential", "N2_concurrent")
WORKLOAD_IDS = {
    "coffee": "W2_standard",
    "method": "W3_vision_heavy",
    "coffee_rocket": "W4_multi_image",
}


def _load_rank_payloads(run: Path) -> list[dict]:
    payloads = [json.loads(path.read_text()) for path in sorted((run / "raw").glob("rank[0-9].json"))]
    if len(payloads) != 4:
        raise RuntimeError(f"expected four rank traces in {run}, found {len(payloads)}")
    for payload in payloads:
        if payload["visible_devices"] != "4,5,6,7":
            raise AssertionError(payload["visible_devices"])
    return payloads


def _logical_forward(run: Path, policy: str) -> pd.DataFrame:
    rows: list[dict] = []
    for payload in _load_rank_payloads(run):
        rank = int(payload["ep_rank"])
        for record in payload["forward_records"]:
            if not record.get("measured", record.get("phase") == "measured"):
                continue
            rows.append({"rank": rank, **record})
    frame = pd.DataFrame(rows)
    output: list[dict] = []
    for keys, group in frame.groupby(["request_id", "iteration", "wave"]):
        rank_spans = []
        for _, local in group.groupby("rank"):
            # start/end share an origin only within one device.
            rank_spans.append(float(local.end_ms.max() - local.start_ms.min()))
        output.append({
            "policy": policy,
            "request_id": keys[0],
            "iteration": int(keys[1]),
            "wave": int(keys[2]),
            "prefill_cuda_ms": max(rank_spans),
        })
    return pd.DataFrame(output)


def _logical_stage(run: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for payload in _load_rank_payloads(run):
        rank = int(payload["ep_rank"])
        for record in payload.get("stage_records", []):
            rows.append({"rank": rank, **record})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keys = ["request_id", "iteration", "wave", "layer", "segment", "stage"]
    # Same logical stage completes when its slowest rank completes.  Durations
    # are same-device event differences, never absolute cross-device deltas.
    return frame.groupby(keys, as_index=False).duration_ms.max()


def _driver_outputs(run: Path) -> pd.DataFrame:
    rows = []
    for rank in (0, 1):
        payload = json.loads((run / f"driver.dp_rank{rank}.json").read_text())
        # N2 completed every scheduled request and flushed CUDA/logit records,
        # then its auxiliary final flush hung during engine shutdown and was
        # interrupted.  Retain the completed records but never infer a clean
        # shutdown from them.
        if not payload.get("ok", False) and not payload.get("records"):
            raise RuntimeError(payload.get("traceback", f"failed driver {run} rank {rank}"))
        for row in payload["records"]:
            if row["phase"] == "correctness":
                rows.append({"driver_dp_rank": rank, **row})
    return pd.DataFrame(rows)


def _correctness(root: Path) -> pd.DataFrame:
    output_rows = []
    ref_driver = _driver_outputs(root / "N0_stock")
    for policy in POLICIES:
        driver = _driver_outputs(root / policy)
        merged = ref_driver.merge(driver, on=["driver_dp_rank", "request_id"], suffixes=("_ref", "_test"))
        for _, row in merged.iterrows():
            output_rows.append({
                "policy": policy,
                "request_id": row.request_id,
                "driver_dp_rank": int(row.driver_dp_rank),
                "greedy_token_agreement": row.output_tokens_ref == row.output_tokens_test,
            })
    for policy in POLICIES:
        ref = np.load(root / "N0_stock/raw/rank0.logits.npz", allow_pickle=True)
        test = np.load(root / f"{policy}/raw/rank0.logits.npz", allow_pickle=True)
        for index, request_id in enumerate(("coffee", "method", "coffee_rocket")):
            key = ("wave_4", "wave_9", "wave_14")[index]
            a = ref[key].astype(np.float64).reshape(-1)
            b = test[key].astype(np.float64).reshape(-1)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cosine = float(np.dot(a, b) / denom)
            rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(a))
            output_rows.append({
                "policy": policy,
                "request_id": request_id,
                "driver_dp_rank": 0,
                "logit_cosine": cosine,
                "logit_rel_l2": rel_l2,
                "logit_max_abs": float(np.max(np.abs(a - b))),
            })
    return pd.DataFrame(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    frames = [_logical_forward(args.root / policy, policy) for policy in POLICIES]
    forward = pd.concat(frames, ignore_index=True)
    forward.to_csv(args.root / "naive_forward_trace.csv", index=False)

    # The first measured iteration carries detailed event observation.  The
    # remaining two are the primary clean physical-split comparison.
    clean = forward[forward.iteration > 0]
    summary = clean.groupby(["policy", "request_id"], as_index=False).prefill_cuda_ms.agg(
        median_ms="median", min_ms="min", max_ms="max", reps="count"
    )
    wide = summary.pivot(index="request_id", columns="policy", values="median_ms").reset_index()
    wide["sequential_split_tax_pct"] = 100 * (wide.N1_sequential / wide.N0_stock - 1)
    wide["concurrent_vs_stock_pct"] = 100 * (wide.N2_concurrent / wide.N0_stock - 1)
    wide["concurrent_vs_sequential_pct"] = 100 * (wide.N2_concurrent / wide.N1_sequential - 1)

    stages = _logical_stage(args.root / "N1_sequential")
    stages.to_csv(args.root / "naive_sequential_stage_trace.csv", index=False)
    opportunity = []
    for request_id, local in stages[stages.iteration == 0].groupby("request_id"):
        piv = local.pivot_table(index="layer", columns=["segment", "stage"], values="duration_ms")
        required = [("tail", "attention"), ("prefix", "moe_total")]
        if not all(column in piv.columns for column in required):
            raise AssertionError((request_id, piv.columns.tolist()))
        overlap_ms = float(np.minimum(piv[("tail", "attention")], piv[("prefix", "moe_total")]).sum())
        opportunity.append({"request_id": request_id, "physical_overlap_opportunity_ms": overlap_ms})
    opportunity_frame = pd.DataFrame(opportunity)
    wide = wide.merge(opportunity_frame, on="request_id")
    wide["realized_overlap_saving_ms"] = wide.N1_sequential - wide.N2_concurrent
    wide["raw_overlap_efficiency"] = wide.realized_overlap_saving_ms / wide.physical_overlap_opportunity_ms
    wide["nonnegative_overlap_efficiency"] = wide.raw_overlap_efficiency.clip(lower=0, upper=1)

    oracle = pd.read_csv(args.root / "oracle_summary.csv")
    wide["workload_id"] = wide.request_id.map(WORKLOAD_IDS)
    wide = wide.merge(oracle[["workload_id", "clean_ttft_ms", "affected_base_ms", "O1_ms", "O4_ms"]],
                      on="workload_id", how="left")
    wide["ideal_O1_saving_ms"] = wide.affected_base_ms - wide.O1_ms
    wide["ideal_O4_saving_ms"] = wide.affected_base_ms - wide.O4_ms
    wide["ideal_O4_prefill_ms"] = wide.N0_stock - wide.ideal_O4_saving_ms
    wide["physical_split_tax_ms"] = wide.N1_sequential - wide.N0_stock
    # Conservative counterfactual: charge the observed exact split tax and
    # recover only the measured nonnegative fraction of ideal overlap.
    wide["contention_corrected_saving_ms"] = (
        wide.ideal_O4_saving_ms * wide.nonnegative_overlap_efficiency - wide.physical_split_tax_ms
    )
    wide["contention_corrected_ttft_gain_pct"] = 100 * wide.contention_corrected_saving_ms / wide.clean_ttft_ms
    wide.to_csv(args.root / "naive_policy_comparison.csv", index=False)

    correctness = _correctness(args.root)
    correctness.to_csv(args.root / "naive_correctness.csv", index=False)
    logit = correctness.dropna(subset=["logit_cosine"])
    tokens_ok = bool(correctness.greedy_token_agreement.dropna().all())
    summary_payload = {
        "scope": "bounded physical split diagnostic; one engine restart per policy, two clean repetitions per workload",
        "greedy_token_agreement_all": tokens_ok,
        "minimum_logit_cosine": float(logit.logit_cosine.min()),
        "maximum_logit_relative_l2": float(logit.logit_rel_l2.max()),
        "median_sequential_split_tax_pct": float(wide.sequential_split_tax_pct.median()),
        "median_concurrent_vs_stock_pct": float(wide.concurrent_vs_stock_pct.median()),
        "median_raw_overlap_efficiency": float(wide.raw_overlap_efficiency.median()),
        "median_contention_corrected_ttft_gain_pct": float(wide.contention_corrected_ttft_gain_pct.median()),
        "contention_gate": "PASS" if float(wide.contention_corrected_ttft_gain_pct.median()) >= 5 else "FAIL",
    }
    (args.root / "naive_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

    figures = args.root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    policy_plot = summary.pivot(index="request_id", columns="policy", values="median_ms")
    policy_plot.insert(0, "ideal_O4", wide.set_index("request_id").ideal_O4_prefill_ms)
    policy_plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Clean prefill CUDA span (ms)")
    fig.tight_layout(); fig.savefig(figures / "06_N0_N1_N2_prefill.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    wide.set_index("request_id")[["sequential_split_tax_pct", "concurrent_vs_stock_pct"]].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Change from stock (%)")
    fig.tight_layout(); fig.savefig(figures / "07_split_tax_contention.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    wide.set_index("request_id")[["raw_overlap_efficiency", "nonnegative_overlap_efficiency"]].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Physical overlap efficiency")
    fig.tight_layout(); fig.savefig(figures / "08_overlap_efficiency.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    oracle_plot = wide.set_index("request_id")[["ideal_O4_saving_ms", "contention_corrected_saving_ms"]].copy()
    oracle_plot.columns = ["Ideal O4", "Contention-corrected"]
    oracle_plot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Projected TTFT saving (ms)")
    fig.tight_layout(); fig.savefig(figures / "11_ideal_vs_corrected_oracle.png", dpi=180); plt.close(fig)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
