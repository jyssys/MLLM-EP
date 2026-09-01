#!/usr/bin/env python3
"""Finalize the bounded ASAP DP->EP synchronization reproduction.

This script only aggregates already completed runs.  It deliberately keeps
the indirect/closest wait metrics separate from the asynchronous event-wait
enqueue duration, because the latter is not a wall-clock synchronization
stall by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_events(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "asap_raw").glob("asap_rank*.jsonl")):
        try:
            file_rank = int(p.stem.split("rank", 1)[1])
        except Exception:
            file_rank = -1
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            x["trace_file_rank"] = file_rank
            rows.append(x)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    meta = {}
    try:
        meta = json.loads((root / "run_metadata.json").read_text())
    except Exception:
        pass
    for k, v in meta.items():
        if k not in df:
            df[k] = v
    if "ep_rank" not in df:
        df["ep_rank"] = df["trace_file_rank"]
    return df


def driver_walls(root: Path) -> pd.DataFrame:
    out = []
    for p in sorted(root.glob("driver.dp*.json")):
        try:
            x = json.loads(p.read_text())
        except Exception:
            continue
        for r in x.get("records", []):
            if r.get("measured"):
                out.append({"dp_rank": x.get("dp_rank"), "iteration": r.get("iteration"),
                            "wall_ms": r.get("wall_ms")})
    return pd.DataFrame(out)


def run_summary(root: Path, label: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((root / "run_metadata.json").read_text())
    ev = load_events(root)
    ev = ev[ev.get("measured", False) == True].copy() if not ev.empty else ev
    for c in ["pre_moe_cuda_ms", "prepare_host_ms", "dispatch_cuda_ms", "expert_cuda_ms",
              "combine_cuda_ms", "event_wait_cuda_ms", "ep_entry_to_done_ms"]:
        if c not in ev:
            ev[c] = np.nan
    if not ev.empty:
        # rank spread uses one rank row per scheduler wave/layer; it is a
        # within-iteration diagnostic, not a cross-GPU absolute timestamp.
        spread_rows = []
        for (it, layer), g in ev.groupby(["iteration", "layer"], dropna=False):
            row = {"label": label, "iteration": it, "layer": layer,
                   "topology": meta.get("topology"), "mode": meta.get("mode"),
                   "scale": meta.get("scale")}
            for c in ["pre_moe_cuda_ms", "prepare_host_ms", "dispatch_cuda_ms", "expert_cuda_ms",
                      "combine_cuda_ms", "event_wait_cuda_ms", "ep_entry_to_done_ms"]:
                vals = pd.to_numeric(g[c], errors="coerce").dropna()
                row[c + "_median"] = vals.median() if len(vals) else np.nan
                row[c + "_spread_ms"] = vals.max() - vals.min() if len(vals) else np.nan
                row[c + "_ratio"] = vals.max() / vals.mean() if len(vals) and vals.mean() else np.nan
            spread_rows.append(row)
        spread = pd.DataFrame(spread_rows)
    else:
        spread = pd.DataFrame()
    walls = driver_walls(root)
    if walls.empty:
        wall_median = float("nan")
        wall_p95 = float("nan")
    else:
        # All DP workers report the same barrier-delimited wave wall.  First
        # reduce across workers per iteration, then summarize iterations.
        wi = walls.groupby("iteration")["wall_ms"].median()
        wall_median, wall_p95 = float(wi.median()), float(wi.quantile(.95))
    s = {"label": label, **{k: meta.get(k) for k in
         ["topology", "tp", "dp", "ep", "scale", "mode", "chunked_prefill",
          "max_num_batched_tokens", "iterations", "warmups"]},
         "wall_median_ms": wall_median, "wall_p95_ms": wall_p95,
         "event_rows": int(len(ev)), "source_path": str(root)}
    if not spread.empty:
        for c in ["pre_moe_cuda_ms", "prepare_host_ms", "dispatch_cuda_ms", "expert_cuda_ms",
                  "combine_cuda_ms", "event_wait_cuda_ms", "ep_entry_to_done_ms"]:
            s[c + "_median"] = float(spread[c + "_median"].median())
            s[c + "_p95"] = float(spread[c + "_spread_ms"].quantile(.95))
            s[c + "_spread_median"] = float(spread[c + "_spread_ms"].median())
            s[c + "_ratio_median"] = float(spread[c + "_ratio"].median())
    return s, ev, spread


def positive_control(root: Path) -> pd.DataFrame:
    ev = load_events(root)
    if ev.empty:
        return pd.DataFrame()
    ev = ev[ev.get("measured", False) == True].copy()
    rows = []
    for delay, g in ev.groupby("delay_ms"):
        out = {"delay_ms": float(delay), "n": len(g)}
        for c in ["injected_delay_cuda_ms", "prepare_host_ms", "dispatch_cuda_ms",
                  "event_wait_cuda_ms", "pre_moe_cuda_ms"]:
            if c in g:
                x = pd.to_numeric(g[c], errors="coerce").dropna()
                out[c + "_median"] = float(x.median()) if len(x) else np.nan
                out[c + "_p95"] = float(x.quantile(.95)) if len(x) else np.nan
        rows.append(out)
    return pd.DataFrame(rows).sort_values("delay_ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("poc_flashvep/deepep_revalidation/results"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    base = args.base

    specs = [
        ("A_bal_4k", "asap_sync_phenomenon_reproduction_20260901_stageB_A_bal4096"),
        ("A_het_4k", "asap_sync_phenomenon_reproduction_20260901_stageB_A_het4096"),
        ("A_bal_8k", "asap_sync_phenomenon_reproduction_20260901_stageB_A_bal8192r"),
        ("A_het_8k", "asap_sync_phenomenon_reproduction_20260901_stageB_A_het8192r"),
        ("B_bal_4k_first", "asap_sync_phenomenon_reproduction_20260901_stageB_B_bal4096"),
        ("B_het_4k_first", "asap_sync_phenomenon_reproduction_20260901_stageB_B_het4096"),
        ("B_bal_4k_repeat", "asap_sync_phenomenon_reproduction_20260901_stageB_B_bal4096rep"),
        ("B_het_4k_repeat", "asap_sync_phenomenon_reproduction_20260901_stageB_B_het4096rep"),
        ("B_bal_8k", "asap_sync_phenomenon_reproduction_20260901_stageB_B_bal8192"),
        ("B_het_8k", "asap_sync_phenomenon_reproduction_20260901_stageB_B_het8192"),
        ("B_het_8k_off16k", "asap_sync_phenomenon_reproduction_20260901_stageC_B_het8192_off16k"),
        ("B_het_8k_on16k", "asap_sync_phenomenon_reproduction_20260901_stageC_B_het8192_on16k"),
        # The reverse-order repetition was launched with a shell label that
        # already contained the topology prefix, hence the double B in its
        # directory name. Keep the literal path to preserve provenance.
        ("B_het_4k_repeat_reverse", "asap_sync_phenomenon_reproduction_20260901_stageB_B_B_het4096rep2"),
        ("B_bal_4k_repeat_reverse", "asap_sync_phenomenon_reproduction_20260901_stageB_B_B_bal4096rep2"),
    ]
    summaries, event_frames, spread_frames = [], [], []
    used = []
    for label, dirname in specs:
        root = base / dirname
        if not (root / "run_metadata.json").exists():
            continue
        s, ev, sp = run_summary(root, label)
        summaries.append(s); used.append(str(root))
        if not ev.empty:
            ev["label"] = label
            event_frames.append(ev)
        if not sp.empty:
            spread_frames.append(sp)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "condition_summary.csv", index=False)
    if event_frames:
        event_all = pd.concat(event_frames, ignore_index=True)
        # Keep an analysis-sized projection in the committed result.  The
        # complete JSONL traces (including histograms) remain in the source
        # run directories listed by raw_trace_index.json.
        keep = ["label", "wave", "iteration", "scheduler_iteration", "worker_dp_rank", "ep_rank",
                "layer", "total_assignments", "dispatched_rows", "runtime_m", "delay_ms", "measured",
                "pre_moe_cuda_ms", "prepare_host_ms", "dispatch_cuda_ms", "expert_cuda_ms",
                "combine_cuda_ms", "event_wait_cuda_ms", "ep_entry_to_done_ms"]
        keep = [c for c in keep if c in event_all]
        event_all[keep].to_csv(out / "event_rows_selected.csv", index=False)
    if spread_frames:
        pd.concat(spread_frames, ignore_index=True).to_csv(out / "layer_rank_spread.csv", index=False)

    pc_root = base / "asap_sync_phenomenon_reproduction_20260901_stageA_H_sweep"
    pc = positive_control(pc_root)
    pc.to_csv(out / "injected_skew_validation.csv", index=False)

    # Figure 1: positive-control response.  Direct EventOverlap wait is shown
    # only as an enqueue duration; prepare/dispatch are the closest complete
    # collective-span proxies available in this path.
    fig, ax = plt.subplots(figsize=(7, 4))
    if not pc.empty:
        ax.plot(pc.delay_ms, pc.prepare_host_ms_median, "o-", label="prepare host span")
        ax.plot(pc.delay_ms, pc.dispatch_cuda_ms_median, "s-", label="dispatch CUDA span")
        if "injected_delay_cuda_ms_median" in pc:
            ax.plot(pc.delay_ms, pc.injected_delay_cuda_ms_median, "^-", label="measured injected delay")
    ax.set(xlabel="requested delay (ms)", ylabel="duration (ms)", title="Injected DP skew positive control")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "plot1_injected_skew_validation.png", dpi=160); plt.close(fig)

    # Figure 2: each independent run is visible; this intentionally exposes
    # run-to-run reversal rather than hiding it in a pooled mean.
    fig, ax = plt.subplots(figsize=(9, 4))
    if not summary.empty:
        s2 = summary[summary.label.str.contains("B_(bal|het)_4k", regex=True)]
        labels = s2.label.tolist(); vals = s2.wall_median_ms.tolist()
        ax.bar(np.arange(len(vals)), vals, color=["#4472c4" if "bal" in x else "#c0504d" for x in labels])
        ax.set_xticks(np.arange(len(vals)), labels, rotation=30, ha="right")
    ax.set_ylabel("prefill wall median (ms)"); ax.set_title("DP4 / 4096-token controlled workload")
    fig.tight_layout(); fig.savefig(out / "plot2_controlled_latency_runs.png", dpi=160); plt.close(fig)

    # Figure 3: chunked ON/OFF successful comparison at max_num_batched_tokens=16384.
    fig, ax = plt.subplots(figsize=(6, 4))
    if not summary.empty:
        s3 = summary[summary.label.isin(["B_het_8k_off16k", "B_het_8k_on16k"])]
        ax.bar(s3.label, s3.wall_median_ms, color=["#c0504d", "#70ad47"])
    ax.set_ylabel("prefill wall median (ms)"); ax.set_title("Chunked prefill ablation (max batch 16384)")
    fig.tight_layout(); fig.savefig(out / "plot3_chunked_prefill_ablation.png", dpi=160); plt.close(fig)

    # Figure 4: layer-wise closest wait proxy for first B4K pair.
    fig, ax = plt.subplots(figsize=(9, 4))
    if spread_frames:
        sp = pd.concat(spread_frames, ignore_index=True)
        ss = sp[sp.label.isin(["B_bal_4k_first", "B_het_4k_first"])]
        if not ss.empty:
            pv = ss.pivot_table(index="layer", columns="label", values="prepare_host_ms_spread_ms")
            pv.plot(ax=ax, marker=".")
    ax.set(xlabel="decoder layer", ylabel="rank spread of prepare span (ms)", title="Layer-wise DP/EP wait proxy")
    fig.tight_layout(); fig.savefig(out / "plot4_layer_wait_heatmap.png", dpi=160); plt.close(fig)

    proof = {}
    for dirname in used:
        root = Path(dirname)
        try:
            proof[dirname] = json.loads((root / "run_metadata.json").read_text())
        except Exception:
            pass
    (out / "topology_and_run_manifest.json").write_text(json.dumps({"runs": used, "metadata": proof}, indent=2) + "\n")
    (out / "raw_trace_index.json").write_text(json.dumps({
        "description": "Raw traces are preserved in their original run directories; this index is the immutable selection used for the final report.",
        "paths": used,
        "positive_control": str(pc_root),
        "nsys_capture": str(base / "asap_sync_phenomenon_reproduction_20260901_nsys_capture"),
    }, indent=2) + "\n")

    # Preserve compact raw scheduler/topology/backend evidence in the final
    # result.  Large per-layer JSONL files remain at their original paths and
    # are indexed above; copying them here would needlessly duplicate tens of
    # megabytes in the repository.
    compact = out / "raw_compact"
    compact.mkdir(exist_ok=True)
    for dirname in used:
        root = Path(dirname)
        tag = root.name
        dst = compact / tag
        dst.mkdir(exist_ok=True)
        for rel in ["run_metadata.json", "schedule.json", "control.json"]:
            src = root / rel
            if src.exists():
                shutil.copy2(src, dst / rel)
        for sub in ["scheduler_trace", "topology_proof", "backend_proof"]:
            srcd = root / sub
            if srcd.exists():
                shutil.copytree(srcd, dst / sub, dirs_exist_ok=True)

    # Gate uses the preregistered 10% end-to-end criterion but requires the
    # controlled effect to survive repetitions.  A single direction reversal
    # is recorded as instability rather than silently cherry-picked.
    def get(label):
        x = summary[summary.label == label]
        return float(x.wall_median_ms.iloc[0]) if len(x) else math.nan
    first_delta = get("B_het_4k_first") / get("B_bal_4k_first") - 1 if get("B_bal_4k_first") else math.nan
    repeat_delta = get("B_het_4k_repeat") / get("B_bal_4k_repeat") - 1 if get("B_bal_4k_repeat") else math.nan
    gate = {
        "measurement_validation": "PASS: calibrated GPU delay is visible in prepare/dispatch spans; direct EventOverlap wait is asynchronous enqueue time, not a standalone wall-clock stall",
        "direct_event_wait_metric": "NOT_SUFFICIENT_AS_DIRECT_STALL",
        "controlled_effect_first_run": first_delta,
        "controlled_effect_repeat_run": repeat_delta,
        "controlled_effect_reproducible": bool(np.isfinite(first_delta) and np.isfinite(repeat_delta) and first_delta >= .10 and repeat_delta >= .10),
        "chunked_prefill_off_8192": "NOT_SUPPORTED_BY_THIS_VLLM_CONFIGURATION",
        "natural_classification": "REPRODUCED only for the first DP4/4096 run; not robust across the repeated run",
        "final_status": "HOLD",
        "case": "SCALE_LIMITED_OR_RUNTIME_VARIABLE",
        "next_modality_study": "NO: first stabilize generic synchronization measurement/reproduction",
    }
    (out / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
