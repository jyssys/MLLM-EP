#!/usr/bin/env python3
"""Analyze corrected fixed-shape online MoE stage observations.

This is deliberately a read-only analyzer.  It never joins CUDA clocks across
devices: all stage durations come from same-device event pairs in the hook.
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
import numpy as np
import pandas as pd

STAGES = ["deepep_layout", "deepep_dispatch", "expert", "deepep_combine"]

def load(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as fh:
        for line in fh:
            try: x = json.loads(line)
            except Exception: continue
            d = {s: np.nan for s in STAGES}
            streams = {}
            for z in x.get("stage_records", []):
                if z.get("stage") in d:
                    d[z["stage"]] = z.get("cuda_ms")
                    streams[z["stage"]] = z.get("stream_id")
            row = {k: x.get(k) for k in (
                "timestamp_ns", "local_invocation_id", "scheduler_iteration_id",
                "route_id", "layer", "dp_rank", "ep_rank", "ep_size", "phase",
                "M", "top_k", "total_assignments", "active_experts",
                "rank_max_mean", "expert_max_mean", "expert_cv", "expert_hhi",
                "expert_entropy", "fanout_mean", "fanout_f4", "wall_ms", "cuda_ms",
                "request_context")}
            row.update(d)
            row["stream_ids"] = ",".join(f"{k}:{v}" for k,v in streams.items())
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty: return df
    for c in ["M","layer","dp_rank","ep_rank","local_invocation_id"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dispatch_plus_combine_ms"] = df["deepep_dispatch"].fillna(0) + df["deepep_combine"].fillna(0)
    df["stage_sum_ms"] = df[STAGES].sum(axis=1)
    # A local previous-state join (same device/rank, no cross-device clocks).
    df = df.sort_values(["dp_rank","ep_rank","local_invocation_id","layer"]).reset_index(drop=True)
    for c in ["cuda_ms", "deepep_dispatch", "expert", "deepep_combine", "M"]:
        df["prev_"+c] = df.groupby(["dp_rank","ep_rank","phase"])[c].shift(1)
    return df

def summarize(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if df.empty: return {"rows":0}
    df.to_csv(out/"stage_rows.csv", index=False)
    # Same-shape medians are the reference for tail localization.
    gcols = ["phase","M","layer","ep_rank"]
    med = df.groupby(gcols, dropna=False)[STAGES+['cuda_ms']].median().add_prefix("med_").reset_index()
    df = df.merge(med, on=gcols, how="left")
    for s in STAGES+['cuda_ms']:
        df["ratio_"+s] = df[s] / df["med_"+s].replace(0, np.nan)
    df["tail_threshold"] = df["med_cuda_ms"] * 1.15
    df["tail"] = df["cuda_ms"] >= df["tail_threshold"]
    # Extreme tails: p99 within fixed shape, useful for giant dispatch spikes.
    p99 = df.groupby(gcols, dropna=False)["cuda_ms"].quantile(.99).rename("p99_cuda_ms").reset_index()
    df = df.merge(p99, on=gcols, how="left")
    df["extreme_tail"] = df["cuda_ms"] >= df["p99_cuda_ms"]
    def quant(s):
        a = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        if not len(a): return {"n":0}
        return {"n":int(len(a)),"p50_ms":float(np.percentile(a,50)),"p90_ms":float(np.percentile(a,90)),"p99_ms":float(np.percentile(a,99)),"max_ms":float(np.max(a)),"cv_pct":float(np.std(a)/np.mean(a)*100) if np.mean(a) else None}
    summary = {}
    for (phase,M), part in df.groupby(["phase","M"], dropna=False):
        key=f"{phase}_M{int(M)}"
        summary[key]={"whole":quant(part["cuda_ms"]),"stages":{s:quant(part[s]) for s in STAGES},"tail_count":int(part["tail"].sum()),"extreme_count":int(part["extreme_tail"].sum())}
    # First divergence: stage with the largest normalized excess per tail row.
    tail = df[df["tail"]].copy()
    if not tail.empty:
        ratios = tail[["ratio_"+s for s in STAGES]].to_numpy()
        tail["first_stage"] = [STAGES[int(np.nanargmax(r))] if np.isfinite(r).any() else "UNKNOWN" for r in ratios]
        tail[["phase","M","layer","dp_rank","ep_rank","local_invocation_id","cuda_ms","p99_cuda_ms","first_stage"]+STAGES+['prev_cuda_ms','prev_deepep_dispatch','prev_expert','prev_deepep_combine']].to_csv(out/"tail_cases.csv",index=False)
    else:
        pd.DataFrame(columns=["phase","M","layer","dp_rank","ep_rank","first_stage"]+STAGES).to_csv(out/"tail_cases.csv",index=False)
    # Cross-rank co-occurrence within a DP invocation/layer.
    keys=["dp_rank","local_invocation_id","layer","phase","M"]
    cr=[]
    for k,part in df.groupby(keys,dropna=False):
        active=int(part["tail"].sum()); n=int(len(part));
        cr.append(dict(zip(keys,k), n_ranks=n, tail_ranks=active, pattern=("one_rank" if active<=1 else "several_ranks" if active<n else "all_ranks"), max_cuda_ms=float(part.cuda_ms.max()), max_dispatch_ms=float(part.deepep_dispatch.max())))
    pd.DataFrame(cr).to_csv(out/"cross_rank_patterns.csv",index=False)
    # State comparison: tails vs non-tails using same-device lagged values.
    state=[]
    for label,part in df.groupby("tail"):
        r={"tail":bool(label),"n":int(len(part))}
        for c in ["prev_cuda_ms","prev_deepep_dispatch","prev_expert","prev_deepep_combine","M","active_experts","rank_max_mean"]:
            r[c+"_median"]=float(part[c].median()) if c in part and part[c].notna().any() else None
        state.append(r)
    pd.DataFrame(state).to_csv(out/"previous_state_summary.csv",index=False)
    df.to_csv(out/"stage_rows_labeled.csv",index=False)
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("trace",type=Path); ap.add_argument("out",type=Path)
    a=ap.parse_args(); s=summarize(load(a.trace),a.out); (a.out/"summary.json").write_text(json.dumps(s,indent=2)+"\n")
    print(json.dumps(s,indent=2))
if __name__=="__main__": main()
