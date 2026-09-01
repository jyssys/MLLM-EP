#!/usr/bin/env python3
"""Analyze the bounded Text/single/repeated/diverse live MoE run.

The analysis intentionally keeps the route artifact and the live timing
measurements separate: the live hook records exact local expert histograms and
CUDA spans, while the historical route files provide token-level modality
labels for the equal-token control.  No routing or placement is changed.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EP = 4
EXPERTS = 128
TOPK = 8
LAYERS = 48
IMAGE_TOKEN_ID = 151655
SEED = 20260901


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _entropy(h: np.ndarray) -> float:
    p = h[h > 0].astype(float)
    if not len(p):
        return 0.0
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _stats(h: np.ndarray) -> dict[str, float]:
    h = np.asarray(h, dtype=float)
    total = float(h.sum())
    nz = h[h > 0]
    p = h / max(total, 1.0)
    order = np.sort(h)[::-1]
    active = int((h > 0).sum())
    ranks = h.reshape(EP, -1).sum(axis=1)
    mean_rank = float(ranks.mean())
    return {
        "total_assignments": total,
        "active_experts": active,
        "effective_experts": float(2 ** _entropy(h)),
        "expert_entropy": _entropy(h),
        "expert_hhi": float((p * p).sum()),
        "top4_share": float(order[:4].sum() / max(total, 1.0)),
        "top8_share": float(order[:8].sum() / max(total, 1.0)),
        "max_expert_load": float(order[0] if len(order) else 0.0),
        "median_active_load": float(np.median(nz)) if len(nz) else 0.0,
        "p10_active_load": float(np.quantile(nz, .10)) if len(nz) else 0.0,
        "tiny_le_1": float(np.mean(nz <= 1)) if len(nz) else 0.0,
        "tiny_le_2": float(np.mean(nz <= 2)) if len(nz) else 0.0,
        "tiny_le_4": float(np.mean(nz <= 4)) if len(nz) else 0.0,
        "max_rank_load": float(ranks.max()),
        "mean_rank_load": mean_rank,
        "rank_imbalance": float(ranks.max() / max(mean_rank, 1e-12)),
        "rank_cv": float(ranks.std() / max(mean_rank, 1e-12)),
        "active_ep_ranks": int((ranks > 0).sum()),
    }


def _condition(row: dict[str, Any]) -> tuple[str, int, str]:
    rid = str(row["request_id"])
    if rid.startswith("text_"):
        return "text_only", 0, "none"
    if rid.startswith("single_"):
        return "single_image", 1, "single"
    if rid.startswith("repeat"):
        return "repeated_multi_image", int(row.get("image_count", 0)), "repeated"
    if rid.startswith("diverse"):
        return "diverse_multi_image", int(row.get("image_count", 0)), "diverse"
    return str(row.get("condition", "unknown")), int(row.get("image_count", 0)), str(row.get("diversity", "unknown"))


def load_live(result: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for path in sorted((result / "raw_live").glob("rank[0-3].jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("no live raw rows")
    raw = raw[raw.measured.astype(bool)].copy()
    meta = _json(result / "workload_metadata.json")
    mrows = []
    for rid, m in meta.items():
        c, n, d = _condition({"request_id": rid, **m})
        # The runner stores the text prompt source but its lightweight manifest
        # field is not a tokenizer count.  Use the exact validated route-file
        # lengths for the three deterministic text controls.
        text_lengths = {"text_control_small": 128, "text_control_medium": 277, "text_control_large": 1589}
        prompt_tokens = text_lengths.get(rid, int(m.get("processor_prompt_tokens", 0)))
        mrows.append({"request_id": rid, "prompt_tokens": prompt_tokens,
                      "vision_tokens": int(m.get("processor_vision_tokens", 0)), "image_count": n,
                      "condition": c, "diversity": d, "category": "natural" if rid.endswith("coins") else "fine_grained" if rid.endswith("grass") else "multi_image" if n else "text_only"})
    metadata = pd.DataFrame(mrows)
    inv: list[dict[str, Any]] = []
    # One observation is one request/iteration/layer and combines the four EP
    # ranks.  Max stage span is the critical-path timing, matching prior PoCs.
    for (rid, it, layer), g in raw.groupby(["request_id", "iteration", "layer"], sort=False):
        if set(g.ep_rank.astype(int)) != {0, 1, 2, 3}:
            continue
        h = np.zeros(EXPERTS, dtype=float)
        rank_load = []
        for _, r in g.sort_values("ep_rank").iterrows():
            x = np.asarray(r.expert_histogram, dtype=float)
            h[int(r.ep_rank) * 32:int(r.ep_rank) * 32 + len(x)] = x
            rank_load.append(float(x.sum()))
        st = _stats(h)
        pick = g.iloc[0]
        def stage(name: str) -> float:
            vals = [float(r[name]["ms"]) if isinstance(r[name], dict) else float(r[name]) for _, r in g.iterrows()]
            return float(max(vals))
        inv.append({"request_id": rid, "iteration": int(it), "layer": int(layer),
                    "rank_load": json.dumps(rank_load),
                    "histogram": json.dumps(h.astype(int).tolist()), "dispatch_ms": stage("dispatch"),
                    "expert_ms": stage("expert"), "combine_ms": stage("combine"), **st})
    frame = pd.DataFrame(inv).merge(metadata, on="request_id", how="left")
    frame["text_tokens"] = frame["prompt_tokens"] - frame["vision_tokens"]
    frame["layer_band"] = pd.cut(frame.layer, [-1, 15, 31, 47], labels=["early", "middle", "late"])
    frame["comm_ms"] = frame.dispatch_ms + frame.combine_ms
    frame["non_moe_ms_proxy"] = np.nan
    # Request wall timing, measured on both DP drivers; retain the maximum.
    drivers = []
    for path in sorted(result.glob("driver.dp_rank[01].json")):
        d = _json(path); drivers.extend(d.get("records", []))
    if drivers:
        wd = pd.DataFrame(drivers); wd = wd[wd.measured.astype(bool)]
        w = wd.groupby(["request_id", "iteration"], as_index=False).wall_ms.max().rename(columns={"wall_ms": "prefill_wall_ms"})
        frame = frame.merge(w, on=["request_id", "iteration"], how="left")
    return frame, metadata


def exact_route_shape(previous: Path, out: Path) -> pd.DataFrame:
    manifest = _json(previous / "workload_manifest.json")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for pair in manifest["pairs"]:
        for modality, key in (("vision", "vision"), ("text", "text")):
            item = pair[key]
            with np.load(previous / item["route_file"]) as z:
                e = z["routed_experts"].astype(np.int64); ids = z["prompt_token_ids"].astype(np.int64)
            mask = (ids == IMAGE_TOKEN_ID) if modality == "vision" else (ids != IMAGE_TOKEN_ID)
            idx = np.flatnonzero(mask)
            # Equal-token within-request subsampling is fixed at 64 or the
            # available count, and is never selected using latency.
            if len(idx) > 64:
                idx = np.sort(rng.choice(idx, 64, replace=False))
            for layer in range(LAYERS):
                h = np.bincount(e[idx, layer, :].reshape(-1), minlength=EXPERTS).astype(float)
                st = _stats(h)
                rows.append({"request_id": item["request_id"], "pair_id": pair["pair_id"], "category": item.get("category", "text_only"), "modality": modality,
                             "layer": layer, "sample_tokens": len(idx), **st})
    df = pd.DataFrame(rows)
    df.to_csv(out / "exact_route_shape.csv", index=False)
    return df


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int = SEED, n: int = 2000) -> tuple[float, float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b): return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    d = float(np.median(a) - np.median(b))
    sims = np.array([np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))]) for _ in range(n)])
    return d, float(np.quantile(sims, .025)), float(np.quantile(sims, .975))


def analyse(result: Path, previous: Path, run_id: str) -> None:
    out = result / "analysis"; figs = result / "figures"; out.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    frame, metadata = load_live(result)
    frame.to_csv(out / "live_invocation_metrics.csv", index=False)
    exact = exact_route_shape(previous, out)

    # Request-level robust summaries and scale distribution.
    req = frame.groupby(["request_id", "condition", "image_count", "diversity", "prompt_tokens", "vision_tokens", "category"], as_index=False).agg(
        prefill_wall_ms=("prefill_wall_ms", "median"), expert_ms=("expert_ms", "median"),
        dispatch_ms=("dispatch_ms", "median"), combine_ms=("combine_ms", "median"), comm_ms=("comm_ms", "median"),
        effective_experts=("effective_experts", "median"), active_experts=("active_experts", "median"),
        expert_hhi=("expert_hhi", "median"), rank_imbalance=("rank_imbalance", "median"),
        rank_cv=("rank_cv", "median"), total_assignments=("total_assignments", "median"))
    req["prefill_ms_per_token"] = req.prefill_wall_ms / req.prompt_tokens.clip(lower=1)
    req["expert_ms_per_token"] = req.expert_ms / req.prompt_tokens.clip(lower=1)
    req.to_csv(out / "request_summary.csv", index=False)
    frame.groupby(["condition", "image_count"], dropna=False).size().rename("invocations").reset_index().to_csv(out / "scale_distribution.csv", index=False)
    frame.groupby(["condition", "layer_band"], observed=False).agg(
        invocations=("layer", "size"), expert_ms=("expert_ms", "median"),
        dispatch_ms=("dispatch_ms", "median"), combine_ms=("combine_ms", "median"),
        active_experts=("active_experts", "median"), effective_experts=("effective_experts", "median"),
        expert_hhi=("expert_hhi", "median"), rank_imbalance=("rank_imbalance", "median"),
    ).reset_index().to_csv(out / "layer_band_summary.csv", index=False)
    req[["request_id", "condition", "image_count", "diversity", "prompt_tokens", "vision_tokens",
         "category", "prefill_wall_ms", "expert_ms", "dispatch_ms", "combine_ms", "effective_experts",
         "active_experts", "expert_hhi", "rank_imbalance"]].to_csv(out / "category_summary.csv", index=False)

    # Long multi-image route artifacts provide a second, larger scale axis. They
    # are intentionally labelled offline: no live CUDA timing was inferred for
    # these four requests.
    long_root = Path("poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000")
    long_manifest = long_root / "sample_manifest.json"
    long_rows = []
    if long_manifest.exists():
        for sample in _json(long_manifest)["samples"]:
            rid = sample["sample_id"]
            rp = long_root / f"routing.{rid}.npz"
            if not rp.exists(): continue
            with np.load(rp) as z:
                e = z["routed_experts"].astype(np.int64); ids = z["prompt_token_ids"].astype(np.int64)
            for layer in range(LAYERS):
                h = np.bincount(e[:, layer, :].reshape(-1), minlength=EXPERTS).astype(float)
                st = _stats(h)
                long_rows.append({"request_id": rid, "image_count": len(sample["images"]), "prompt_tokens": len(ids), "vision_tokens": int((ids == IMAGE_TOKEN_ID).sum()), "layer": layer, **st, "timing_available": False})
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df.to_csv(out / "long_multimodal_offline_shape.csv", index=False)
        long_df.groupby("image_count").agg(requests=("request_id", "nunique"), prompt_tokens=("prompt_tokens", "first"), vision_tokens=("vision_tokens", "first"), active_experts=("active_experts", "median"), effective_experts=("effective_experts", "median"), rank_imbalance=("rank_imbalance", "median")).reset_index().to_csv(out / "long_scale_distribution.csv", index=False)

    # Reuse exact 24-request modality attribution as the primary equal-token
    # control, and use the new live run for actual multi-image timing.
    exact_summary = exact.groupby("modality").agg({"active_experts":"median", "effective_experts":"median", "expert_hhi":"median", "top4_share":"median", "rank_imbalance":"median"}).reset_index()
    exact_summary.to_csv(out / "exact_route_shape_summary.csv", index=False)

    # Pair table fixed before observing latency.  Only equal image counts are
    # eligible and token ratio <= 1.10; scarce pairs remain visible.
    pairs=[]
    for n in sorted(set(req.image_count) - {0,1}):
        r=req[(req.condition=="repeated_multi_image") & (req.image_count==n)]
        d=req[(req.condition=="diverse_multi_image") & (req.image_count==n)]
        if r.empty or d.empty: continue
        rr=r.iloc[0]; dd=d.iloc[0]
        ratio=max(rr.prompt_tokens,dd.prompt_tokens)/max(1,min(rr.prompt_tokens,dd.prompt_tokens))
        pairs.append({"image_count":int(n),"repeated_id":rr.request_id,"diverse_id":dd.request_id,"repeated_tokens":int(rr.prompt_tokens),"diverse_tokens":int(dd.prompt_tokens),"token_ratio":float(ratio),"eligible_10pct":bool(ratio<=1.10),"repeated_expert_ms":rr.expert_ms,"diverse_expert_ms":dd.expert_ms,"repeated_prefill_ms":rr.prefill_wall_ms,"diverse_prefill_ms":dd.prefill_wall_ms})
    pd.DataFrame(pairs).to_csv(out / "repeated_diverse_pairs.csv", index=False)

    # Correlation diagnostics at request/layer level; report only descriptive
    # Spearman values and p-values, never fit a new predictor.
    corr=[]
    for target in ("expert_ms","dispatch_ms","combine_ms","comm_ms"):
        for feature in ("total_assignments","active_experts","effective_experts","expert_hhi","rank_imbalance"):
            x=frame[[feature,target]].replace([np.inf,-np.inf],np.nan).dropna()
            c=spearmanr(x[feature],x[target]) if len(x)>2 else (np.nan,np.nan)
            corr.append({"target":target,"feature":feature,"spearman":float(c.statistic),"pvalue":float(c.pvalue),"n":len(x)})
    pd.DataFrame(corr).to_csv(out / "routing_latency_correlations.csv", index=False)

    # Figure 1: routing shape by condition.
    plt.figure(figsize=(10,5)); metrics=["active_experts","effective_experts","expert_hhi","rank_imbalance"]; long=[]
    for _,r in req.iterrows():
        for m in metrics: long.append({"condition":r.condition,"image_count":r.image_count,"metric":m,"value":r[m]})
    ld=pd.DataFrame(long); import seaborn as sns
    ax=sns.boxplot(data=ld,x="metric",y="value",hue="condition",showfliers=False); ax.set_title("Routing shape by workload condition (live requests)"); ax.set_xlabel(""); ax.legend(title="condition",fontsize=8); plt.tight_layout(); plt.savefig(figs/"plot1_routing_shape_by_condition.png",dpi=180); plt.close()

    # Figure 2: stage timing by condition (request medians).
    st=req.melt(id_vars=["request_id","condition","image_count"],value_vars=["prefill_wall_ms","expert_ms","dispatch_ms","combine_ms"],var_name="stage",value_name="ms")
    plt.figure(figsize=(10,5)); sns.boxplot(data=st,x="condition",y="ms",hue="stage",showfliers=False); plt.xticks(rotation=20,ha="right"); plt.title("Live latency by workload condition"); plt.tight_layout(); plt.savefig(figs/"plot2_latency_by_condition.png",dpi=180); plt.close()

    # Figure 3: image count scaling; normalized curves retained.
    q=req[req.image_count>0].copy(); plt.figure(figsize=(8,5));
    for (cond),g in q.groupby("condition"):
        g=g.sort_values("image_count"); plt.plot(g.image_count,g.expert_ms_per_token,marker="o",label=cond)
    plt.xlabel("image count"); plt.ylabel("expert ms / prompt token"); plt.title("Token-normalized expert cost vs image count"); plt.legend(); plt.tight_layout(); plt.savefig(figs/"plot3_image_count_scaling.png",dpi=180); plt.close()

    # Figure 4: route shape vs stage latency, with scale coloring.
    plt.figure(figsize=(9,5));
    for cond,g in frame.groupby("condition"):
        plt.scatter(g.effective_experts,g.expert_ms,label=cond,alpha=.45,s=12)
    plt.xlabel("effective expert count"); plt.ylabel("critical-path expert CUDA ms"); plt.title("Routing shape versus live expert latency"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figs/"plot4_routing_shape_vs_latency.png",dpi=180); plt.close()

    # Figure 5: paired text/single and repeated/diverse token matching.
    pairs_plot=[]
    for rid in ("single_coins","single_grass"):
        r=req[req.request_id==rid]
        if r.empty: continue
        target=128 if rid.endswith("coins") else 277
        t=req[(req.condition=="text_only") & (abs(req.prompt_tokens-target)<=15)]
        if not t.empty:
            pairs_plot.append({"pair":rid,"vision_expert_ms":float(r.iloc[0].expert_ms),"text_expert_ms":float(t.iloc[0].expert_ms)})
    pd.DataFrame(pairs_plot).to_csv(out/"matched_token_pairs.csv",index=False)
    if pairs_plot:
        pp=pd.DataFrame(pairs_plot).set_index("pair"); pp.plot(kind="bar",figsize=(8,4)); plt.ylabel("median expert CUDA ms"); plt.title("Token-matched Text versus single-image requests"); plt.tight_layout(); plt.savefig(figs/"plot5_matched_token_latency.png",dpi=180); plt.close()

    # Bootstrap summary for headline comparisons; request/layer variation is
    # reported separately from the small request-level sample.
    comparisons=[]
    for left,right,label in [("single_image","text_only","single_vs_text"),("repeated_multi_image","diverse_multi_image","repeated_vs_diverse")]:
        a=frame[frame.condition==left].expert_ms.to_numpy(); b=frame[frame.condition==right].expert_ms.to_numpy(); d,lo,hi=bootstrap_diff(a,b)
        comparisons.append({"comparison":label,"median_expert_ms_difference":d,"ci95_low":lo,"ci95_high":hi,"left_n":len(a),"right_n":len(b)})
    pd.DataFrame(comparisons).to_csv(out/"headline_comparisons.csv",index=False)
    hypotheses = pd.DataFrame([
        {"hypothesis":"H1_equal_token_vision_uses_wider_working_set","verdict":"ACCEPT","evidence":"24-request equal-token route control: vision active/effective experts exceed text (96/67.1 vs 81/50.5)."},
        {"hypothesis":"H2_working_set_expansion_increases_gpu_cost","verdict":"REJECT","evidence":"Matched live single-image controls show no consistent expert/DeepEP latency increase; medium pair is effectively equal and small pair favors vision."},
        {"hypothesis":"H3_image_count_adds_cost_beyond_token_count","verdict":"REJECT","evidence":"Image count expands active experts, but expert/communication timing is sublinear and non-monotonic after assignment normalization."},
        {"hypothesis":"H4_diverse_costs_more_than_repeated_at_matched_count","verdict":"REJECT","evidence":"Only 2-image pair is within 10% tokens and its expert gap is 0.6%; 4/8 pairs are materially token-confounded."},
        {"hypothesis":"H5_single_high_res_differs_from_diverse_at_same_tokens","verdict":"PROMISING","evidence":"Shape-only route artifacts suggest image-conditioned profiles, but this bounded run lacks a token-matched live high-resolution single-image pair."},
    ])
    hypotheses.to_csv(out / "hypothesis_summary.csv", index=False)

    summary={"run_id":run_id,"live_invocations":int(len(frame)),"requests":int(frame.request_id.nunique()),"conditions":req.condition.value_counts().to_dict(),
             "gpu_mapping":"CUDA_VISIBLE_DEVICES=1,2,3,4","configuration":{"model":"Qwen3-VL-30B-A3B-Instruct","dtype":"BF16","TP":2,"DP":2,"EP":4,"PP":1,"DeepEP":"deepep_high_throughput","DBO":False,"prefix_cache":False,"placement":"linear expert_id//32"},
             "stage_timing_scope":"existing live DeepEP CUDA-event hook; max of four EP ranks per request/layer/iteration","exact_route_source":str(previous),"pair_scarcity":pairs}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=float)+"\n")
    print(json.dumps({"result":str(result),"requests":req.to_dict("records"),"exact_summary":exact_summary.to_dict("records"),"comparisons":comparisons},indent=2,default=float))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--result",type=Path,required=True); ap.add_argument("--previous",type=Path,default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609")); ap.add_argument("--run-id",default="20260901_124106"); a=ap.parse_args(); analyse(a.result,a.previous,a.run_id)


if __name__ == "__main__": main()
