#!/usr/bin/env python3
"""Build the discovery report from the immutable analysis artifacts."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

RUN = Path("poc_flashvep/deepep_revalidation/results/multimodal_routing_ep_discovery_20260831_173000")
AN = RUN / "analysis"
REPORT = Path("poc_flashvep/reports/multimodal_routing_ep_discovery.md")
SEED = 20260831

def read(name):
    p = AN / name
    if not p.exists(): p = AN / (name + ".gz")
    return pd.read_csv(p)

def ci(values, n=2000):
    x = np.asarray(values, float); x=x[np.isfinite(x)]
    rng=np.random.default_rng(SEED)
    if not len(x): return (float("nan"),float("nan"))
    means=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(n)])
    return tuple(np.quantile(means,[.025,.975]))

def f(v, nd=4):
    return "NA" if not np.isfinite(v) else f"{v:.{nd}f}"

def main():
    b=read("boundary_transitions.csv"); c=read("boundary_negative_controls.csv"); s=read("spatial_pairs.csv")
    p=read("cross_layer_persistence.csv"); q=read("traffic_bursts.csv"); w=read("working_sets.csv")
    x=read("cross_image_consistency.csv"); d=read("transition_directionality.csv")
    man=json.loads((Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609")/"workload_manifest.json").read_text())
    rows=[]
    for pair in man["pairs"]:
        z=np.load(Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609")/pair["vision"]["route_file"])
        ids=z["prompt_token_ids"]; nv=int((ids==151655).sum()); rows.append((pair["vision"]["request_id"],pair["vision"]["category"],len(ids),nv,len(ids)-nv,nv/len(ids)))
    wd=pd.DataFrame(rows,columns=["request_id","category","tokens","vision_tokens","text_tokens","vision_ratio"])
    # request/layer paired boundary effects
    bm=b.groupby(["request_id","layer","type"])[["expert_distance","dest_jsd"]].mean().reset_index().pivot_table(index=["request_id","layer"],columns="type")
    bd={}
    for m in ["expert_distance","dest_jsd"]:
        for typ in ["TV","VT"]:
            z=(bm[m][typ]-bm[m]["TT"]).dropna().values
            bd[(m,typ)]=(float(np.mean(z)),float(np.median(z)),ci(z),float(np.mean(z>0)))
    # spatial paired random-minus relation means
    sm=s.groupby(["request_id","image_index","layer","relation"])[["dest_jsd","expert_jsd","expert_jaccard","dest_jaccard"]].mean().reset_index().pivot_table(index=["request_id","image_index","layer"],columns="relation")
    sd={}
    for m in ["dest_jsd","expert_jsd","expert_jaccard"]:
        z=(sm[m]["random"]-sm[m]["adjacent"]).dropna().values; sd[m]=(float(np.mean(z)),ci(z),float(np.mean(z>0)))
    pp=p.pivot_table(index=["request_id","layer"],columns="modality")
    pdiff={}
    for m in ["expert_jaccard","dest_jaccard","expert_top1_same","dest_top1_same"]:
        z=(pp[m]["vision"]-pp[m]["text"]).dropna().values; pdiff[m]=(float(np.mean(z)),ci(z))
    ww=w.pivot_table(index=["request_id","layer"],columns="modality")
    wdiff={}
    for m in ["unique_experts","effective_experts","expert_entropy","top4_fraction","top8_fraction"]:
        z=(ww[m]["vision"]-ww[m]["text"]).dropna().values; wdiff[m]=(float(np.mean(z)),ci(z))
    # robust directionality consistency: cosine to direction-specific mean, excluding zero vectors
    dirstats={}
    for typ,g in d.groupby("direction"):
        vec=g[["dR0","dR1","dR2","dR3"]].to_numpy(float); mean=vec.mean(0); den=np.linalg.norm(vec,axis=1)*np.linalg.norm(mean)
        cos=np.divide(vec@mean,den,out=np.zeros_like(den),where=den>1e-12)
        dirstats[typ]=(float(np.linalg.norm(mean)),float(g.delta_l2.mean()),float(np.mean(cos)),int(len(g)))
    out=[]
    out.append("# Multimodal Routing × EP Discovery (offline PoC)\n")
    out.append("## Executive conclusion\n")
    out.append("The strongest reproducible signals are (i) a modality-boundary expert-routing shock and (ii) 2-D spatial locality in visual routing. Cross-layer persistence and visual traffic bursts were not supported. This is a discovery result, not an optimization implementation, and no new GPU run was required.\n")
    out.append("**Primary recommendation:** run one bounded live DeepEP instrumentation study that tags dispatch/combine timing by the already identified boundary and spatial-window regimes; do not implement a scheduler until that attribution exists.\n")
    out.append("## Candidate scorecard\n")
    out.append("Scores are ordinal 0–3 (effect, consistency, MLLM-specificity, MoE-specificity, EP relevance, control robustness, optimization potential, novelty potential), used to rank candidates rather than as a statistical gate.\n\n")
    out.append("| Candidate phenomenon | Effect size | Consistency | MLLM-specific | MoE-specific | EP relevance | Verdict |\n|---|---:|---:|---:|---:|---:|---|\n")
    out.append("| Modality-boundary expert-routing shock (A) | 3 | 3 | 3 | 3 | 2 | STRONG |\n")
    out.append("| Spatial locality × expert/EP routing (C) | 3 | 3 | 3 | 3 | 3 | STRONG |\n")
    out.append("| Image-conditioned visual working-set expansion / cross-image inconsistency (F/G) | 2 | 3 | 3 | 3 | 2 | PROMISING |\n")
    out.append("| Local visual EP traffic burst (D) | 0 | 1 | 2 | 2 | 1 | REJECT |\n")
    out.append("| Cross-layer visual persistence (B) | 0 | 1 | 2 | 2 | 1 | REJECT |\n")
    out.append("| Directional EP migration (H) | 1 | 1 | 2 | 2 | 1 | WEAK |\n")
    out.append("\nTop 3: boundary shock; spatially local but globally image-conditioned routing; modality-dependent expert working-set expansion.\n")
    out.append("## Configuration and provenance\n")
    out.append("* Analysis branch/base at start: `flashvep/multimodal-routing-ep-discovery`, HEAD `51152d5c5c4b179bf190b2d7c1e5b9cee4649631`.\n* Route source: `poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/`, 24 image requests plus paired text controls, 48 layers, top-8, 128 experts, EP4 linear map `expert_id // 32`, image token ID `151655`.\n* Historical capture provenance was TP2/DP2/EP4, BF16, DeepEP high-throughput, DBO off, eager; that artifact was captured on physical GPUs 4–7. The requested mapping for any live follow-up is `CUDA_VISIBLE_DEVICES=1,2,3,4`; this run was offline CPU-only and used no GPU.\n* Spatial metadata: `tile_slack_mechanism_20260820_150852/stage_a/sample_manifest.json`, using `token_span`, `post_merge_grid_hw`, and image boundaries; no hard-coded 784/grid shape.\n* Exact analysis command: `python poc_flashvep/multimodal_routing_ep_discovery/analyze.py --run-id 20260831_173000`. Fixed seed `20260831`; fixed spatial pair cap 128/relation/image/layer; fixed window sizes 32/64/128; bootstrap seed and policy are recorded in code.\n")
    out.append("## Workload and controls\n")
    out.append(wd.groupby("category").agg(requests=("request_id","count"),median_tokens=("tokens","median"),median_vision_ratio=("vision_ratio","median")).round(4).to_markdown()+"\n\n")
    out.append(f"Across 24 image requests, visual tokens have median ratio {wd.vision_ratio.median():.3f} (range {wd.vision_ratio.min():.3f}–{wd.vision_ratio.max():.3f}); the internal Text comparison always uses non-image tokens from the same image-containing request. Paired text-only routes are diagnostic controls for arbitrary boundaries/working sets, not the primary modality label.\n\n")
    out.append("## PoC A — modality-boundary routing transition\n")
    out.append("For adjacent-token expert-set distance (`1 − Jaccard`), medians are: **TT " + f(b.groupby("type").expert_distance.median()["TT"]) + ", VV " + f(b.groupby("type").expert_distance.median()["VV"]) + ", TV " + f(b.groupby("type").expert_distance.median()["TV"]) + ", VT " + f(b.groupby("type").expert_distance.median()["VT"]) + "**. EP-rank JSD medians are TT " + f(b.groupby("type").dest_jsd.median()["TT"]) + ", VV " + f(b.groupby("type").dest_jsd.median()["VV"]) + ", TV " + f(b.groupby("type").dest_jsd.median()["TV"]) + ", VT " + f(b.groupby("type").dest_jsd.median()["VT"]) + ".\n\n")
    out.append(f"Request/layer paired expert-distance difference vs TT: TV mean +{bd[("expert_distance","TV")][0]:.4f} (bootstrap 95% CI [{bd[("expert_distance","TV")][2][0]:.4f}, {bd[("expert_distance","TV")][2][1]:.4f}], positive in {bd[("expert_distance","TV")][3]*100:.1f}%), VT mean +{bd[("expert_distance","VT")][0]:.4f} (CI [{bd[("expert_distance","VT")][2][0]:.4f}, {bd[("expert_distance","VT")][2][1]:.4f}], positive in {bd[("expert_distance","VT")][3]*100:.1f}%).\n\n")
    out.append(f"The paired text-only arbitrary-boundary control has median distance {c[c.boundary_type=="synthetic_text"].expert_distance.median():.4f} and median destination JSD {c[c.boundary_type=="synthetic_text"].dest_jsd.median():.4f}, below the image boundary values. The label-shuffle control has median distance {c[c.boundary_type=="cross_shuffled"].expert_distance.median():.4f}; it removes modality alignment while preserving counts.\n\n")
    out.append("Interpretation: a boundary shock is repeated at every image boundary and layer, with a larger and asymmetric vision-entry/exit expert-set change. Destination-rank changes are smaller than expert-ID changes, so this is strongest as a routing-state transition, not proof of a communication latency penalty. See `figures/plot1_modality_boundary_transition.png` and `plot7_transition_directionality.png`.\n\n")
    out.append("## PoC B — cross-layer persistence\n")
    out.append("Equal-cap samples give mean expert-set Jaccard Vision " + f(p.groupby("modality").expert_jaccard.mean()["vision"]) + " vs Text " + f(p.groupby("modality").expert_jaccard.mean()["text"]) + ", destination-set Jaccard Vision " + f(p.groupby("modality").dest_jaccard.mean()["vision"]) + " vs Text " + f(p.groupby("modality").dest_jaccard.mean()["text"]) + ". Vision−Text expert Jaccard difference is " + f(pdiff["expert_jaccard"][0]) + " (CI [" + f(pdiff["expert_jaccard"][1][0]) + ", " + f(pdiff["expert_jaccard"][1][1]) + "]); although the CI excludes zero, the absolute effect is only about 0.0023. Top-1 persistence is near zero for expert IDs and ~0.25 for rank IDs.\n\n")
    out.append("Verdict: **REJECT** as a useful Vision-specific persistence phenomenon; statistical detectability here does not imply a practically meaningful depth-temporal effect. `figures/plot2_cross_layer_persistence.png`.\n\n")
    out.append("## PoC C — 2-D spatial locality × expert routing\n")
    ss=s.groupby("relation")[["expert_jaccard","expert_jsd","dest_jsd"]].median(); out.append("Median metrics: adjacent expert Jaccard " + f(ss.loc["adjacent","expert_jaccard"]) + ", expert JSD " + f(ss.loc["adjacent","expert_jsd"]) + ", destination JSD " + f(ss.loc["adjacent","dest_jsd"]) + "; random expert Jaccard " + f(ss.loc["random","expert_jaccard"]) + ", expert JSD " + f(ss.loc["random","expert_jsd"]) + ", destination JSD " + f(ss.loc["random","dest_jsd"]) + ".\n\n")
    out.append(f"At the request/image/layer level, random-minus-adjacent destination-JSD mean is {sd['dest_jsd'][0]:.4f} (bootstrap CI [{sd['dest_jsd'][1][0]:.4f}, {sd['dest_jsd'][1][1]:.4f}], positive in {sd['dest_jsd'][2]*100:.1f}%). Random-minus-adjacent expert-JSD mean is {sd['expert_jsd'][0]:.4f} (positive in {sd['expert_jsd'][2]*100:.1f}%). The direction is present in early, middle, and late layer strata, while far-vs-random is weaker.\n\n")
    out.append("Verdict: **STRONG** routing-level candidate. It is a genuine 2-D visual structure coupled to expert and EP-rank routing, but live dispatch/combine attribution remains unmeasured. `figures/plot3_spatial_routing_locality.png` and `plot8_representative_ep_heatmap.png`.\n\n")
    out.append("## PoC D — spatial-region EP traffic burst\n")
    qb=q.groupby(["modality","window"])[["max_dest_fraction","dest_hhi","dest_entropy"]].median(); out.append(qb.to_markdown()+"\n\n")
    out.append("Visual windows are *less* destination-concentrated than text windows (for 32 tokens, max-rank fraction 0.0625 vs 0.0938; HHI 0.0247 vs 0.0386). Thus the proposed visual burst/whole-window concentration does not hold in these traces. Verdict: **REJECT**. `figures/plot4_spatial_region_ep_burst.png`.\n\n")
    out.append("## PoC F/G — working set and cross-image consistency\n")
    wg=w.groupby("modality")[["unique_experts","effective_experts","expert_entropy","top4_fraction","top8_fraction","ep_coverage"]].median(); out.append(wg.to_markdown()+"\n\n")
    out.append("With equal per-request token subsampling, Vision has median 70 unique experts vs Text 59, effective experts 54.72 vs 41.98, and lower top-4/top-8 concentration (Vision 0.181/0.306 vs Text 0.245/0.397). This expansion repeats across natural, chart/document, fine-grained, and multi-image categories; it is not only a token-count effect.\n\n")
    out.append("Across requests at equal 64-token samples, visual expert-histogram cosine is " + f(x.groupby("modality").cosine.mean()["vision"]) + " vs text " + f(x.groupby("modality").cosine.mean()["text"]) + ", and JSD is " + f(x.groupby("modality").jsd.mean()["vision"]) + " vs " + f(x.groupby("modality").jsd.mean()["text"]) + ". This means visual routing is local/structured but strongly image-content-conditioned rather than a single shared visual expert vocabulary. Verdict: **PROMISING** combined F/G phenomenon. `figures/plot5_working_set.png` and `plot6_cross_image_consistency.png`.\n\n")
    out.append("## PoC H — transition directionality\n")
    out.append("TV mean centered rank-migration norm/mean step " + f(dirstats["TV"][0]) + "/" + f(d[d.direction=="TV"].delta_l2.mean()) + "; VT " + f(dirstats["VT"][0]) + "/" + f(d[d.direction=="VT"].delta_l2.mean()) + ". Mean-vector norms are small relative to per-transition norms and cosine consistency is not stable enough to claim a repeated rank migration direction. Verdict: **WEAK**.\n\n")
    out.append("## PoC E — live EP execution latency\n")
    out.append("Not run. Existing artifacts have token-level routes and EP destination labels but not boundary/spatial-window-attributed dispatch/expert/combine timing. A full live run was intentionally avoided in this discovery pass because it would require new instrumentation and could not be interpreted without changing the bounded analysis protocol. Therefore this report does not claim a latency or makespan effect.\n\n")
    out.append("## Negative controls and limitations\n")
    out.append("* Label-shuffle and arbitrary text-boundary controls are included. Spatial pairs use a fixed random seed and equal relation caps; coordinate permutation control is represented by the random-pair baseline rather than a second coordinate permutation file.\n* No dense MLLM execution was run; Q1 (absence in Dense MLLM) is a theoretical boundary, not an empirical negative-control result. The text-only paired routes are MoE controls, not a dense-model control.\n* Expert IDs are mapped to EP ranks using the historical validated linear placement `expert_id//32`; alternative placement could change rank-level effects.\n* Route artifacts were historically captured on GPUs 4–7. This branch performed CPU-only analysis; any follow-up live command must use only physical GPUs 1–4.\n* Router outputs are top-k IDs without probabilities, so entropy is assignment/set entropy, not router-logit confidence.\n* CSVs are compressed where large; all exact top-k IDs are retained in `raw/per_token_layer.csv.gz`.\n")
    out.append("## Direct answers\n")
    out.append("1. **Beyond histogram?** Yes: modality-boundary position and 2-D adjacency predict expert-set/rank similarity; global visual working-set expansion is also repeatable.\n2. **Token/spatial/temporal structure?** Strong position/spatial signals; no useful Vision-specific cross-layer persistence.\n3. **EP destination?** Spatial adjacency changes destination-rank JSD modestly but consistently; visual burst concentration is refuted.\n4. **Best MLLM-specific candidate?** Spatially local yet image-conditioned expert/EP routing, with a repeatable modality-boundary shock. Dense absence is not directly tested.\n5. **Different optimization opportunity?** A future boundary/spatial-window-aware EP communication/phase mechanism is plausible, but no implementation is justified yet because live latency attribution is missing.\n6. **One next PoC:** bounded live Qwen3-VL run with `CUDA_VISIBLE_DEVICES=1,2,3,4`, tagging DeepEP dispatch/combine events by modality boundary and spatial window, with no routing or placement change.\n")
    out.append("## Artifact index\n")
    out.append("* Result directory: `poc_flashvep/deepep_revalidation/results/multimodal_routing_ep_discovery_20260831_173000/`\n* Figures: `plot1_modality_boundary_transition.png`, `plot2_cross_layer_persistence.png`, `plot3_spatial_routing_locality.png`, `plot4_spatial_region_ep_burst.png`, `plot5_working_set.png`, `plot6_cross_image_consistency.png`, `plot7_transition_directionality.png`, `plot8_representative_ep_heatmap.png`.\n* Analysis code: `poc_flashvep/multimodal_routing_ep_discovery/analyze.py`; report builder: `make_report.py`.\n")
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text("\n".join(out))
    print(REPORT)

if __name__=="__main__": main()
