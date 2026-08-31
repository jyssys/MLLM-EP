#!/usr/bin/env python3
"""Offline chunk-boundary feasibility analysis for Qwen3-VL route traces.

Only token cut points are changed in the analysis.  Token order, top-k expert
IDs, and assignment multiplicity are immutable inputs.  No model or GPU is
required.
"""
from __future__ import annotations
import argparse, csv, gzip, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260831
RANDOM_SEEDS = tuple(range(10))
BUDGETS = (128, 256, 512, 1024)
LAYERS = 48; EXPERTS = 128; EP = 4; TOPK = 8; IMAGE_ID = 151655
SPATIAL_MANIFEST = Path("poc_flashvep/deepep_revalidation/results/tile_slack_mechanism_20260820_150852/stage_a/sample_manifest.json")
ROUTE_BASE = Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609")

def entropy(h):
    h=np.asarray(h,float); p=h[h>0]; p=p/max(h.sum(),1e-12)
    return float(-np.sum(p*np.log2(p))) if len(p) else 0.0

def gini(h):
    h=np.asarray(h,float); den=2*len(h)*max(h.sum(),1e-12)
    return float(np.abs(h[:,None]-h[None,:]).sum()/den)

def block_m(n):
    # vLLM 0.20 default BF16 TritonExperts heuristic (source audited in report).
    return 16 if n<=32 else 32 if n<=96 else 64 if n<=512 else 128

def chunks_fixed(n,b):
    out=list(range(0,n,b))
    if not out or out[-1] != n: out.append(n)
    return out

def valid_range(start, n, b):
    lo=start+max(1,int(math.ceil(.75*b))); hi=min(n,start+int(math.floor(1.25*b)))
    return lo,hi

def choose_boundary(n,b,start,bounds):
    lo,hi=valid_range(start,n,b); cand=[x for x in bounds if lo<=x<=hi]
    if not cand: return min(n,start+b)
    return min(cand,key=lambda x:(abs(x-(start+b)),x))

def chunks_boundary(n,b,bounds):
    out=[0]; start=0
    while start<n:
        end=choose_boundary(n,b,start,bounds)
        if end<=start: end=min(n,start+b)
        out.append(end); start=end
    return out

def chunk_cost(e, ids, mask, st, en, scope="vision"):
    x=e[st:en]
    if scope=="vision": x=x[mask[st:en]]
    if len(x)==0:
        return {"token_count":0,"assignments":0,"active_experts":0,"effective_experts":0.0,"median_expert_batch":0.0,"p10_expert_batch":0.0,"le1":0.0,"le2":0.0,"le4":0.0,"effective_tiles":0,"full_tiles":0,"tail_experts":0,"padding_rows":0,"padding_fraction":0.0,"fragmentation":0.0,"max_rank_load":0,"rank_max_mean":0.0,"rank_cv":0.0,"modality_mix":0.0}
    h=np.bincount(x.reshape(-1),minlength=EXPERTS); nz=h[h>0]
    # TritonExperts chooses BLOCK_SIZE_M from the complete invocation M
    # (the unsplit chunk), even when this row is a vision-only attribution.
    # Keep the attribution histogram visual-only, but use the actual chunk M
    # for the tile/padding proxy.
    bm=block_m(en-st); q=np.ceil(h[h>0]/bm).astype(int)
    rh=np.bincount((x//32).reshape(-1),minlength=EP); mean=rh.mean()
    # Modality mix is calculated on the original token span, not assignment rows.
    localmask=mask[st:en]; mix=float(min(localmask.sum(),len(localmask)-localmask.sum())/max(1,len(localmask)))
    return {"token_count":int(len(x)),"assignments":int(x.shape[0]*TOPK),"active_experts":int(len(nz)),"effective_experts":float(2**entropy(h)),"median_expert_batch":float(np.median(nz)),"p10_expert_batch":float(np.quantile(nz,.10)),"le1":float(np.mean(nz<=1)),"le2":float(np.mean(nz<=2)),"le4":float(np.mean(nz<=4)),"effective_tiles":int(q.sum()),"full_tiles":int(np.floor(h[h>0]/bm).sum()),"tail_experts":int(np.count_nonzero(h[h>0]%bm)),"padding_rows":int((q*bm).sum()-h.sum()),"padding_fraction":float(((q*bm).sum()-h.sum())/max(1,h.sum())),"fragmentation":float(q.sum()/max(1,len(nz))),"max_rank_load":int(rh.max()),"rank_max_mean":float(rh.max()/max(mean,1e-12)),"rank_cv":float(rh.std()/max(mean,1e-12)),"modality_mix":mix,"block_m":bm}

def visual_prefix_counts(e, mask):
    """Prefix expert counts, shape [layer, expert, token]."""
    n, l, k = e.shape
    out = np.zeros((l, EXPERTS, n + 1), dtype=np.int32)
    for pos in range(n):
        out[:, :, pos + 1] = out[:, :, pos]
        if mask[pos]:
            for layer in range(l):
                out[layer, np.asarray(e[pos, layer], dtype=int), pos + 1] += 1
    return out

def chunks_oracle(e,mask,b,prefix):
    """Exact bounded partition DP minimizing total vision tile count.

    The DP is intentionally route-aware and is used only for the upper-bound
    control.  It optimizes the complete partition, including the number of
    chunks, rather than making greedy local cuts.
    """
    n=len(mask); dp=np.full(n+1,np.inf); prev=np.full(n+1,-1,dtype=int); dp[0]=0.0
    for st in range(n):
        if not np.isfinite(dp[st]): continue
        lo,hi=valid_range(st,n,b)
        ends=range(lo,hi+1) if lo<=hi else ([n] if n-st<=int(math.floor(1.25*b)) else [])
        for en in ends:
            bm=block_m(en-st)
            h=prefix[:,:,en]-prefix[:,:,st]
            q=((h+bm-1)//bm).sum()
            value=dp[st]+float(q)
            if value < dp[en]-1e-9 or (abs(value-dp[en])<=1e-9 and (prev[en]<0 or abs(en-(st+b))<abs(en-(prev[en]+b)))):
                dp[en]=value; prev[en]=st
    if prev[n] < 0: return chunks_fixed(n,b)
    ends=[n]; cur=n
    while cur>0:
        cur=int(prev[cur]); ends.append(cur)
    return list(reversed(ends))

def get_bounds(meta, n, mask, kind):
    mod=[i for i in range(1,n) if bool(mask[i])!=bool(mask[i-1])]
    spatial=[]
    for im in meta.get("images",[]):
        st,en=im["token_span"]; gh,gw=im["post_merge_grid_hw"]
        spatial.extend([st+r*gw for r in range(1,gh) if st+r*gw<en])
        spatial.extend([st,en])
    spatial=sorted(set(x for x in spatial if 0<x<n))
    if kind=="modality": return sorted(set(mod+[0,n]))
    if kind=="spatial": return sorted(set(spatial+[0,n]))
    if kind=="both": return sorted(set(mod+spatial+[0,n]))
    return [0,n]

def get_shuffled_spatial_bounds(meta, n, seed):
    """Negative control: preserve each image grid's number of row cuts but
    randomly permute 2-D coordinates across the flattened visual positions.

    This keeps the image/token counts and approximate boundary cardinality while
    removing true spatial locality.  It is metadata-only and never changes the
    route or token order.
    """
    rng=np.random.default_rng(seed); bounds=[0,n]
    for im in meta.get("images",[]):
        st,en=map(int,im["token_span"]); gh,gw=map(int,im["post_merge_grid_hw"])
        length=max(0,min(n,en)-st)
        if length <= 1: continue
        coords=np.arange(gh*gw,dtype=np.int64)
        if len(coords) < length: coords=np.resize(coords,length)
        else: coords=coords[:length]
        perm=rng.permutation(coords)
        # In a row-major grid, a column of gw-1 is the row boundary.
        for q,coord in enumerate(perm[:-1]):
            if int(coord)%gw == gw-1:
                bounds.append(st+q+1)
        bounds.extend([st,en])
    return sorted(set(x for x in bounds if 0<=x<=n))

def load():
    man=json.loads((ROUTE_BASE/"workload_manifest.json").read_text()); sm=json.loads(SPATIAL_MANIFEST.read_text()); smap={x["sample_id"]:x for x in sm["samples"]}
    data=[]
    for pair in man["pairs"]:
        v=pair["vision"]; z=np.load(ROUTE_BASE/v["route_file"]); e=z["routed_experts"].astype(np.int16); ids=z["prompt_token_ids"].astype(np.int64); mask=ids==IMAGE_ID
        data.append({"id":v["request_id"],"category":v["category"],"pair_id":pair["pair_id"],"e":e,"ids":ids,"mask":mask,"meta":smap.get(v["request_id"],{})})
    return man,data

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-id",default="20260831_180000"); args=ap.parse_args()
    out=Path("poc_flashvep/deepep_revalidation/results")/f"spatial_chunked_prefill_{args.run_id}"; analysis=out/"analysis"; figs=out/"figures"; raw=out/"raw"; analysis.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True); raw.mkdir(parents=True,exist_ok=True)
    man,data=load(); rows=[]; summary=[]; random_rows=[]
    for d in data:
        n=len(d["ids"]); e=d["e"]; mask=d["mask"]
        for b in BUDGETS:
            prefix=visual_prefix_counts(e,mask)
            strategies={"fixed":chunks_fixed(n,b),"modality":chunks_boundary(n,b,get_bounds(d["meta"],n,mask,"modality")),"spatial":chunks_boundary(n,b,get_bounds(d["meta"],n,mask,"spatial")),"spatial_shuffled":chunks_boundary(n,b,get_shuffled_spatial_bounds(d["meta"],n,SEED+int(b)*100+d["pair_id"])),"modality_spatial":chunks_boundary(n,b,get_bounds(d["meta"],n,mask,"both")),"oracle":chunks_oracle(e,mask,b,prefix)}
            for strategy,ends in strategies.items():
                for ci,(st,en) in enumerate(zip(ends[:-1],ends[1:])):
                    for l in range(LAYERS):
                        for scope in ("vision","all"):
                            z=chunk_cost(e[:,l,:],d["ids"],mask,st,en,scope)
                            rows.append({"request_id":d["id"],"category":d["category"],"budget":b,"strategy":strategy,"chunk":ci,"start":st,"end":en,"chunk_tokens":en-st,"layer":l,"scope":scope,"boundary_distance":min(abs(st-x) for x in get_bounds(d["meta"],n,mask,"both")),**z})
                # Request-level total costs retain exact all-token work equality.
                for scope in ("vision","all"):
                    costs=[chunk_cost(e[:,l,:],d["ids"],mask,st,en,scope) for st,en in zip(ends[:-1],ends[1:]) for l in range(LAYERS)]
                    summary.append({"request_id":d["id"],"category":d["category"],"budget":b,"strategy":strategy,"scope":scope,"chunks":len(ends)-1,"total_tokens":n if scope=="all" else int(mask.sum()),"total_assignments":sum(x["assignments"] for x in costs),"tile_sum":sum(x["effective_tiles"] for x in costs),"padding_sum":sum(x["padding_rows"] for x in costs),"tail_expert_sum":sum(x["tail_experts"] for x in costs),"median_batch":float(np.median([x["median_expert_batch"] for x in costs])),"p10_batch":float(np.median([x["p10_expert_batch"] for x in costs])),"small_le4":float(np.mean([x["le4"] for x in costs])),"frag_mean":float(np.mean([x["fragmentation"] for x in costs])),"rank_cv_mean":float(np.mean([x["rank_cv"] for x in costs]))})
            # Random shifted controls: no metadata, same bounded size range.
            for seed in RANDOM_SEEDS:
                rng=np.random.default_rng(SEED+seed+int(b)*1000+d["pair_id"]); ends=[0]; st=0
                while st<n:
                    lo,hi=valid_range(st,n,b)
                    end=n if lo > hi else int(rng.integers(lo,hi+1))
                    ends.append(end); st=end
                costs=[chunk_cost(e[:,l,:],d["ids"],mask,st,en,"vision") for st,en in zip(ends[:-1],ends[1:]) for l in range(LAYERS)]
                random_rows.append({"request_id":d["id"],"category":d["category"],"budget":b,"seed":seed,"chunks":len(ends)-1,"tile_sum":sum(x["effective_tiles"] for x in costs),"padding_sum":sum(x["padding_rows"] for x in costs),"frag_mean":float(np.mean([x["fragmentation"] for x in costs])),"p10_batch":float(np.median([x["p10_expert_batch"] for x in costs]))})
    def write(df,path): df.to_csv(path,index=False)
    write(pd.DataFrame(rows),analysis/"chunk_layer_metrics.csv"); write(pd.DataFrame(summary),analysis/"chunk_summary.csv"); write(pd.DataFrame(random_rows),analysis/"random_shifted_control.csv")
    # Aggregate report metrics and figures.
    sm=pd.DataFrame(summary); rr=pd.DataFrame(random_rows); fixed=sm[(sm.strategy=="fixed")&(sm.scope=="vision")].set_index(["request_id","budget"])
    agg=[]
    for b in BUDGETS:
        f=fixed.xs(b,level="budget")
        for strat in ["fixed","modality","spatial","spatial_shuffled","modality_spatial","oracle"]:
            x=sm[(sm.budget==b)&(sm.scope=="vision")&(sm.strategy==strat)].set_index("request_id"); merged=f.join(x,lsuffix="_f",rsuffix="_x")
            agg.append({"budget":b,"strategy":strat,"tile_sum_median":float(merged.tile_sum_x.median()),"padding_sum_median":float(merged.padding_sum_x.median()),"tile_ratio_vs_fixed":float(np.median(merged.tile_sum_f/merged.tile_sum_x)),"padding_ratio_vs_fixed":float(np.median(merged.padding_sum_f/np.maximum(merged.padding_sum_x,1))),"p10_batch_median":float(merged.p10_batch_x.median()),"small_le4_median":float(merged.small_le4_x.median()),"chunks_median":float(merged.chunks_x.median())})
    ad=pd.DataFrame(agg); ad.to_csv(analysis/"strategy_aggregate.csv",index=False); ad.to_json(out/"summary.json",orient="records",indent=2)
    plt.figure(figsize=(9,4));
    for strat in ["fixed","modality","spatial","spatial_shuffled","modality_spatial","oracle"]:
        x=ad[ad.strategy==strat]; plt.plot(x.budget,x.tile_ratio_vs_fixed,marker="o",label=strat)
    plt.axhline(1,color="k",lw=.7); plt.xscale("log",base=2); plt.xticks(BUDGETS,BUDGETS); plt.ylabel("Vision tile-cost ratio vs fixed"); plt.xlabel("Chunk budget"); plt.legend(); plt.tight_layout(); plt.savefig(figs/"plot1_chunk_tile_headroom.png",dpi=160); plt.close()
    plt.figure(figsize=(9,4));
    for strat in ["fixed","modality","spatial","spatial_shuffled","modality_spatial","oracle"]:
        x=ad[ad.strategy==strat]; plt.plot(x.budget,x.padding_ratio_vs_fixed,marker="o",label=strat)
    plt.axhline(1,color="k",lw=.7); plt.xscale("log",base=2); plt.xticks(BUDGETS,BUDGETS); plt.ylabel("Vision padded-row ratio vs fixed"); plt.xlabel("Chunk budget"); plt.legend(); plt.tight_layout(); plt.savefig(figs/"plot2_padding_fragmentation.png",dpi=160); plt.close()
    plt.figure(figsize=(9,4));
    for strat in ["modality","spatial","spatial_shuffled","modality_spatial","oracle"]:
        x=ad[ad.strategy==strat]; plt.plot(x.budget,x.p10_batch_median,marker="o",label=strat)
    plt.xscale("log",base=2); plt.xticks(BUDGETS,BUDGETS); plt.ylabel("Median chunk p10 expert batch"); plt.xlabel("Chunk budget"); plt.legend(); plt.tight_layout(); plt.savefig(figs/"plot3_expert_batch_density.png",dpi=160); plt.close()
    plt.figure(figsize=(8,4));
    for strat in ["fixed","modality","spatial","spatial_shuffled","modality_spatial","oracle"]:
        x=ad[ad.strategy==strat]; plt.plot(x.budget,1-x.tile_ratio_vs_fixed,marker="o",label=strat)
    plt.axhline(0,color="k",lw=.7); plt.xscale("log",base=2); plt.xticks(BUDGETS,BUDGETS); plt.ylabel("Tile-cost reduction vs fixed"); plt.xlabel("Chunk budget"); plt.legend(); plt.tight_layout(); plt.savefig(figs/"plot4_oracle_headroom.png",dpi=160); plt.close()
    prov={"run_id":args.run_id,"seed":SEED,"budgets":BUDGETS,"random_seeds":RANDOM_SEEDS,"range_fraction":[.75,1.25],"requests":len(data),"layers":LAYERS,"experts":EXPERTS,"top_k":TOPK,"image_token_id":IMAGE_ID,"route_base":str(ROUTE_BASE),"spatial_manifest":str(SPATIAL_MANIFEST),"expert_to_rank":"expert_id//32","gpu_execution":False,"current_requested_gpu_mapping":[1,2,3,4]}
    (out/"provenance.json").write_text(json.dumps(prov,indent=2)); print(json.dumps({"out":str(out),"aggregate":agg},indent=2))

if __name__=="__main__": main()
