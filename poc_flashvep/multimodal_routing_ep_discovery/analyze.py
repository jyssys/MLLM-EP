#!/usr/bin/env python3
"""Offline discovery analysis for multimodal routing/EP traces.

This script deliberately consumes previously captured route artifacts.  It does
not run a model, alter routing, or make a scheduling decision.  Random controls
use a fixed seed and all pair/window sample caps are fixed before analysis.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 20260831
N_BOOT = 500
EP = 4
EXPERTS = 128
TOPK = 8
LAYERS = 48
IMAGE_TOKEN_ID = 151655
PAIR_CAP = 128
WINDOW_SIZES = (32, 64, 128)


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12)
    m = 0.5 * (a + b)
    def kl(x, y):
        z = x > 0
        return float(np.sum(x[z] * np.log2(x[z] / np.maximum(y[z], 1e-12))))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def entropy(p: np.ndarray, norm: float | None = None) -> float:
    p = np.asarray(p, dtype=float); total = float(p.sum()); p = p[p > 0]
    if not len(p): return 0.0
    p = p / max(total, 1e-12)
    v = float(-np.sum(p * np.log2(p)))
    return v / math.log2(norm) if norm else v


def set_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = set(map(int, a)); b = set(map(int, b))
    return len(a & b) / max(1, len(a | b))


def dest_hist(d: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(d, dtype=int).ravel(), minlength=EP).astype(float)


def expert_hist(e: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(e, dtype=int).ravel(), minlength=EXPERTS).astype(float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else 0.0


def bootstrap_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values): return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(N_BOOT)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def safe_write(rows, path: Path, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def fixed_sample(rng, xs, n):
    xs = list(xs)
    if len(xs) <= n: return xs
    return [xs[i] for i in rng.choice(len(xs), n, replace=False)]


def load_data(base: Path, spatial_manifest: Path):
    manifest = json.loads((base / "workload_manifest.json").read_text())
    sm = json.loads(spatial_manifest.read_text())
    spatial = {x["sample_id"]: x for x in sm["samples"]}
    data = []
    for pair in manifest["pairs"]:
        v = pair["vision"]; t = pair["text"]
        vz = np.load(base / v["route_file"]); tz = np.load(base / t["route_file"])
        ve = vz["routed_experts"].astype(np.int16); te = tz["routed_experts"].astype(np.int16)
        vi = vz["prompt_token_ids"].astype(np.int64); ti = tz["prompt_token_ids"].astype(np.int64)
        assert ve.ndim == 3 and ve.shape[1:] == (LAYERS, TOPK)
        assert np.all((ve >= 0) & (ve < EXPERTS)) and np.all((te >= 0) & (te < EXPERTS))
        data.append({"pair_id": pair["pair_id"], "id": v["request_id"], "category": v["category"],
                     "vision_e": ve, "vision_ids": vi, "text_e": te, "text_ids": ti,
                     "meta": spatial.get(v["request_id"], {})})
    return manifest, data


def token_features(data, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "per_token_layer.csv.gz"
    fields = ["request_id","pair_id","category","modality","token","layer","token_type","token_id","experts","dest_ranks","expert_entropy","dest_entropy","top1_expert","top1_rank"]
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        def emit(row):
            nonlocal count
            w.writerow(row); count += 1
        for d in data:
            for modality, e, ids in (("vision_request", d["vision_e"], d["vision_ids"]),
                                     ("text_control", d["text_e"], d["text_ids"])):
                dm = e // 32
                vis = ids == IMAGE_TOKEN_ID
                for tok in range(len(ids)):
                    for l in range(LAYERS):
                        eh = expert_hist(e[tok, l]); dh = dest_hist(dm[tok, l])
                        emit({"request_id": d["id"], "pair_id": d["pair_id"], "category": d["category"],
                              "modality": modality, "token": tok, "layer": l,
                              "token_type": "vision" if modality == "vision_request" and vis[tok] else "text",
                              "token_id": int(ids[tok]), "experts": ",".join(map(str, e[tok,l])),
                              "dest_ranks": ",".join(map(str, sorted(set(dm[tok,l])))),
                              "expert_entropy": entropy(eh, EXPERTS), "dest_entropy": entropy(dh, EP),
                              "top1_expert": int(e[tok,l,0]), "top1_rank": int(dm[tok,l,0])})
    return count


def boundary_analysis(data, out: Path):
    rng = np.random.default_rng(SEED); rows=[]; window=[]; ctrl=[]
    for d in data:
        e=d["vision_e"]; ids=d["vision_ids"]; dm=e//32
        for l in range(LAYERS):
            for i in range(len(ids)-1):
                a,b=ids[i]==IMAGE_TOKEN_ID,ids[i+1]==IMAGE_TOKEN_ID
                typ="VV" if a and b else "TT" if not a and not b else ("TV" if not a else "VT")
                rows.append({"request_id":d["id"],"layer":l,"position":i,"type":typ,
                             "expert_jaccard":set_jaccard(e[i,l],e[i+1,l]),
                             "expert_distance":1-set_jaccard(e[i,l],e[i+1,l]),
                             "dest_jaccard":set_jaccard(dm[i,l],dm[i+1,l]),
                             "dest_jsd":jsd(dest_hist(dm[i,l]),dest_hist(dm[i+1,l]))})
            bpos=[i for i in range(len(ids)-1) if (ids[i]==IMAGE_TOKEN_ID)!=(ids[i+1]==IMAGE_TOKEN_ID)]
            for bp in bpos:
                typ="TV" if ids[bp]!=IMAGE_TOKEN_ID else "VT"
                for off in range(-16,17):
                    i=bp+off
                    if 0<=i<len(ids)-1:
                        window.append({"request_id":d["id"],"layer":l,"boundary_type":typ,"offset":off,
                                       "expert_distance":1-set_jaccard(e[i,l],e[i+1,l]),
                                       "dest_jsd":jsd(dest_hist(dm[i,l]),dest_hist(dm[i+1,l]))})
        # fixed arbitrary text midpoint controls, with identical +/-16 window
        # Text-only control uses the paired text route (no image boundary), with
        # a fixed midpoint boundary and the same ±16-token window.
        tids=d["text_ids"]; te=d["text_e"]; tdm=te//32
        if len(tids)>=34:
            bp=len(tids)//2
            for l in range(LAYERS):
                for off in range(-16,17):
                    i=bp+off
                    if 0<=i<len(tids)-1:
                        ctrl.append({"request_id":d["id"],"layer":l,"boundary_type":"synthetic_text",
                                     "offset":off,"expert_distance":1-set_jaccard(te[i,l],te[i+1,l]),
                                     "dest_jsd":jsd(dest_hist(tdm[i,l]),dest_hist(tdm[i+1,l]))})
        # label-shuffle negative control: preserve exact count and transition values, shuffle labels.
        for l in range(LAYERS):
            labels=np.array(ids==IMAGE_TOKEN_ID); sh=rng.permutation(labels)
            for i in range(len(ids)-1):
                a,b=sh[i],sh[i+1]; typ="VV" if a and b else "TT" if not a and not b else "cross_shuffled"
                ctrl.append({"request_id":d["id"],"layer":l,"boundary_type":typ,"offset":0,
                             "expert_distance":1-set_jaccard(e[i,l],e[i+1,l]),
                             "dest_jsd":jsd(dest_hist(dm[i,l]),dest_hist(dm[i+1,l]))})
    safe_write(rows,out/"boundary_transitions.csv")
    safe_write(window,out/"boundary_window.csv")
    safe_write(ctrl,out/"boundary_negative_controls.csv")
    return pd.DataFrame(rows),pd.DataFrame(window),pd.DataFrame(ctrl)


def persistence(data,out:Path):
    rows=[]; rng=np.random.default_rng(SEED)
    for d in data:
        for mod,e,ids in (("vision",d["vision_e"],d["vision_ids"]),("text",d["text_e"],d["text_ids"])):
            mask=(ids==IMAGE_TOKEN_ID) if mod=="vision" else (ids!=IMAGE_TOKEN_ID)
            for l in range(LAYERS-1):
                idx=np.flatnonzero(mask); idx=np.array(fixed_sample(rng,idx,64),dtype=int)
                vals=[]
                for i in idx:
                    vals.append({"request_id":d["id"],"layer":l,"modality":mod,
                                 "expert_jaccard":set_jaccard(e[i,l],e[i,l+1]),
                                 "dest_jaccard":set_jaccard(e[i,l]//32,e[i,l+1]//32),
                                 "expert_top1_same":int(e[i,l,0]==e[i,l+1,0]),
                                 "dest_top1_same":int(e[i,l,0]//32==e[i,l+1,0]//32),
                                 "entropy_delta":abs(entropy(expert_hist(e[i,l]),EXPERTS)-entropy(expert_hist(e[i,l+1]),EXPERTS))})
                if vals:
                    x=pd.DataFrame(vals); r={"request_id":d["id"],"layer":l,"modality":mod}
                    for c in x.columns[3:]: r[c]=float(x[c].mean())
                    rows.append(r)
    safe_write(rows,out/"cross_layer_persistence.csv")
    return pd.DataFrame(rows)


def spatial_analysis(data,out:Path):
    rng=np.random.default_rng(SEED); rows=[]; heat=None
    for d in data:
        e=d["vision_e"]; dm=e//32; meta=d["meta"]
        for im in meta.get("images",[]):
            st,en=im["token_span"]; n=en-st; gh,gw=im["post_merge_grid_hw"]
            if n != gh*gw or st<0 or en>len(e): continue
            coords=[(q//gw,q%gw) for q in range(n)]
            groups={"adjacent":[],"moderate":[],"far":[],"random":[]}
            adj=[]; mod=[]; far=[]
            for q,(r,c) in enumerate(coords):
                for q2 in (q+1,q+gw):
                    if q2<n:
                        dist=abs(r-coords[q2][0])+abs(c-coords[q2][1]);
                        (adj if dist==1 else mod if 2<=dist<=4 else far).append((q,q2))
            groups["adjacent"]=fixed_sample(rng,adj,PAIR_CAP); groups["moderate"]=fixed_sample(rng,mod,PAIR_CAP)
            groups["far"]=fixed_sample(rng,far,PAIR_CAP)
            allpairs=[(a,b) for a in range(n) for b in range(a+1,n)] if n<=700 else None
            if allpairs is not None: groups["random"]=fixed_sample(rng,allpairs,PAIR_CAP)
            else: groups["random"]=[(int(rng.integers(0,n)),int(rng.integers(0,n))) for _ in range(PAIR_CAP)]
            for l in range(LAYERS):
                for rel,pairs in groups.items():
                    for a,b in pairs:
                        if a==b: continue
                        ea,eb=e[st+a,l],e[st+b,l]; da,db=dm[st+a,l],dm[st+b,l]
                        rows.append({"request_id":d["id"],"image_index":im["image_index"],"layer":l,"relation":rel,
                                     "expert_jaccard":set_jaccard(ea,eb),"dest_jaccard":set_jaccard(da,db),
                                     "expert_jsd":jsd(expert_hist(ea),expert_hist(eb)),"dest_jsd":jsd(dest_hist(da),dest_hist(db))})
            if heat is None or n>heat["n"]: heat={"n":n,"d":d,"im":im}
    safe_write(rows,out/"spatial_pairs.csv")
    return pd.DataFrame(rows),heat


def bursts(data,out:Path):
    rows=[]
    for d in data:
        for mod,e,ids in (("vision",d["vision_e"],d["vision_ids"]),("text",d["text_e"],d["text_ids"])):
            spans=[]
            if mod=="vision":
                spans=[(x["token_span"][0],x["token_span"][1]) for x in d["meta"].get("images",[])]
            else: spans=[(0,len(ids))]
            for l in range(LAYERS):
                for st,en in spans:
                    for w in WINDOW_SIZES:
                        for k in range(st,en-w+1,w):
                            h=dest_hist(e[k:k+w,l]); p=h/h.sum(); rows.append({"request_id":d["id"],"layer":l,"modality":mod,"start":k,"window":w,
                                "max_dest_fraction":float(p.max()),"dest_hhi":float(np.sum(p*p)),"dest_entropy":entropy(h,EP),"active_ranks":int(np.count_nonzero(h)),"active_experts":int(np.unique(e[k:k+w,l]).size)})
    safe_write(rows,out/"traffic_bursts.csv")
    return pd.DataFrame(rows)


def working_sets(data,out:Path):
    rng=np.random.default_rng(SEED); rows=[]
    for d in data:
        counts={"vision":np.sum(d["vision_ids"]==IMAGE_TOKEN_ID),"text":np.sum(d["vision_ids"]!=IMAGE_TOKEN_ID)}
        n=min(counts.values())
        for mod,e,ids in (("vision",d["vision_e"],d["vision_ids"]),("text",d["text_e"],d["text_ids"])):
            idx=np.flatnonzero(ids==IMAGE_TOKEN_ID) if mod=="vision" else np.flatnonzero(ids!=IMAGE_TOKEN_ID)
            idx=np.array(fixed_sample(rng,idx,int(n)),dtype=int)
            for l in range(LAYERS):
                h=expert_hist(e[idx,l]); p=h/h.sum(); s=np.sort(p)[::-1]
                rows.append({"request_id":d["id"],"pair_id":d["pair_id"],"layer":l,"modality":mod,"sample_tokens":int(n),
                             "unique_experts":int(np.count_nonzero(h)),"effective_experts":float(2**entropy(h)),"expert_entropy":entropy(h,EXPERTS),
                             "top4_fraction":float(s[:4].sum()),"top8_fraction":float(s[:8].sum()),"ep_coverage":int(np.count_nonzero(dest_hist(e[idx,l])))})
    safe_write(rows,out/"working_sets.csv")
    return pd.DataFrame(rows)


def cross_image(data,out:Path):
    rng=np.random.default_rng(SEED); rows=[]; nvis=min(64,min(np.sum(x["vision_ids"]==IMAGE_TOKEN_ID) for x in data)); ntxt=min(64,min(np.sum(x["text_ids"]!=IMAGE_TOKEN_ID) for x in data))
    for l in range(LAYERS):
        for mod,n in (("vision",int(nvis)),("text",int(ntxt))):
            vec=[]
            for d in data:
                e=d["vision_e"] if mod=="vision" else d["text_e"]; ids=d["vision_ids"] if mod=="vision" else d["text_ids"]
                idx=np.flatnonzero(ids==IMAGE_TOKEN_ID) if mod=="vision" else np.flatnonzero(ids!=IMAGE_TOKEN_ID); idx=np.array(fixed_sample(rng,idx,n))
                vec.append(expert_hist(e[idx,l]) / max(1, n*TOPK))
            for (a,va),(b,vb) in combinations(enumerate(vec),2):
                rows.append({"layer":l,"modality":mod,"request_a":data[a]["id"],"request_b":data[b]["id"],"cosine":cosine(va,vb),"jsd":jsd(va,vb)})
    safe_write(rows,out/"cross_image_consistency.csv")
    return pd.DataFrame(rows)


def directionality(boundary:pd.DataFrame, data,out:Path):
    rows=[]
    for d in data:
        e=d["vision_e"]; ids=d["vision_ids"]; dm=e//32
        for l in range(LAYERS):
            for i in range(len(ids)-1):
                if (ids[i]==IMAGE_TOKEN_ID)==(ids[i+1]==IMAGE_TOKEN_ID): continue
                before=dest_hist(dm[i,l]); after=dest_hist(dm[i+1,l]); delta=(after-after.mean())-(before-before.mean())
                rows.append({"request_id":d["id"],"layer":l,"direction":"TV" if ids[i]!=IMAGE_TOKEN_ID else "VT",
                             "dR0":delta[0],"dR1":delta[1],"dR2":delta[2],"dR3":delta[3],"delta_l2":float(np.linalg.norm(delta))})
    safe_write(rows,out/"transition_directionality.csv")
    return pd.DataFrame(rows)


def metrics_and_figures(bd,bw,bc,pers,spat,burst,work,cross,direc,data,figdir:Path):
    figdir.mkdir(parents=True,exist_ok=True)
    summary={}
    # A: transition distances
    if len(bd):
        a=bd.groupby("type")["expert_distance"].agg(["median","mean","count"]).to_dict("index"); summary["boundary"]=a
        plt.figure(figsize=(7,4)); bd.boxplot(column="expert_distance",by="type"); plt.suptitle(""); plt.title("Adjacent-token routing transition distance"); plt.ylabel("1 - expert-set Jaccard"); plt.tight_layout(); plt.savefig(figdir/"plot1_modality_boundary_transition.png",dpi=160); plt.close()
    # B persistence
    if len(pers):
        x=pers.groupby(["layer","modality"]).expert_jaccard.mean().reset_index();
        plt.figure(figsize=(8,4));
        for m,g in x.groupby("modality"): plt.plot(g.layer,g.expert_jaccard,label=m)
        plt.xlabel("Layer pair l→l+1"); plt.ylabel("Expert-set Jaccard"); plt.title("Cross-layer routing persistence"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"plot2_cross_layer_persistence.png",dpi=160); plt.close()
        summary["persistence"]=pers.groupby("modality")["expert_jaccard"].mean().to_dict()
    # C spatial locality
    if len(spat):
        x=spat.groupby("relation")["dest_jsd"].agg(["median","mean","count"]); summary["spatial"]=x.to_dict("index")
        plt.figure(figsize=(8,4)); spat.boxplot(column="dest_jsd",by="relation"); plt.suptitle(""); plt.title("Spatial relation vs destination-rank routing JSD"); plt.ylabel("EP-rank JSD"); plt.tight_layout(); plt.savefig(figdir/"plot3_spatial_routing_locality.png",dpi=160); plt.close()
        # heatmap for representative largest image at layer 24
    if len(burst):
        x=burst.groupby(["modality","window"])["max_dest_fraction"].median().reset_index();
        plt.figure(figsize=(8,4));
        for (m,w),g in x.groupby(["modality","window"]): plt.scatter([f"{m}-{w}"],[g.max_dest_fraction.iloc[0]])
        plt.xticks(rotation=45,ha="right"); plt.ylabel("Median max destination fraction"); plt.title("Local EP traffic bursts"); plt.tight_layout(); plt.savefig(figdir/"plot4_spatial_region_ep_burst.png",dpi=160); plt.close(); summary["bursts"]=burst.groupby("modality")["max_dest_fraction"].median().to_dict()
    if len(work):
        x=work.groupby("modality")["unique_experts","effective_experts"] if False else work.groupby("modality")["unique_experts"].mean(); summary["working_sets"]=x.to_dict()
        plt.figure(figsize=(8,4)); work.boxplot(column="unique_experts",by="modality"); plt.suptitle(""); plt.title("Equal-token expert working set"); plt.tight_layout(); plt.savefig(figdir/"plot5_working_set.png",dpi=160); plt.close()
    if len(cross):
        summary["cross_image"]=cross.groupby("modality")["cosine","jsd"] if False else cross.groupby("modality")["cosine"].mean().to_dict()
        x=cross.groupby(["layer","modality"]).cosine.mean().reset_index(); plt.figure(figsize=(8,4));
        for m,g in x.groupby("modality"): plt.plot(g.layer,g.cosine,label=m)
        plt.ylabel("Cross-request visual expert-hist cosine"); plt.xlabel("Layer"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"plot6_cross_image_consistency.png",dpi=160); plt.close()
    if len(direc):
        summary["directionality"]=direc.groupby("direction")["delta_l2"].agg(["mean","median","count"]).to_dict("index")
        means=direc.groupby("direction")[["dR0","dR1","dR2","dR3"]].mean(); means.T.plot(kind="bar",figsize=(7,4)); plt.ylabel("Mean centered rank migration"); plt.title("Boundary transition directionality"); plt.tight_layout(); plt.savefig(figdir/"plot7_transition_directionality.png",dpi=160); plt.close()
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base",type=Path,default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609")); ap.add_argument("--spatial-manifest",type=Path,default=Path("poc_flashvep/deepep_revalidation/results/tile_slack_mechanism_20260820_150852/stage_a/sample_manifest.json")); ap.add_argument("--run-id",default="20260831_173000"); args=ap.parse_args()
    out=Path("poc_flashvep/deepep_revalidation/results")/f"multimodal_routing_ep_discovery_{args.run_id}"; raw=out/"raw"; analysis=out/"analysis"; figs=out/"figures"; out.mkdir(parents=True,exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    manifest,data=load_data(args.base,args.spatial_manifest)
    prov={"run_id":args.run_id,"seed":SEED,"bootstrap_reps":N_BOOT,"route_artifact":str(args.base),"spatial_manifest":str(args.spatial_manifest),"n_requests":len(data),"layers":LAYERS,"experts":EXPERTS,"top_k":TOPK,"ep":EP,"expert_to_rank":"expert_id//32","image_token_id":IMAGE_TOKEN_ID,"gpu_execution":False,"historical_capture_gpu_mapping":manifest.get("configuration",{}).get("physical_gpus"),"current_requested_gpu_mapping":[1,2,3,4],"source_route_sha256":{}}
    for d in data:
        for typ in ("vision","text"):
            p=args.base/manifest["pairs"][d["pair_id"]][typ]["route_file"]; prov["source_route_sha256"][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/"provenance.json").write_text(json.dumps(prov,indent=2))
    # Per-token file is intentionally a compressed CSV; all exact top-k IDs are retained.
    token_features(data,raw)
    bd,bw,bc=boundary_analysis(data,analysis); pers=persistence(data,analysis); spat,heat=spatial_analysis(data,analysis); burst=bursts(data,analysis); work=working_sets(data,analysis); cross=cross_image(data,analysis); direc=directionality(bd,data,analysis)
    if heat:
        d=heat["d"]; im=heat["im"]; e=d["vision_e"]; dm=e//32; st,en=im["token_span"]; gh,gw=im["post_merge_grid_hw"]; arr=np.argmax(np.stack([np.bincount(dm[st+q,24],minlength=EP) for q in range(en-st)]),axis=1).reshape(gh,gw); plt.figure(figsize=(7,5)); plt.imshow(arr,cmap="tab10",vmin=0,vmax=EP-1); plt.colorbar(label="dominant EP rank"); plt.title(f"Representative spatial EP map: {d['id']} layer 24"); plt.tight_layout(); plt.savefig(figs/"plot8_representative_ep_heatmap.png",dpi=160); plt.close()
    summary=metrics_and_figures(bd,bw,bc,pers,spat,burst,work,cross,direc,data,figs); (out/"summary.json").write_text(json.dumps(summary,indent=2,default=float))
    print(json.dumps({"out":str(out),"summary":summary},indent=2,default=float))


if __name__=="__main__": main()
