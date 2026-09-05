"""Analyze online FusedMoE observations and produce the geometry gate.

The script is intentionally conservative: it reports insufficient data rather
than turning duplicated worker rows into a spurious positive result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("invocations.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                # Preserve old debugging captures on disk, but exclude the
                # pre-fix layer=-1 rows from the causal online dataset.
                if int(row.get("layer", -1)) < 0:
                    continue
                row["trace_source"] = str(p.parent.relative_to(root))
                rows.append(row)
            except json.JSONDecodeError: pass
    return rows


def flat_rows(rows: list[dict]) -> list[dict]:
    out=[]
    for r in rows:
        # vLLM's memory-profile dummy forward uses a synthetic M=4096 route.
        # It is useful as a separately labeled runtime-state control but is
        # excluded from the natural online fanout/pair analysis.
        if int(r.get("M", 0) or 0) > 2048:
            continue
        x=dict(r)
        for key in ("M", "active_experts", "total_assignments", "expert_cv", "expert_hhi",
                    "expert_entropy", "fanout_mean", "fanout_p10", "fanout_median",
                    "fanout_p90", "fanout_f4", "rank_max_mean", "expert_max_mean", "wall_ms", "cuda_ms"):
            x[key]=float(x.get(key, 0.0) or 0.0)
        eh=np.asarray(x.get("expert_hist", []), dtype=float)
        rh=np.asarray(x.get("rank_loads", []), dtype=float)
        nz=eh[eh>0]
        x["expert_max_load"]=float(nz.max()) if nz.size else 0.0
        x["expert_p90_load"]=float(np.quantile(nz,.9)) if nz.size else 0.0
        x["rank_cv"]=float(rh.std()/rh.mean()) if rh.size and rh.mean() else 0.0
        mat=np.asarray(x.get("sender_dest_matrix", []), dtype=float)
        if mat.size and mat.sum():
            p=mat.reshape(-1); p=p[p>0]/p.sum()
            x["traffic_entropy"]=float(-(p*np.log(p+1e-12)).sum())
            x["traffic_concentration"]=float((p*p).sum())
        else: x["traffic_entropy"]=0.0; x["traffic_concentration"]=0.0
        # Kernel tile size is backend-selected and is not guessed here.  This
        # transparent work proxy is the exact routed assignment count.
        x["padded_work_proxy"]=x["total_assignments"]
        out.append(x)
    return out


def fit(rows: list[dict], features: list[str], target="cuda_ms") -> dict:
    valid=[r for r in rows if all(math.isfinite(float(r.get(f,0))) for f in features+[target])]
    if len(valid)<12: return {"status":"INSUFFICIENT", "n":len(valid), "features":features}
    valid.sort(key=lambda r: float(r.get("timestamp_ns",0)))
    cut=max(1,min(len(valid)-1,int(len(valid)*.7)))
    tr,te=valid[:cut],valid[cut:]
    X=np.asarray([[1.0]+[float(r.get(f,0)) for f in features] for r in tr])
    y=np.asarray([float(r.get(target,0)) for r in tr])
    Xt=np.asarray([[1.0]+[float(r.get(f,0)) for f in features] for r in te])
    yt=np.asarray([float(r.get(target,0)) for r in te])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]; pred=Xt@beta
    rmse=float(np.sqrt(np.mean((pred-yt)**2))); mae=float(np.mean(np.abs(pred-yt)))
    base=float(np.mean((yt-yt.mean())**2)); r2=1.0-float(np.sum((pred-yt)**2))/float(np.sum((yt-yt.mean())**2)+1e-12)
    return {"status":"OK", "n":len(valid), "train_n":len(tr), "test_n":len(te), "r2":r2,
            "rmse":rmse, "mae":mae, "p90_abs_error":float(np.quantile(np.abs(pred-yt),.9)),
            "beta":beta.tolist(), "features":features, "target":target}


def phase_metrics(rows: list[dict]) -> dict:
    """Fit the hierarchy separately for prefill and decode."""
    dist=["M","active_experts","expert_max_load","expert_p90_load","expert_cv",
          "expert_hhi","expert_entropy","padded_work_proxy"]
    rank=dist+["rank_max_mean","rank_cv"]
    geom=rank+["fanout_mean","fanout_p10","fanout_median","fanout_p90","fanout_f4",
               "traffic_entropy","traffic_concentration"]
    result={}
    for phase in sorted({str(r.get("phase", "unknown")) for r in rows}):
        subset=[r for r in rows if str(r.get("phase"))==phase]
        result[phase]={"n_rows":len(subset)}
        if subset:
            result[phase]["target_quantiles_ms"]=[float(np.quantile(
                [r["cuda_ms"] for r in subset], q)) for q in (.5,.9,.99)]
        for label, fs in (("model0", ["M"]), ("model1_distribution", dist),
                          ("model2_distribution_plus_rank", rank),
                          ("model3_plus_fanout_geometry", geom)):
            result[phase][label]=fit(subset, fs)
        m2=result[phase]["model2_distribution_plus_rank"]
        m3=result[phase]["model3_plus_fanout_geometry"]
        result[phase]["model2_to_model3_rmse_reduction_pct"]=(
            100*(m2["rmse"]-m3["rmse"])/(m2["rmse"]+1e-12)
            if m2.get("status")==m3.get("status")=="OK" else None)
        # Sensitivity view with the upper 1% (mostly first-use/idle outliers)
        # capped.  The uncapped fit above remains the primary result.
        if len(subset) >= 20:
            cutoff=float(np.quantile([r["cuda_ms"] for r in subset], .99))
            trimmed=[r for r in subset if float(r["cuda_ms"])<=cutoff]
            a,b=fit(trimmed, rank), fit(trimmed, geom)
            result[phase]["trim_p99_cutoff_ms"]=cutoff
            result[phase]["trim_p99_model2_to_model3_rmse_reduction_pct"]=(
                100*(a["rmse"]-b["rmse"])/(a["rmse"]+1e-12)
                if a.get("status")==b.get("status")=="OK" else None)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); args=ap.parse_args()
    rows=flat_rows(read_rows(args.root)); out=args.root/"models"; out.mkdir(exist_ok=True)
    if rows:
        fields=sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float,str))})
        with (args.root/"online_invocations.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    dist=["M","active_experts","expert_max_load","expert_p90_load","expert_cv","expert_hhi","expert_entropy","padded_work_proxy"]
    rank=dist+["rank_max_mean","rank_cv"]
    geom=rank+["fanout_mean","fanout_p10","fanout_median","fanout_p90","fanout_f4","traffic_entropy","traffic_concentration"]
    metrics={"n_rows":len(rows), "model0":fit(rows,["M"]), "model1_distribution":fit(rows,dist),
             "model2_distribution_plus_rank":fit(rows,rank), "model3_plus_fanout_geometry":fit(rows,geom)}
    metrics["phase_metrics"] = phase_metrics(rows)
    m2=metrics["model2_distribution_plus_rank"]; m3=metrics["model3_plus_fanout_geometry"]
    if m2.get("status")==m3.get("status")=="OK":
        metrics["model2_to_model3_rmse_reduction_pct"]=100*(m2["rmse"]-m3["rmse"])/(m2["rmse"]+1e-12)
    else: metrics["model2_to_model3_rmse_reduction_pct"]=None
    (out/"model_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    if rows:
        fan=np.asarray([r["fanout_mean"] for r in rows]);
        summary={"mean":float(fan.mean()),"p10":float(np.quantile(fan,.1)),"p50":float(np.quantile(fan,.5)),"p90":float(np.quantile(fan,.9)),"min":float(fan.min()),"max":float(fan.max()),"f4_range":[float(min(r.get("fanout_f4",0) for r in rows)),float(max(r.get("fanout_f4",0) for r in rows))]}
    else: summary={"status":"NO_ROWS"}
    (args.root/"natural_fanout_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    # Nearest natural pairs: same layer/phase and similar basic load, then
    # rank by fanout separation.  No fabricated pair is emitted.
    pairs=[]
    # Worker/TP rows are intentionally retained for model fitting, but pair
    # search operates on a bounded deterministic sample to avoid quadratic
    # work on a long decode trace.
    pair_rows = rows if len(rows) <= 3000 else [rows[i] for i in np.linspace(0, len(rows)-1, 3000, dtype=int)]
    for i,a in enumerate(pair_rows):
        best=None
        for j,b in enumerate(pair_rows):
            if j<=i or a.get("layer")!=b.get("layer") or a.get("phase")!=b.get("phase"): continue
            if abs(a["M"]-b["M"])/max(1,a["M"],b["M"])>.05: continue
            if abs(a["active_experts"]-b["active_experts"])>8: continue
            if abs(a["rank_max_mean"]-b["rank_max_mean"])/max(1,a["rank_max_mean"],b["rank_max_mean"])>.05: continue
            score=abs(a["fanout_mean"]-b["fanout_mean"])
            if best is None or score>best[0]: best=(score,j,b)
        if best:
            b=best[2]
            pairs.append({"a":i,"b":best[1],"fanout_delta":best[0],
                          "t_moe_delta_pct":100*(b.get("cuda_ms",0)-a.get("cuda_ms",0))/max(1e-9,a.get("cuda_ms",0))})
    (args.root/"matched_pairs"/"pairs.json").parent.mkdir(exist_ok=True)
    (args.root/"matched_pairs"/"pairs.json").write_text(json.dumps(pairs[:100],indent=2),encoding="utf-8")
    print(json.dumps({"rows":len(rows),"fanout":summary,"model2_to_model3_pct":metrics["model2_to_model3_rmse_reduction_pct"],"pairs":len(pairs)},indent=2))

if __name__=="__main__": main()
