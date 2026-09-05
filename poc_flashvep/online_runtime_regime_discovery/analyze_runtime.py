"""Conservative analysis for the online runtime-regime discovery sprint.

The online driver records one stock FusedMoE interval per EP rank.  This
script deliberately treats dispatch/expert/combine as unavailable when they
were not instrumented separately, and reports the full interval as T_MoE
instead of manufacturing a decomposition.
"""
from __future__ import annotations

import argparse, csv, gzip, json, math, re, time
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_rows(paths, stride=1):
    rows=[]
    for root in paths:
        root=Path(root)
        files = list(root.rglob("invocations.jsonl")) + list(root.rglob("invocations.jsonl.gz"))
        for p in sorted(files):
            source=p.parent.name
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt", encoding="utf-8", errors="ignore") as fh:
              for lineno,line in enumerate(fh):
                if stride > 1 and lineno % stride:
                    continue
                try: r=json.loads(line)
                except Exception: continue
                if int(r.get("layer",-1))<0 or float(r.get("M",0))>2048: continue
                r=dict(r); r["source"] = source; r["trace_path"] = str(p)
                ctx=str(r.get("request_context", ""))
                m=re.search(r"sms(\d+)",ctx)
                r["sms"] = int(m.group(1)) if m else (20 if "sms20" in ctx else None)
                r["backend"] = "deepep_low_latency" if "low_latency" in ctx else "deepep_high_throughput"
                for k in ("M","cuda_ms","wall_ms","active_experts","total_assignments","rank_max_mean","expert_max_mean","expert_cv","expert_hhi","expert_entropy","fanout_mean","fanout_f4"):
                    try: r[k]=float(r.get(k,0) or 0)
                    except Exception: r[k]=0.0
                eh=np.asarray(r.get("expert_hist",[]),float); nz=eh[eh>0]
                r["expert_max_load"]=float(nz.max()) if nz.size else 0.0
                r["expert_p90_load"]=float(np.quantile(nz,.9)) if nz.size else 0.0
                rh=np.asarray(r.get("rank_loads",[]),float)
                r["rank_cv"]=float(rh.std()/rh.mean()) if rh.size and rh.mean() else 0.0
                mat=np.asarray(r.get("sender_dest_matrix",[]),float)
                if mat.size and mat.sum():
                    q=mat.ravel(); q=q[q>0]/q.sum(); r["traffic_entropy"]=float(-(q*np.log(q+1e-12)).sum()); r["traffic_concentration"]=float((q*q).sum())
                else: r["traffic_entropy"]=0.0; r["traffic_concentration"]=0.0
                rows.append(r)
    return rows


def cv_summary(rows):
    vals=[]
    groups={}
    for r in rows:
        key=(r["source"],r["phase"],int(r["layer"]),int(r["M"]),int(r["ep_rank"]))
        groups.setdefault(key,[]).append(float(r["cuda_ms"]))
    for k,x in groups.items():
        if len(x)>=3 and np.mean(x)>0: vals.append({"key":k,"n":len(x),"median_ms":float(np.median(x)),"cv":float(np.std(x)/np.mean(x))})
    c=np.asarray([x["cv"] for x in vals]) if vals else np.array([])
    return {"groups":len(vals),"median_cv":float(np.median(c)) if c.size else None,"p90_cv":float(np.quantile(c,.9)) if c.size else None,"pass":"PASS" if c.size and np.median(c)<=.10 else ("CAUTION" if c.size and np.median(c)<=.20 else "INSUFFICIENT"),"details":vals}


def ols(rows, features):
    good=[r for r in rows if all(np.isfinite(float(r.get(f,0))) for f in features+["cuda_ms"])]
    if len(good)<20: return {"status":"INSUFFICIENT","n":len(good)}
    good.sort(key=lambda r: float(r.get("timestamp_ns",0)))
    n_total=len(good)
    # Fit on a deterministic time-ordered cap.  The raw online trace is
    # retained; using millions of rows in four dense least-squares matrices
    # adds hours without changing the residual comparison.  The cap preserves
    # the pre-registered time-block split and is reported explicitly.
    fit_cap=400_000
    if len(good)>fit_cap:
        idx=np.linspace(0,len(good)-1,fit_cap,dtype=np.int64)
        good=[good[int(i)] for i in idx]
    cut=max(1,min(len(good)-1,int(.7*len(good))))
    tr,te=good[:cut],good[cut:]
    def mat(a): return np.asarray([[1.0]+[float(r.get(f,0)) for f in features] for r in a])
    X,y=mat(tr),np.asarray([r["cuda_ms"] for r in tr]); Xt,yt=mat(te),np.asarray([r["cuda_ms"] for r in te])
    try: b=np.linalg.lstsq(X,y,rcond=None)[0]; pred=Xt@b
    except Exception: return {"status":"FAIL","n":len(good)}
    err=np.abs(pred-yt); rmse=float(np.sqrt(np.mean(err*err))); base=float(np.mean((yt-yt.mean())**2));
    return {"status":"OK","n":len(good),"n_total":n_total,"sampled":n_total>len(good),"train_n":len(tr),"test_n":len(te),"rmse":rmse,"mae":float(np.mean(err)),"p90_abs_error":float(np.quantile(err,.9)),"r2":1-float(np.sum((pred-yt)**2))/(float(np.sum((yt-yt.mean())**2))+1e-12),"features":features}


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not fields: fields=sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float,str))})
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trace",action="append",required=True); ap.add_argument("--out",required=True); ap.add_argument("--stride",type=int,default=1,help="deterministic row stride for large traces"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); rows=load_rows(args.trace,max(1,args.stride))
    write_csv(out/"online_invocations.csv",rows)
    sanity=cv_summary(rows)
    dist=["M","active_experts","expert_max_load","expert_p90_load","expert_cv","expert_hhi","expert_entropy"]
    rank=dist+["rank_max_mean","rank_cv"]
    geom=rank+["fanout_mean","fanout_f4","traffic_entropy","traffic_concentration"]
    models={"model0_M":ols(rows,["M"]),"model1_distribution":ols(rows,dist),"model2_distribution_plus_rank":ols(rows,rank),"model3_plus_fanout_geometry":ols(rows,geom)}
    if models["model2_distribution_plus_rank"].get("status")==models["model3_plus_fanout_geometry"].get("status")=="OK":
        a=models["model2_distribution_plus_rank"]["rmse"]; b=models["model3_plus_fanout_geometry"]["rmse"]; models["model2_to_model3_rmse_reduction_pct"]=100*(a-b)/(a+1e-12)
    else: models["model2_to_model3_rmse_reduction_pct"]=None
    # Configuration summaries are descriptive; traces use varied natural waves,
    # so they are not treated as paired causal route replays.  Build the
    # summaries in one pass: rescanning the multi-million-row online trace for
    # every M is prohibitively expensive and can look like a hung analysis.
    cfg=[]
    cfg_groups={}
    for r in rows:
        sms=r.get("sms")
        if sms is not None: cfg_groups.setdefault((sms,r["phase"]),[]).append(r)
    for (sms,phase),z in sorted(cfg_groups.items()):
        vals=np.asarray([r["cuda_ms"] for r in z],dtype=float)
        cfg.append({"sms":sms,"phase":phase,"n":len(z),"M_median":float(np.median([r["M"] for r in z])),"cuda_p50_ms":float(np.median(vals)),"cuda_p90_ms":float(np.quantile(vals,.9)),"cuda_mean_ms":float(np.mean(vals))})
    write_csv(out/"communication_sms_summary.csv",cfg)
    # Per regime lower envelope over safely measured SMS values.
    env=[]
    env_groups={}
    for r in rows:
      sms=r.get("sms")
      if sms is not None: env_groups.setdefault((r["phase"],int(r["M"]),sms),[]).append(float(r["cuda_ms"]))
    by_m={}
    for (phase,M,sms),vals in env_groups.items(): by_m.setdefault((phase,M),{})[sms]=float(np.median(np.asarray(vals)))
    for (phase,M),by in sorted(by_m.items()):
      if by:
        best=min(by,key=by.get); static=by.get(20,min(by.values())); env.append({"phase":phase,"M":M,"best_sms":best,"best_ms":by[best],"static_sms20_ms":static,"static_to_oracle_pct":100*(static-by[best])/(static+1e-12),"config_medians":json.dumps(by,sort_keys=True)})
    write_csv(out/"oracle_envelope.csv",env)
    # Tail mining within same phase/M/layer; rank-local records are retained.
    tails=[]; grouped={}
    for r in rows: grouped.setdefault((r["phase"],int(r["M"]),int(r["layer"])),[]).append(r)
    for (phase,M,layer),z in grouped.items():
        if len(z)<10: continue
        t=np.asarray([r["cuda_ms"] for r in z]); q=float(np.quantile(t,.95)); med=float(np.quantile(t,.5)); slow=[r for r in z if r["cuda_ms"]>=q]; fast=[r for r in z if r["cuda_ms"]<=med]
        sm=float(np.median([r["cuda_ms"] for r in slow])); fm=float(np.median([r["cuda_ms"] for r in fast]))
        tails.append({"phase":phase,"M":M,"layer":layer,"n":len(z),"p50_ms":float(np.median(t)),"p95_ms":q,"tail_gap_pct":100*(sm-fm)/(fm+1e-12),"slow_fanout":float(np.mean([r["fanout_mean"] for r in slow])),"fast_fanout":float(np.mean([r["fanout_mean"] for r in fast])),"slow_rank_max_mean":float(np.mean([r["rank_max_mean"] for r in slow])),"fast_rank_max_mean":float(np.mean([r["rank_max_mean"] for r in fast]))})
    write_csv(out/"tail_mining.csv",tails)
    # Small plots for the report (only if matplotlib is usable).
    if plt and rows:
        plt.figure(figsize=(7,4));
        for sms in sorted({r.get("sms") for r in rows if r.get("sms") is not None}):
            z=[r for r in rows if r.get("sms")==sms and r["phase"]=="prefill"]
            if z: plt.scatter([r["M"] for r in z],[r["cuda_ms"] for r in z],s=5,label=f"SMS {sms}")
        plt.xlabel("M"); plt.ylabel("stock FusedMoE CUDA interval (ms)"); plt.legend(); plt.tight_layout(); plt.savefig(out/"sms_vs_tmoe.png",dpi=150); plt.close()
        plt.figure(figsize=(7,4));
        z=[r for r in rows if r["phase"]=="prefill"]
        if z: plt.scatter([r["fanout_mean"] for r in z],[r["cuda_ms"] for r in z],s=5,c=[r["M"] for r in z]); plt.xlabel("mean fanout"); plt.ylabel("T_MoE CUDA ms"); plt.tight_layout(); plt.savefig(out/"fanout_vs_tmoe.png",dpi=150); plt.close()
    summary={"created_unix":time.time(),"rows":len(rows),"sources":sorted({r["source"] for r in rows}),"phases":{p:sum(r["phase"]==p for r in rows) for p in sorted({r["phase"] for r in rows})},"timing_sanity":sanity,"models":models,"sms_summary":cfg,"oracle_rows":len(env),"tail_rows":len(tails),"limitations":["stock FusedMoE hook records one full apply interval; dispatch/expert/combine are not separately CUDA-event instrumented in this online path","SMS traces use different engine restarts and natural waves, so lower envelope is descriptive, not a matched-route causal estimate"]}
    (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"rows":len(rows),"sanity":sanity,"model2_to_model3_pct":models["model2_to_model3_rmse_reduction_pct"],"oracle_rows":len(env),"tails":len(tails)},indent=2))

if __name__=="__main__": main()
