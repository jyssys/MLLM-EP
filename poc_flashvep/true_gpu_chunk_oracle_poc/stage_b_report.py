"""Aggregate selected-cut validation and apply the Stage B gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGIES=("fixed","balanced","tile_same_count","true_gpu")
def main()->None:
 ap=argparse.ArgumentParser(); ap.add_argument("--result",required=True); args=ap.parse_args(); root=Path(args.result); rows=[]
 for p in sorted((root/"stage_b_validate").glob("rank*.json")):
  payload=json.loads(p.read_text())
  for x in payload["rows"]:
   rows.append({"task_id":x["task_id"],"request_id":x["request_id"],"source":x["source"],"layer":x["layer"],"budget":x["budget"],"rank":x["rank"],"strategy":x["strategy"],"chunks":x["chunks"],"min_chunk":min(x["chunk_sizes"]),"max_chunk":max(x["chunk_sizes"]),"size_cv":float(np.std(x["chunk_sizes"])/max(np.mean(x["chunk_sizes"]),1e-12)),"expert_ms":x["expert_stats"]["median_ms"],"wall_ms":x["wall_stats"]["median_ms"],"dispatch_ms":x["dispatch_stats"]["median_ms"],"combine_ms":x["combine_stats"]["median_ms"],"correctness":x["correctness"],"route_identity":x["route_identity"],"token_partition_identity":x["token_partition_identity"]})
 raw=pd.DataFrame(rows); raw.to_csv(root/"stage_b_validation_per_rank.csv",index=False)
 agg=raw.groupby(["task_id","request_id","source","layer","budget","strategy"],as_index=False).agg({"expert_ms":"max","wall_ms":"max","dispatch_ms":"max","combine_ms":"max","chunks":"first","min_chunk":"first","max_chunk":"first","size_cv":"first","correctness":"all","route_identity":"all","token_partition_identity":"all"})
 agg.to_csv(root/"stage_b_validation.csv",index=False)
 summary=[]
 for (source,b),g in agg.groupby(["source","budget"]):
  p=g.pivot(index="task_id",columns="strategy",values="expert_ms")
  for s in STRATEGIES: 
   if s not in p:p[s]=np.nan
   vals=1-p[s]/p["fixed"]
   summary.append({"source":source,"budget":b,"strategy":s,"tasks":int(vals.notna().sum()),"expert_ms_median":float(g[g.strategy==s].expert_ms.median()),"expert_reduction_vs_fixed_median":float(vals.median()),"expert_positive_fraction":float((vals>0).mean()),"chunks_median":float(g[g.strategy==s].chunks.median()),"size_cv_median":float(g[g.strategy==s].size_cv.median()),"correctness_all":bool(g[g.strategy==s].correctness.all())})
 sm=pd.DataFrame(summary); sm.to_csv(root/"stage_b_summary.csv",index=False)
 figdir=root/"figures"; figdir.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(10,4.8)); x=np.arange(4); width=.18
 for j,s in enumerate(STRATEGIES[1:]): ax.bar(x+(j-1)*width,[100*sm[(sm.source=="short")&(sm.budget==b)&(sm.strategy==s)].expert_reduction_vs_fixed_median.iloc[0] if len(sm[(sm.source=="short")&(sm.budget==b)&(sm.strategy==s)]) else np.nan for b in (128,256,512,1024)],width,label=s)
 ax.axhline(0,color="k",lw=.7); ax.set_xticks(x,["128","256","512","1024"]); ax.set(xlabel="Budget",ylabel="Max-rank expert reduction vs Fixed (%)",title="Stage B: true GPU-cost oracle validation"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"plot3_true_gpu_oracle_comparison.png",dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5)); ax.hist(raw["expert_ms"],bins=30,alpha=.8); ax.set(xlabel="Interval expert CUDA median (ms)",ylabel="Count",title="Measured interval-cost distribution"); fig.tight_layout(); fig.savefig(figdir/"plot4_interval_cost_distribution.png",dpi=180); plt.close(fig)
 true=[]
 for b in (128,256):
  g=agg[(agg.source=="short")&(agg.budget==b)].pivot(index="task_id",columns="strategy",values="expert_ms"); true.append(float((1-g["true_gpu"]/g["balanced"]).median()))
 gate="STRONG_GO" if all(x>=.10 for x in true) else "GO" if any(x>=.10 for x in true) and all(x>0 for x in true) else "HOLD" if any(x>=.05 for x in true) else "NO-GO"
 (root/"stage_b_gate.json").write_text(json.dumps({"status":gate,"balanced_to_true_gpu_reduction_short_128_256":true,"validation_correctness":bool(agg.correctness.all()),"route_identity":bool(agg.route_identity.all() and agg.token_partition_identity.all())},indent=2)+"\n")
 print(sm.to_string(index=False)); print(json.dumps({"stage_b_status":gate,"balanced_to_true":true}))
if __name__=="__main__":main()
