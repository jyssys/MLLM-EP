"""Analyze Stage A same-M sensitivity and apply fixed preregistered gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--result",required=True); args=ap.parse_args(); root=Path(args.result); rows=[]
    for p in sorted((root/"replay").glob("rank*.json")):
        payload=json.loads(p.read_text());
        for o in payload.get("observations",[]):
            r={k:o[k] for k in ("pair_id","request_id","category","source","layer","M","candidate","start","end")}; r["rank"]=payload["rank"]
            for st in ("wall","expert","dispatch","combine"): r[f"{st}_ms"]=o[f"{st}_stats"]["median_ms"]
            for k,v in o["features"].items(): r[f"shape_{k}"]=v
            r["correctness"]=bool(o["correctness"]["passed"]); rows.append(r)
    raw=pd.DataFrame(rows); raw.to_csv(root/"stage_a_per_rank.csv",index=False)
    if raw.empty: raise RuntimeError("no replay observations")
    g=raw.groupby(["pair_id","request_id","source","category","layer","M","candidate"],as_index=False)
    agg=g.agg({"wall_ms":"max","expert_ms":"max","dispatch_ms":"max","combine_ms":"max","correctness":"all", **{c:"first" for c in raw.columns if c.startswith("shape_")}})
    pivot=agg.pivot_table(index=["pair_id","request_id","source","category","layer","M"],columns="candidate",values=["wall_ms","expert_ms","dispatch_ms","combine_ms"])
    out=[]
    for idx,row in pivot.iterrows():
        d={k:v for k,v in zip(pivot.index.names,idx)}
        for st in ("wall","expert","dispatch","combine"):
            a=float(row[(f"{st}_ms","a")]); b=float(row[(f"{st}_ms","b")]); d[f"{st}_a_ms"]=a; d[f"{st}_b_ms"]=b; d[f"{st}_abs_gap_pct"]=100*abs(a-b)/max(min(a,b),1e-12)
        fa=agg[(agg.pair_id==d["pair_id"])&(agg.candidate=="a")].iloc[0]; fb=agg[(agg.pair_id==d["pair_id"])&(agg.candidate=="b")].iloc[0]
        d["shape_score"]=float(np.abs(np.asarray(json.loads("[]"))) .sum()) if False else np.nan
        for metric in ("active_experts","entropy","hhi","max_expert_load","p10_active_load","median_active_load","tiny_le_1","tiny_le_2","tiny_le_4","rank_imbalance","rank_cv"): d[f"delta_{metric}"]=float(fb[f"shape_{metric}"]-fa[f"shape_{metric}"])
        out.append(d)
    pairs=pd.DataFrame(out); pairs.to_csv(root/"stage_a_pair_results.csv",index=False)
    summary=[]
    for m,gm in pairs.groupby("M"):
        summary.append({"M":int(m),"pairs":len(gm),"expert_gap_median_pct":float(gm.expert_abs_gap_pct.median()),"expert_gap_p25_pct":float(gm.expert_abs_gap_pct.quantile(.25)),"expert_gap_p75_pct":float(gm.expert_abs_gap_pct.quantile(.75)),"wall_gap_median_pct":float(gm.wall_abs_gap_pct.median()),"dispatch_gap_median_pct":float(gm.dispatch_abs_gap_pct.median()),"combine_gap_median_pct":float(gm.combine_abs_gap_pct.median()),"expert_gap_ge_5_fraction":float((gm.expert_abs_gap_pct>=5).mean()),"expert_gap_ge_10_fraction":float((gm.expert_abs_gap_pct>=10).mean())})
    sm=pd.DataFrame(summary); sm.to_csv(root/"stage_a_summary.csv",index=False)
    figdir=root/"figures"; figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4.5)); data=[pairs.loc[pairs.M==m,"expert_abs_gap_pct"] for m in sorted(pairs.M.unique())]; ax.boxplot(data,tick_labels=[str(m) for m in sorted(pairs.M.unique())],showfliers=False); ax.axhline(5,color="tab:orange",ls="--",label="5% gate"); ax.axhline(10,color="tab:red",ls=":",label="10% strong"); ax.set(xlabel="Same window token count M",ylabel="Absolute max-rank expert latency gap (%)",title="Stage A: same-M routing-shape sensitivity"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"plot1_same_m_shape_vs_latency.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4));
    for ax,(metric,label) in zip(axes,(("delta_active_experts","Δ active experts"),("delta_hhi","Δ HHI"),("delta_max_expert_load","Δ max expert load"))): ax.scatter(pairs[metric],pairs.expert_abs_gap_pct,c=pairs.M,cmap="viridis",alpha=.75); ax.set(xlabel=label,ylabel="Expert latency gap (%)"); ax.grid(alpha=.2)
    fig.suptitle("Routing-shape deltas versus same-M GPU gap"); fig.tight_layout(); fig.savefig(figdir/"plot2_same_m_routing_features.png",dpi=180); plt.close(fig)
    med=float(pairs.expert_abs_gap_pct.median()); frac=float((pairs.expert_abs_gap_pct>=5).mean());
    # User preregistered gate: a median 5--10% same-M sensitivity is
    # PROMISING and warrants the bounded Stage B follow-up; consistency and
    # tail fractions are reported separately rather than used to hide it.
    status="STRONG" if med>=10 else "PROMISING" if med>=5 else "WEAK/NO-GO"
    (root/"stage_a_gate.json").write_text(json.dumps({"status":status,"primary_metric":"absolute paired max-rank expert latency gap","median_gap_pct":med,"fraction_ge_5_pct":100*frac,"stage_b_run":status in {"STRONG","PROMISING"},"pairs":len(pairs)},indent=2)+"\n")
    print(sm.to_string(index=False)); print(json.dumps({"stage_a_status":status,"median_expert_gap_pct":med,"pairs":len(pairs)}))
if __name__=="__main__": main()
