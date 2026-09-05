"""Analyze paired sender/destination geometry replay."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

def med(o, k): return float(o[k]["median_ms"])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True); a=ap.parse_args()
    rows=[]
    for p in sorted((a.result/"replay").glob("rank*_layer*.json")):
        d=json.loads(p.read_text());
        if d.get("status")!="ok": continue
        rank=int(d["rank"])
        for o in d.get("observations",[]):
            m=re.search(r"_M(\d+)_F2_A(\d+)_(concentrated|cyclic)(?:_R(\d+))?$",o["case_id"])
            if not m: continue
            M,A,shape,rep=m.groups(); rep=int(rep) if rep else -1
            rows.append({"case_id":o["case_id"],"M":int(M),"active":int(A),"geometry":shape,"rep":rep,"rank":rank,
                         "wall_ms":med(o,"wall_stats"),"layout_ms":med(o,"layout_stats"),"dispatch_ms":med(o,"dispatch_stats"),
                         "expert_ms":med(o,"expert_stats"),"combine_ms":med(o,"combine_stats"),"correctness":bool(o["correctness"]["passed"])})
    raw=pd.DataFrame(rows); raw.to_csv(a.result/"geometry_rank_timing_raw.csv",index=False)
    agg=raw.groupby(["case_id","M","active","geometry","rep"],as_index=False).agg(critical_wall_ms=("wall_ms","max"),layout_ms=("layout_ms","max"),dispatch_ms=("dispatch_ms","max"),expert_ms=("expert_ms","max"),combine_ms=("combine_ms","max"),correctness=("correctness","all"))
    agg.to_csv(a.result/"geometry_metrics.csv",index=False)
    cond=agg.groupby(["M","active","geometry"],as_index=False).agg(**{c:(c,"median") for c in ["critical_wall_ms","layout_ms","dispatch_ms","expert_ms","combine_ms"]},repetitions=("rep","nunique"),correctness=("correctness","all"))
    cond.to_csv(a.result/"geometry_condition_medians.csv",index=False)
    out=[]
    for M,g in cond.groupby("M"):
        c=g[g.geometry=="concentrated"].iloc[0]; y=g[g.geometry=="cyclic"].iloc[0]
        out.append({"M":int(M),"cyclic_vs_concentrated_expert_pct":(y.expert_ms/c.expert_ms-1)*100,"cyclic_vs_concentrated_dispatch_pct":(y.dispatch_ms/c.dispatch_ms-1)*100,"cyclic_vs_concentrated_combine_pct":(y.combine_ms/c.combine_ms-1)*100,"cyclic_vs_concentrated_wall_pct":(y.critical_wall_ms/c.critical_wall_ms-1)*100})
    pd.DataFrame(out).to_csv(a.result/"geometry_effects.csv",index=False)
    print(json.dumps({"status":"ok","raw_rows":len(raw),"condition_rows":len(cond),"correctness_all":bool(raw.correctness.all())},indent=2))
if __name__=="__main__": main()
