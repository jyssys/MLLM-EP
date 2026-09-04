"""Analyze the randomized persistent-worker H5 target-B replay."""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True); a = ap.parse_args()
    rows=[]
    for p in sorted((a.result / "replay").glob("rank*_layer24.json")):
        payload=json.loads(p.read_text())
        for o in payload.get("observations",[]):
            role=o.get("history_role","")
            cid=o["case_id"]
            if cid.startswith("B_steady_"): cond="steady"
            elif cid.startswith("B_after_A_"): cond="alternating"
            elif cid.startswith("B_after_S_"): cond="similar"
            elif cid.startswith("B_after_D_"): cond="disjoint"
            else: continue
            rows.append({"rank":payload["rank"],"case_id":o["case_id"],"condition":cond,
                         "wall_ms":float(o["wall_stats"]["median_ms"]),"dispatch_ms":float(o["dispatch_stats"]["median_ms"]),
                         "expert_ms":float(o["expert_stats"]["median_ms"]),"combine_ms":float(o["combine_stats"]["median_ms"]),
                         "iterations":o["iterations"]})
    raw=pd.DataFrame(rows)
    if raw.empty: raise RuntimeError("no randomized target observations")
    raw.to_csv(a.result/"randomized_target_rank_timings.csv",index=False)
    # Critical path is max across EP ranks for each target case.
    case=raw.groupby(["case_id","condition"],as_index=False).agg(critical_wall_ms=("wall_ms","max"),dispatch_ms=("dispatch_ms","max"),expert_ms=("expert_ms","max"),combine_ms=("combine_ms","max"),ranks=("rank","nunique"))
    case.to_csv(a.result/"randomized_target_case_timings.csv",index=False)
    stats=case.groupby("condition",as_index=False).agg(n=("case_id","count"),median_ms=("critical_wall_ms","median"),mean_ms=("critical_wall_ms","mean"),p90_ms=("critical_wall_ms",lambda x:float(x.quantile(.9))),cv_pct=("critical_wall_ms",lambda x:float(x.std(ddof=0)/x.mean()*100)),dispatch_median_ms=("dispatch_ms","median"),expert_median_ms=("expert_ms","median"),combine_median_ms=("combine_ms","median"))
    stats.to_csv(a.result/"randomized_condition_summary.csv",index=False)
    baseline=float(stats.loc[stats.condition=="steady","median_ms"].iloc[0])
    effects={r.condition:float((r.median_ms-baseline)/baseline*100) for r in stats.itertuples()}
    # Order-position diagnostic: regress target duration on position and
    # condition.  A position effect invalidates a history interpretation.
    case["position"]=case.case_id.str.extract(r"_(\d+)$").astype(float)
    pos_corr=float(case[["position","critical_wall_ms"]].corr().iloc[0,1]) if case.position.notna().all() else float("nan")
    plt.figure(figsize=(7,4)); present=[c for c in ["steady","alternating","similar","disjoint"] if c in case.condition.unique()]
    plt.boxplot([case.loc[case.condition==c,"critical_wall_ms"] for c in present],tick_labels=present); plt.ylabel("critical target-B wall (ms)"); plt.title("H5 randomized persistent worker"); plt.tight_layout(); plt.savefig(a.result/"randomized_history_boxplot.png",dpi=140); plt.close()
    summary={"target_cases":int(len(case)),"target_cases_per_condition":{c:int((case.condition==c).sum()) for c in present},"baseline_steady_median_ms":baseline,"condition_medians_ms":{k:float(v) for k,v in stats.set_index("condition").median_ms.to_dict().items()},"effects_vs_steady_pct":effects,"position_duration_pearson":pos_corr,"max_condition_cv_pct":float(stats.cv_pct.max()),"h5_gate":"HOLD" if max(abs(v) for k,v in effects.items() if k!="steady")>=5 and stats.cv_pct.max()>20 else "NO_GO","interpretation":"randomized target count is adequate, but condition variance/position-state must be inspected before causal warmth claim"}
    (a.result/"randomized_h5_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
