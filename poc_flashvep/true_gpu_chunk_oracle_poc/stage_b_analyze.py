"""Fit the strict same-count GPU-cost DP and summarize validation replay."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGIES=("fixed","balanced","tile_same_count","true_gpu")

def strict_dp(task:dict, costs:dict[str,float])->list[int]:
    n=int(task["n"]); b=int(task["budget"]); k=len(task["fixed_cuts"])-1; pts=[int(x) for x in task["candidate_boundaries"]]
    dp=np.full((k+1,n+1),np.inf); prev=np.full((k+1,n+1),-1,dtype=int); dp[0,0]=0.0
    for part in range(1,k+1):
        for end in pts:
            if end < part or end > n-(k-part): continue
            for start in pts:
                if start>=end or end-start>b: continue
                key=f"{start}:{end}"
                if dp[part-1,start] < np.inf and key in costs:
                    value=float(dp[part-1,start]+costs[key])
                    if value < dp[part,end]-1e-12: dp[part,end]=value; prev[part,end]=start
    if not np.isfinite(dp[k,n]): return [int(x) for x in task["fixed_cuts"]]
    cuts=[n]; end=n
    for part in range(k,0,-1): end=int(prev[part,end]); cuts.append(end)
    return list(reversed(cuts))

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--result",required=True); args=ap.parse_args(); root=Path(args.result)
    tasks=json.loads((root/"stage_b_intervals.json").read_text())["tasks"]
    rows=[]
    measured_costs={}
    for path in sorted((root/"stage_b_cost" ).glob("rank*.json")):
        payload=json.loads(path.read_text());
        for x in payload["rows"]:
            interval = tuple(int(v) for v in x["interval"])
            rows.append({"task_id":x["task_id"],"request_id":x["request_id"],"source":x["source"],"layer":x["layer"],"budget":x["budget"],"rank":x["rank"],"interval":interval,"expert_ms":x["expert_stats"]["median_ms"],"wall_ms":x["wall_stats"]["median_ms"],"dispatch_ms":x["dispatch_stats"]["median_ms"],"combine_ms":x["combine_stats"]["median_ms"],"expert_cv":x["expert_stats"]["cv"]})
            measured_costs.setdefault((x["task_id"], interval), []).append(float(x["expert_stats"]["median_ms"]))
    raw=pd.DataFrame(rows); raw.to_csv(root/"stage_b_interval_costs.csv",index=False)
    costs={(task, f"{interval[0]}:{interval[1]}"): float(max(values)) for (task, interval), values in measured_costs.items()}
    cut_rows=[]
    for task in tasks:
        c={k[1]:v for k,v in costs.items() if k[0]==task["task_id"]}; true=strict_dp(task,c)
        cut_rows.append({"task_id":task["task_id"],"request_id":task["request_id"],"source":task["source"],"layer":task["layer"],"budget":task["budget"],"n":task["n"],"fixed_cuts":task["fixed_cuts"],"balanced_cuts":task["balanced_cuts"],"tile_same_count_cuts":task["tile_same_count_cuts"],"true_gpu_cuts":true,"true_gpu_dp_fallback":true==task["fixed_cuts"]})
    (root/"stage_b_cuts.json").write_text(json.dumps({"status":"ok","tasks":cut_rows,"objective":"rank-max measured expert CUDA median interval cost","constraints":"contiguous; strict <=B; exact Fixed K"},separators=(",",":"))+"\n")
    # If a candidate never had a feasible DP path, keep the explicit fallback
    # visible rather than silently calling it an oracle.
    pd.DataFrame(cut_rows).to_csv(root/"stage_b_selected_cuts.csv",index=False)
    print("prepared",len(cut_rows),"selected cuts")

if __name__=="__main__": main()
