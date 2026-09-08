"""Fresh Qwen route transfer using the supplied actual Libra Cython predictor consumer.

Four equal source volumes are matched by sampling, not padded into the Cython
array. This is a route/planner analysis, not measured EP speed or an E2E oracle.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from libra_supplement import SupplementPlanner


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extension",type=Path,required=True)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--layers",default=",".join(map(str,range(1,48))))
    ap.add_argument("--groups",type=int,default=24)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    planner=SupplementPlanner(args.extension)
    root=args.input/"routes"
    requests=sorted(p.name.removesuffix("_prefill_l00.pt") for p in root.glob("*_prefill_l00.pt"))
    rows=[]
    for group_idx in range(min(len(requests)//4,args.groups)):
        group=requests[group_idx*4:group_idx*4+4]
        for layer in map(int,args.layers.split(",")):
            curr=[torch.load(root/f"{r}_prefill_l{layer:02d}.pt",map_location="cpu",weights_only=True) for r in group]
            prev=[torch.load(root/f"{r}_prefill_l{layer-1:02d}.pt",map_location="cpu",weights_only=True) for r in group]
            for modality in ["all","vision","text"]:
                eligible=[np.flatnonzero(np.ones(len(c["ids"]),dtype=bool) if modality=="all" else c[modality].numpy()) for c in curr]
                budgets=sorted(set([min(map(len,eligible)),16,64,256]))
                for m in budgets:
                    if m<=0 or any(len(a)<m for a in eligible):continue
                    rng=np.random.default_rng(7317+group_idx*100+layer)
                    chosen=[rng.choice(a,m,replace=False) for a in eligible]
                    ids=np.stack([c["ids"][s].numpy() for c,s in zip(curr,chosen)])
                    pred=np.stack([p["next_prediction"][s].numpy() for p,s in zip(prev,chosen)])
                    recall=(ids[...,None]==pred[...,None,:]).any(-1).mean()
                    pp=planner.prefetch(pred);op=planner.prefetch(ids)
                    pr=planner.rebalance(ids,pp);orr=planner.rebalance(ids,op)
                    home=np.eye(4,dtype=bool)[np.arange(128)//32]
                    pe=pp["placement"] & ~home;oe=op["placement"] & ~home
                    pred_local=pr["local_mask"].sum()/ids.size
                    oracle_local=orr["local_mask"].sum()/ids.size
                    row={"requests":";".join(group),"layer":layer,"modality":modality,
                         "tokens_per_source":m,"total_tokens":m*4,"topk_recall":recall,
                         "pred_local_fraction":pred_local,"oracle_local_fraction":oracle_local,
                         "local_fraction_delta":oracle_local-pred_local,
                         "pred_rank_max_mean":pr["loads"].max()/pr["loads"].mean(),
                         "oracle_rank_max_mean":orr["loads"].max()/orr["loads"].mean(),
                         "replica_precision":(pe & oe).sum()/max(pe.sum(),1),
                         "pred_replicas":int(pe.sum()),"oracle_replicas":int(oe.sum()),
                         "prefetch_cpu_ms":pp["prefetch_cpu_ms"],
                         "rebalance_cpu_ms_max":max(pr["rebalance_cpu_ms_per_rank"]),
                         "selection":"MATCHED_EQUAL_SOURCE_REAL_TOKEN_SUBSAMPLE",
                         "scope":"ACTUAL_SUPPLEMENT_PLANNER_NOT_LATENCY_ORACLE"}
                    rows.append(row)
        print(json.dumps({"group":group_idx,"rows":len(rows)}),flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(args.out/"prediction_plan_results.csv",index=False)
    summary=df.groupby("modality").agg(pairs=("topk_recall","size"),
        recall=("topk_recall","median"),pred_local=("pred_local_fraction","median"),
        oracle_local=("oracle_local_fraction","median"),local_delta=("local_fraction_delta","median"),
        pred_max_mean=("pred_rank_max_mean","median"),oracle_max_mean=("oracle_rank_max_mean","median"))
    summary.to_csv(args.out/"summary.csv")
    print(summary.to_string())


if __name__=="__main__":main()
