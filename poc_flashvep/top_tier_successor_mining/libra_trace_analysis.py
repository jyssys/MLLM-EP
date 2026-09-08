"""Lookahead-to-placement diagnostic on freshly captured genuine Qwen routes.

Prescribed four-source partition is an OFFLINE CONTROL, not a live EP execution.
Count makespan is NEVER converted linearly to latency or E2E benefit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from libra_paper_plan import plan,evaluate_plan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--layers",default=",".join(str(x) for x in range(1,48)))
    ap.add_argument("--phase2-balance",action="store_true",
                    help="Sensitivity only: run sharding inside replica selection")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    routes=args.input/"routes"
    request_ids=sorted(p.name.removesuffix("_prefill_l00.pt") for p in routes.glob("*_prefill_l00.pt"))
    summaries=[]
    home=np.eye(4,dtype=bool)[np.arange(128)//32]
    rng=np.random.default_rng(7317)
    for offset in range(0,len(request_ids)-3,4):
        group=request_ids[offset:offset+4]
        for layer in [int(x) for x in args.layers.split(",")]:
            loaded=[]
            for rid in group:
                prev=torch.load(routes/f"{rid}_prefill_l{layer-1:02d}.pt",map_location="cpu",weights_only=True)
                curr=torch.load(routes/f"{rid}_prefill_l{layer:02d}.pt",map_location="cpu",weights_only=True)
                loaded.append((prev["next_prediction"].numpy(),curr))
            for modality in ["all","vision","text"]:
                for budget in [None,16,32]:
                    actual=np.zeros((4,128),dtype=np.int64)
                    predicted=np.zeros_like(actual)
                    recalls=[]
                    eligible=True
                    for source,(pred,curr) in enumerate(loaded):
                        ids=curr["ids"].numpy()
                        mask=np.ones(len(ids),dtype=bool) if modality=="all" else curr[modality].numpy()
                        chosen=np.flatnonzero(mask)
                        if budget is not None:
                            if len(chosen)<budget:
                                eligible=False;break
                            chosen=rng.choice(chosen,budget,replace=False)
                        ids,pred=ids[chosen],pred[chosen]
                        actual[source]=np.bincount(ids.reshape(-1),minlength=128)
                        predicted[source]=np.bincount(pred.reshape(-1),minlength=128)
                        recalls.extend((ids[:,:,None]==pred[:,None,:]).any(-1).mean(-1).tolist())
                    if not eligible:
                        continue
                    pp=plan(predicted,phase2_balance=args.phase2_balance)
                    op=plan(actual,phase2_balance=args.phase2_balance)
                    base=evaluate_plan(actual,home)
                    got=evaluate_plan(actual,pp["placement"])
                    oracle=evaluate_plan(actual,op["placement"])
                    extra_pred=pp["placement"] & ~home
                    extra_oracle=op["placement"] & ~home
                    row={"requests":";".join(group),"layer":layer,"modality":modality,
                         "matched_tokens_per_source":budget,"total_assignments":int(actual.sum()),
                         "topk_recall":float(np.mean(recalls)),
                         "predicted_replica_count":int(extra_pred.sum()),
                         "oracle_replica_count":int(extra_oracle.sum()),
                         "replica_set_precision":float((extra_pred & extra_oracle).sum()/max(extra_pred.sum(),1)),
                         "base_max_assignments":int(base["loads"].max()),
                         "pred_max_assignments":int(got["loads"].max()),
                         "oracle_max_assignments":int(oracle["loads"].max()),
                         "base_local_fraction":float(base["local"].sum()/actual.sum()),
                         "pred_local_fraction":float(got["local"].sum()/actual.sum()),
                         "oracle_local_fraction":float(oracle["local"].sum()/actual.sum()),
                         "phase2_balance":args.phase2_balance,
                         "latency_headroom":"NOT_IDENTIFIED_BY_COUNTS"}
                    summaries.append(row)
    df=pd.DataFrame(summaries)
    df.to_csv(args.out/"prediction_placement.csv",index=False)
    agg=df.groupby(["modality","matched_tokens_per_source"],dropna=False).agg(
        pairs=("topk_recall","size"),recall=("topk_recall","mean"),
        replica_precision=("replica_set_precision","mean"),
        predicted_local=("pred_local_fraction","mean"),oracle_local=("oracle_local_fraction","mean"))
    agg.to_csv(args.out/"summary.csv")
    print(agg.to_string())


if __name__=="__main__":main()
