"""Official-style GQA exact / ChartQA relaxed scoring with image-cluster CIs."""
import argparse
import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--official-source",type=Path,required=True)
    args=ap.parse_args()
    tree=ast.parse(args.official_source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="relaxed_correctness")
    env={"Optional":Optional}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(args.official_source),"exec"),env)
    records=[json.loads(s) for f in args.input.glob("predictions_*.jsonl") for s in f.read_text().splitlines()]
    for r in records:
        pred=r["prediction"].strip()
        if r["dataset"]=="chartqa":
            r["correct"]=max(float(env["relaxed_correctness"](a,pred)) for a in r["answers"])
        else:
            r["correct"]=max(float(str(a).strip().lower()==pred.lower()) for a in r["answers"])
    df=pd.DataFrame(records)
    assert not df.duplicated(["request_id","policy"]).any()
    args.output.mkdir(parents=True,exist_ok=True)
    df.drop(columns=["policy_definition"]).to_csv(args.output/"scored_predictions.csv",index=False)
    summary=[]
    for dataset,sub in df.groupby("dataset"):
        pivot=sub.pivot(index="request_id",columns="policy",values="correct")
        image_ids=sub.drop_duplicates("request_id").set_index("request_id")["image_id"]
        for policy in pivot.columns:
            pairs=pivot[["vanilla",policy]].dropna() if policy!="vanilla" else pivot[[policy]].dropna()
            vals=(pairs[policy]-pairs["vanilla"]).to_numpy()
            clusters=image_ids.loc[pairs.index].to_numpy()
            unique=np.unique(clusters)
            grouped=[vals[clusters==g] for g in unique]
            rng=np.random.default_rng(7317)
            boot=[np.concatenate([grouped[i] for i in rng.integers(0,len(unique),len(unique))]).mean()*100
                  for _ in range(3000)]
            r={"dataset":dataset,"policy":policy,"n":len(pairs),"image_clusters":len(unique),
               "accuracy_percent":100*float(pairs[policy].mean()),
               "paired_delta_pp":float(vals.mean()*100),"paired_delta_95ci_pp":np.quantile(boot,[.025,.975]).tolist(),
               "vanilla_accuracy_on_same_requests":100*float(pairs["vanilla"].mean())}
            summary.append(r)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
