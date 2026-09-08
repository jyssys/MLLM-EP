"""Separate answer-content and answer-termination diagnostics; not a new scorer.

The headline score remains official relaxed ChartQA / exact GQA. First-line and
prefix acceptance are deliberately labeled optimistic formatting controls.
"""
import argparse
import ast
import json
import re
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
    node=next(n for n in ast.parse(args.official_source.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=="relaxed_correctness")
    env={"Optional":Optional}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(args.official_source),"exec"),env)
    relaxed=env["relaxed_correctness"]
    def score(row,pred):
        if row["dataset"]=="chartqa":
            return max(float(relaxed(a,pred.strip())) for a in row["answers"])
        return float(any(str(a).strip().lower()==pred.strip().lower() for a in row["answers"]))
    records=[json.loads(s) for f in args.input.glob("predictions_*.jsonl") for s in f.read_text().splitlines()]
    output=[]
    for row in records:
        pred=row["prediction"].strip()
        first=pred.splitlines()[0].strip() if pred else ""
        # Optimistic answer-prefix classification is a diagnostic upper bound,
        # NOT an implementable evaluator: it uses the labeled answer.
        prefix=any(re.match(re.escape(str(a).strip())+r"(?=$|\s|[,;])",pred,re.I)
                   for a in row["answers"])
        tokens=row["output_tokens"]
        output.append({"request_id":row["request_id"],"policy":row["policy"],
                       "image_id":row["image_id"],"correct":score(row,pred),
                       "first_line_correct":score(row,first),"labeled_answer_prefix":bool(prefix),
                       "output_tokens":len(tokens),"terminated":bool(tokens and tokens[-1] in (151645,151643)),
                       "prediction":pred,"answer":" | ".join(map(str,row["answers"]))})
    df=pd.DataFrame(output)
    assert not df.duplicated(["request_id","policy"]).any()
    baseline=df[df.policy=="vanilla"].set_index("request_id")
    summary=[]
    for name,g in df.groupby("policy"):
        b=baseline.loc[g.request_id]
        harmful=(b.correct.to_numpy()==1)&(g.correct.to_numpy()==0)
        s={"policy":name,"requests":len(g),"official_accuracy_percent":g.correct.mean()*100,
           "optimistic_first_line_accuracy_percent":g.first_line_correct.mean()*100,
           "termination_rate_percent":g.terminated.mean()*100,
           "mean_output_tokens":g.output_tokens.mean(),"p90_output_tokens":g.output_tokens.quantile(.9),
           "mean_baseline_tokens":b.output_tokens.mean(),
           "additional_tokens_per_request":float(np.mean(g.output_tokens.to_numpy()-b.output_tokens.to_numpy())),
           "baseline_correct_method_wrong":int(harmful.sum()),
           "harmful_first_line_recoverable":int((harmful&(g.first_line_correct.to_numpy()==1)).sum()),
           "harmful_labeled_prefix_present":int((harmful&g.labeled_answer_prefix.to_numpy()).sum()),
           "harmful_no_eos":int((harmful&~g.terminated.to_numpy()).sum())}
        summary.append(s)
    args.output.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output/"termination_rows.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
