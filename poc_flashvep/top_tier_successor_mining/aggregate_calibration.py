"""Aggregate official MoDES modality-normalized layer KL statistics."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    rows=[json.loads(s) for f in args.input.glob("quality_*.jsonl") for s in f.read_text().splitlines()]
    ids=[r["request_id"] for r in rows]
    assert len(set(ids))==len(ids),"Duplicate calibration samples"
    sums=np.zeros((48,2),dtype=np.float64)
    for row in rows:
        for l in range(48):
            for m,mod in enumerate(["text","vision"]):
                sums[l,m]+=row["kl_sums"][f"{l}_{mod}"]
    alpha=sums/np.maximum(sums.sum(axis=0,keepdims=True),1e-30)
    out={"alpha":alpha.tolist(),"raw_kl_sums":sums.tolist(),"requests":len(rows),
         "request_ids":ids,"answer_tokens":sum(r["answer_tokens"] for r in rows),
         "semantics":sorted(set(r["ablation_semantics"] for r in rows)),
         "logit_mask":sorted(set(r["logit_mask"] for r in rows)),
         "source":str(args.input),"normalization":"sum layer KL separately per modality, official implementation"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({"requests":len(rows),"alpha_sum":alpha.sum(0).tolist(),"output":str(args.output)}))


if __name__ == "__main__":
    main()
