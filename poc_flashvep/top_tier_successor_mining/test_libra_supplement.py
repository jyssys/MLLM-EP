"""Implementation invariants, not scientific synthetic workload evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from libra_supplement import SupplementPlanner


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extension",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    rng=np.random.default_rng(7317)
    rows=[]
    for n,l in [(4,4),(8,4),(8,2)]:
        planner=SupplementPlanner(args.extension,n,l)
        for m in [1,8,64,512]:
            for variant in ["same","shifted","uniform","concentrated"]:
                pred=np.argsort(rng.random((4,m,128)),axis=-1)[...,:8].copy()
                actual=pred.copy() if variant=="same" else np.argsort(rng.random((4,m,128)),axis=-1)[...,:8].copy()
                if variant=="shifted":actual=(pred+32)%128
                if variant=="concentrated":actual%=16
                p=planner.prefetch(pred)
                r=planner.rebalance(actual,p)
                rows.append({"M":m,"N":n,"L":l,"variant":variant,"loads":r["loads"].tolist(),
                             "assignments":int(actual.size),"status":"PASS"})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({"tests":len(rows),"rows":rows},indent=2))
    print(json.dumps({"tests":len(rows),"status":"PASS"}))


if __name__=="__main__":main()
