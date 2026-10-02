"""Secondary fixed-layer pairs; bootstrap requests, not75k layer events."""
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from poc_speculative_moe_ep.contract import ROOT,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--trace-version',default='phase_a_v1');a=p.parse_args()
    root=ROOT/'artifacts'/a.version
    if (root/'layer_matched_pairs.json').exists():raise FileExistsError(root/'layer_matched_pairs.json')
    traced=ROOT/'artifacts'/a.trace_version
    routes=pd.read_parquet(traced/'route_metrics.parquet')
    v=pd.read_parquet(traced/'verification_metrics.parquet')
    routes=routes.merge(v[['run','wave','forward','committed_position','verified_rows','committed_progress']],on=['run','wave','forward'],validate='many_to_one')
    clean=pd.read_parquet(root/'verification_metrics.parquet')
    durations=defaultdict(list)
    for run in clean.run.unique():
        subset=clean[clean.run==run]
        look={(r.wave,r.forward):r for r in subset.itertuples()}
        for rank in range(4):
            for line in (ROOT/'raw'/run/f'frames.rank{rank}.jsonl').read_text().splitlines():
                f=json.loads(line);row=look.get((f['wave'],f['forward']))
                if row is None or row.critical_rank!=rank:continue
                for layer,l in enumerate(f['layers']):
                    key=(row.request_id,row.gamma,row.committed_position,layer)
                    durations[key].append(l['dispatch_ms']+l['combine_ms'])
    routes=routes[routes.gamma>0].copy()
    routes['ep_clean_ms']=[np.median(durations[(r.request_id,r.gamma,r.committed_position,r.layer)]) for r in routes.itertuples()]
    pairs=[]
    for key,g in routes.groupby(['layer','gamma','verified_rows','committed_progress','unique_experts','actual_unique_experts']):
        rows=g.sort_values(['max_rank_bytes','request_id','committed_position']).to_dict('records');used=set()
        for low in rows:
            if low['request_id'] in used:continue
            available=[r for r in rows if r['request_id'] not in used and r['request_id']!=low['request_id'] and r['max_rank_bytes']>=1.15*low['max_rank_bytes']]
            if not available:continue
            high=available[0];used.update((low['request_id'],high['request_id']))
            pairs.append(dict(layer=int(key[0]),gamma=int(key[1]),verified_rows=int(key[2]),committed_progress=int(key[3]),
                unique_experts=int(key[4]),actual_unique_experts=int(key[5]),
                low_request=low['request_id'],high_request=high['request_id'],
                low_position=int(low['committed_position']),high_position=int(high['committed_position']),
                low_bytes=int(low['max_rank_bytes']),high_bytes=int(high['max_rank_bytes']),
                low_ep_ms=float(low['ep_clean_ms']),high_ep_ms=float(high['ep_clean_ms']),
                traffic_delta_pct=float(100*(high['max_rank_bytes']/low['max_rank_bytes']-1)),
                ep_delta_pct=float(100*(high['ep_clean_ms']/low['ep_clean_ms']-1))))
    ids=sorted({r[k] for r in pairs for k in ('low_request','high_request')})
    uncertainty=None
    if pairs:
        n=len(ids);index={rid:i for i,rid in enumerate(ids)}
        lo=np.asarray([index[r['low_request']] for r in pairs]);hi=np.asarray([index[r['high_request']] for r in pairs])
        values=np.asarray([r['ep_delta_pct'] for r in pairs]);rng=np.random.default_rng(3407);draws=[]
        for _ in range(2000):
            count=np.bincount(rng.integers(n,size=n),minlength=n)
            weights=count[lo]*count[hi]
            if weights.sum():draws.append(float(np.average(values,weights=weights)))
        uncertainty=dict(mean=float(values.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
            requests=n,bootstrap=2000,unit='dyadic request-cluster resampling; pair weights=count(low)*count(high); not independent layers')
    write(root/'layer_matched_pairs.json',dict(status='SECONDARY_FIXED_LAYER_MATCHING',n_pairs=len(pairs),requests=len(ids),pairs=pairs,
        median_ep_delta_pct=float(np.median([r['ep_delta_pct'] for r in pairs])) if pairs else None,
        mean_ci=uncertainty,matching='exact layer/gamma/rows/progress/user unique/executed unique; traffic>=15%; timing never used to select pairs',
        timing='median3 clean repeats; one invocation cumulative-critical rank supplies each layer communication',
        limitation='may reuse a request across layers; CI resamples request IDs; descriptive within-width evidence, no individual layer samples counted as independent'))
    print(json.dumps(dict(pairs=len(pairs),requests=len(ids),median_ep_delta_pct=float(np.median([r['ep_delta_pct'] for r in pairs])) if pairs else None,ci=uncertainty)))


if __name__=='__main__':main()
