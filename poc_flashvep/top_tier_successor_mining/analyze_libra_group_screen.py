"""Across-input native Libra summary; independent groups, never four-rank pseudoreps."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
r=a.result;out=r/'analysis/libra_eight_groups_20260908';out.mkdir(parents=True,exist_ok=True)
records=[];details=[]
for folder in sorted((r/'online').glob('libra_native_vl48_*_frozen_control_20260908')):
    if len(list(folder.glob('completed_*.json')))!=4:continue
    args=json.loads((folder/'arguments.json').read_text())
    data=pd.DataFrame([json.loads(s) for f in folder.glob('rank*.jsonl') for s in f.read_text().splitlines()])
    data=data[data['repeat']>=0].copy()
    assert not data.duplicated(['rank','case','repeat','policy']).any()
    assert data[data.policy=='libra_frozen_oracle'].oracle_realized_topk_recall_min.min()==1
    data['group']=args['vl_group'];data['M_per_source']=int(args['tokens'])
    records.append(data)
    critical=data.groupby(['repeat','policy']).prefill_ms.max().unstack('policy')
    gain=100*(1-critical.libra_frozen_oracle/critical.libra_frozen)
    natural_gain=100*(1-critical.libra_oracle/critical.libra)
    details.append(dict(group=args['vl_group'],M_per_source=int(args['tokens']),
        paired_executions=len(critical),vanilla_critical_p50_ms=float(critical.vanilla.median()),
        libra_critical_p50_ms=float(critical.libra.median()),
        fixed_current_route_perfect_prediction_gain_percent=float(gain.median()),
        vanilla_trace_lookahead_gain_percent=float(natural_gain.median())))
assert records,'No completed native groups'
assert a.allow_partial or len(records)==8, 'Final report requires all eight groups'
df=pd.concat(records,ignore_index=True);group_table=pd.DataFrame(details)
summary=[]
for policy,g in df.groupby('policy'):
    summary.append(dict(policy=policy,groups=g.group.nunique(),distinct_requests=g.dataset_request.nunique(),
        rank_observations=len(g),greedy_vs_native_vanilla_equal_fraction=float(g.greedy_first_equal.mean()),
        greedy_vs_hf_reference_equal_fraction=float(g.hf_reference_greedy_equal.mean()),
        max_kl_vs_native_vanilla=float(g.logit_kl_vanilla_to_policy.max()),
        max_kl_vs_hf=float(g.hf_reference_kl_to_policy.max()),
        min_logit_cosine_vs_hf=float(g.hf_reference_cosine.min()),
        fixed_route_replay_not_quality=policy in ('libra_frozen','libra_frozen_oracle')))
values=group_table.fixed_current_route_perfect_prediction_gain_percent.to_numpy()
rng=np.random.default_rng(9821)
boots=np.median(values[rng.integers(0,len(values),(5000,len(values)))],axis=1)
result=dict(complete_groups=len(records),expected_groups=8,partial=len(records)!=8,
    scope='Author native post-vision48-layer prefill; not full request latency. Predictor computation retained in both fixed-route variants.',
    oracle_gain_group_matched_median_percent=float(np.median(values)),
    oracle_gain_group_bootstrap95ci_percent=np.quantile(boots,[.025,.975]).tolist(),
    oracle_gain_range_percent=[float(values.min()),float(values.max())],
    group_results=details,quality_parity=summary,
    warnings=['Original paper235B/355B8H200 performance is not reproduced.',
              'Perfect prediction of fixed current routes is not a universal successor upper bound.',
              'BF16 cross-backend numerical parity is approximate; full generation/benchmark parity is not measured.'])
bad_groups=set(df[(df.policy=='vanilla') & (~df.hf_reference_greedy_equal)].group)
kept=group_table[~group_table.group.isin(bad_groups)]
result['hf_disagreement_group_sensitivity']=dict(excluded_groups=sorted(bad_groups),
    remaining_groups=len(kept),
    oracle_gain_group_median_percent=float(kept.fixed_current_route_perfect_prediction_gain_percent.median()) if len(kept) else None)
group_table.to_csv(out/'paired_group_results.csv',index=False)
df.to_csv(out/'native_rank_observations.csv',index=False)
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
