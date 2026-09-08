"""Paired port equivalence/cost controls, separate from scientific successors."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
r=a.result;out=r/'analysis/resume_controls_20260908';out.mkdir(parents=True,exist_ok=True)
results={}
source=r/'online/ep_port_control_metadata_20260908'
if (source/'completed.json').exists():
    rows=[json.loads(s) for f in source.glob('requests_dp*.jsonl') for s in f.read_text().splitlines()]
    df=pd.DataFrame([x for x in rows if not x['warmup']]);paired=[];summary=[]
    keys=['cohort','dp_rank','dataset_request']
    for policy in sorted(df.policy.unique()):
        if not policy.endswith('_metadata_cached'):continue
        reference=policy.removesuffix('_metadata_cached')
        pair=df[df.policy==policy].merge(df[df.policy==reference],on=keys,suffixes=('','_ref'),validate='one_to_one')
        assert len(pair)>0
        same=np.array([x==y for x,y in zip(pair.output_tokens,pair.output_tokens_ref)])
        reduction=100*(1-pair.e2e_ms/pair.e2e_ms_ref)
        groups=[g.index.to_numpy() for _,g in pair.groupby('cohort')]
        rng=np.random.default_rng(7382)
        boots=[float(reduction.iloc[np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])].median()) for _ in range(3000)]
        summary.append(dict(policy=policy,reference=reference,paired_requests=len(pair),
            independent_engine_runs=1,greedy_sequence_equal_fraction=float(same.mean()),
            metadata_cache_reduction_median_percent=float(reduction.median()),
            reduction_paired_cohort_95ci_percent=np.quantile(boots,[.025,.975]).tolist(),
            paired_cohort_units=len(groups),
            claim_scope='EQUIVALENT_PORT_LIFETIME_CONTROL_NOT_NEW_METHOD'))
        for i,row in pair.iterrows():
            paired.append({**{k:row[k] for k in keys},'policy':policy,'same_output':bool(same[i]),
                'e2e_reduction_percent':float(reduction.iloc[i])})
    pd.DataFrame(paired).to_csv(out/'metadata_pairs.csv',index=False)
    results['metadata_cache']=summary
source=r/'online/ep_port_control_scheduler_20260908'
if (source/'completed.json').exists():
    df=pd.DataFrame([json.loads(s) for f in source.glob('scheduler_context.pid*.jsonl') for s in f.read_text().splitlines()])
    assert not df.duplicated(['dp_rank','step']).any()
    df.to_csv(out/'scheduled_populations.csv',index=False)
    summary=[]
    for (policy,phase),g in df.groupby(['policy','phase']):
        summary.append(dict(policy=policy,phase=phase,source_steps=len(g),
            active_requests_min=int(g.active_requests.min()),active_requests_p50=float(g.active_requests.median()),
            active_requests_p90=float(g.active_requests.quantile(.9)),active_requests_max=int(g.active_requests.max()),
            scheduled_tokens_p50=float(g.scheduled_tokens.median()),
            count_includes_warmup=True,offered_cohort_cap_per_dp=16,
            note='One TP representative per DP; not logical EP-layer invocations.'))
    results['scheduler_membership']=summary
for key in ('fixed1','fixed16','phase','decision'):
    source=r/'analysis'/f'ep_port_control_{key}_20260908'/'summary.json'
    if source.exists():
        results[key]=[{k:row[k] for k in ('policy','paired_quality_delta_pp',
            'paired_e2e_reduction_median_percent','paired_e2e_reduction_95ci_percent',
            'ttft_ms_p50','tpot_ms_p50','quality_metric_interpretable','fixed_output',
            'output_tokens_mean','cohort_throughput_output_tokens_s')} for row in json.loads(source.read_text())]
(out/'summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
