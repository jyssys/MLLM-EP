"""Measured parameter/fallback attacks, not an ideal unmeasured successor."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
r=a.result;out=r/'analysis/trivial_fix_attacks_20260908';out.mkdir(parents=True,exist_ok=True)
rows=[]
for source in sorted((r/'online').glob('ep_clean_renderer_*_run0_20260908')):
    if not (source/'completed.json').exists() or (source/'INVALIDATED.json').exists():continue
    summary=json.loads((r/'analysis'/source.name/'summary.json').read_text())
    by_policy={x['policy']:x for x in summary}
    raw=pd.DataFrame([json.loads(s) for f in source.glob('requests_dp*.jsonl') for s in f.read_text().splitlines()])
    raw=raw[~raw.warmup]
    base='sere_s2_r05_official'
    for fix in ('sere_s4_r05_official','sere_s2_r07_official'):
        if base not in by_policy or fix not in by_policy:continue
        pair=raw[raw.policy==fix].merge(raw[raw.policy==base],
            on=['cohort','dp_rank','dataset_request'],suffixes=('','_ref'),validate='one_to_one')
        reduction=100*(1-pair.e2e_ms/pair.e2e_ms_ref)
        groups=[g.index.to_numpy() for _,g in pair.groupby('cohort')]
        rng=np.random.default_rng(9122)
        boots=[float(reduction.iloc[np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])].median()) for _ in range(3000)]
        delta_base=by_policy[base]['paired_quality_delta_pp']
        delta_fix=by_policy[fix]['paired_quality_delta_pp']
        rows.append(dict(run=source.name,base=base,fix=fix,
            baseline_quality_delta_pp=delta_base,fix_quality_delta_pp=delta_fix,
            observed_quality_recovery_pp=delta_fix-delta_base,
            fraction_of_observed_quality_loss_recovered=(delta_fix-delta_base)/(-delta_base) if delta_base<0 else None,
            e2e_reduction_vs_original_policy_median_percent=float(reduction.median()),
            e2e_paired_cohort_ci95_percent=np.quantile(boots,[.025,.975]).tolist(),
            quality_loss_recovery_scope='Observed finite sample; not per-request perfect oracle',
            timing_scope='One engine per dataset/cohort cap; randomized within-cohort policies'))
(out/'summary.json').write_text(json.dumps(rows,indent=2))
pd.DataFrame(rows).to_csv(out/'measured_trivial_fixes.csv',index=False)
print(json.dumps(rows,indent=2))
