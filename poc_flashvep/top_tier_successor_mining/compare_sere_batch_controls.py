"""Matched B1/B16 official quality controls; not EP speed or modality causality."""
import argparse,json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
r=a.result;single=pd.read_csv(r/'analysis/official_confirmatory_b1_20260908/scored_predictions.csv')
summary=[]
for dataset in ('chartqa','gqa'):
    stem='chart' if dataset=='chartqa' else 'gqa'
    batch=pd.read_csv(r/f'analysis/sere_official_{stem}128_b16_20260908/scored_predictions.csv')
    a1=single[single.dataset==dataset]
    pairs=a1.merge(batch,on=['request_id','policy'],suffixes=('_b1','_b16'),validate='one_to_one')
    v=pairs[pairs.policy=='vanilla'].set_index('request_id')
    stable=set(v[(v.correct_b1==1)&(v.correct_b16==1)].index)
    for policy,g in pairs.groupby('policy'):
        s=g[g.request_id.isin(stable)]
        failed=s[s.correct_b1==0]
        summary.append(dict(dataset=dataset,policy=policy,n=len(g),
            accuracy_b1=100*g.correct_b1.mean(),accuracy_b16=100*g.correct_b16.mean(),
            output_text_same_fraction=(g.prediction_b1.fillna('')==g.prediction_b16.fillna('')).mean(),
            stable_correct_vanilla_requests=len(stable),b1_failures_on_stable_vanilla=len(failed),
            rescued_at_b16=int(failed.correct_b16.sum()),
            rescued_fraction=float(failed.correct_b16.mean()) if len(failed) else None,
            scope='OFFICIAL_HF_BATCH_CONTROL; no EP speed claim; not independent modality control',
            serving_scope_caveat='HF keeps cohort tensor rows while finished sequences become padding; vLLM removes EOS requests. Active union lifetime differs; online B16 still required.'))
out=r/'analysis/sere_batch_control_20260908';out.mkdir(exist_ok=True)
pd.DataFrame(summary).to_csv(out/'paired_batch_control.csv',index=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
