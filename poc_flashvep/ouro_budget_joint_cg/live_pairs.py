"""Same-process paired pilot effects. No token-IID confidence intervals."""
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .cost_predictor import csv_file
from .metrics import summarize,paired_block_interval

def main():
    blocks=defaultdict(list);parity=[];raw=[]
    for pattern in ('cb_p3_budget_*','cb_p5_confirm_*'):
        for folder in (ROOT/'raw').glob(pattern):
            if not (folder/'completed.json').exists():
                continue
            identity=json.loads((folder/'identity.json').read_text());job=identity['job']
            lane='QUIET_FINAL' if folder.name.startswith('cb_p5_confirm') else 'TRAIN_OBSERVER'
            B=next(c['budget_ms'] for c in job['cells'] if c['label']=='J1')
            by_trial=defaultdict(dict)
            for file in folder.glob('*_T*.json'):
                data=json.loads(file.read_text())
                if 'requests' not in data:
                    continue
                by_trial[data['trial']][data['label']]=data
            for trial,policies in by_trial.items():
                if 'J1' not in policies:
                    continue
                for reference in ('C1A','C1S','G1','M0','M1','F1A'):
                    if reference not in policies:
                        continue
                    a=policies[reference];b=policies['J1']
                    av,bv=summarize(a),summarize(b)
                    for metric in ('TTFT_P95_ms','ITL_pooled_P95_ms','ITL_pooled_P99_ms',
                            'drained_output_tokens_s','joint_declared_drain_goodput_rps'):
                        blocks[(lane,job['shape'][0],B,reference,metric,folder.name)].append((av[metric],bv[metric]))
                    left={r['id']:r for r in a['requests']};right={r['id']:r for r in b['requests']}
                    parity.append(dict(run_id=folder.name,trial=trial,P=job['shape'][0],budget_ms=B,reference=reference,
                        requests=len(left),token_exact=sum(left[k]['output_ids']==right[k]['output_ids'] for k in left),
                        length_exact=sum(len(left[k]['output_ids'])==len(right[k]['output_ids']) for k in left),
                        reference_tokens=sum(len(r['output_ids']) for r in left.values()),
                        joint_tokens=sum(len(r['output_ids']) for r in right.values()),evidence=b['lane']))
                    for ttft in (1.,2.,5.,10.):
                        for tbt in (.1,.15,.2,.25,.5):
                            for label,data in ((reference,a),('J1',b)):
                                row=summarize(data,ttft,tbt)
                                raw.append(dict(run_id=folder.name,trial=trial,P=job['shape'][0],budget_ms=B,
                                    reference_pair=reference,policy=label,**row,evidence=data['lane']))
    grouped=defaultdict(list)
    for (lane,P,B,reference,metric,run),values in blocks.items():
        grouped[(lane,P,B,reference,metric)].append((np.mean([a for a,b in values]),np.mean([b for a,b in values])))
    effects=[]
    for (lane,P,B,reference,metric),values in sorted(grouped.items()):
        ci=paired_block_interval(values)
        a=np.mean([x for x,y in values]);b=np.mean([y for x,y in values])
        effects.append(dict(lane=lane,P=P,budget_ms=B,reference=reference,candidate='J1',metric=metric,
            reference_mean=a,joint_mean=b,ratio_joint_over_reference=b/a if a else None,**ci,
            scope='Block-paired natural pilot. Mixed implementation versions in initial TRAIN only. Quiet corrected confirmation separate.'))
    csv_file(ROOT/'pilot_paired_effects.csv',effects);csv_file(ROOT/'pilot_output_parity.csv',parity)
    csv_file(ROOT/'pilot_slo_grid.csv.gz',raw)
    print('Paired effects',len(effects),'request-cohort parity pairs',len(parity))

if __name__=='__main__':
    main()
