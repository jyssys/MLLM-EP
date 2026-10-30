"""Disjoint SLO calibration and a frozen, small static tuning budget."""
import argparse
import json
import math
import random
from .common import *
from .prepare import base

def make():
    for model in ('ouro','huginn'):
        for wi,w in enumerate(('W1','W2','W3','W4')):
            name=f'calibration_{model}_{w.lower()}_pd_g1'
            cfg=base(model,'PD_G1',name,f'{w}_calibration.json',8,[0,1])
            cfg.update(workload=w,arrival_offsets=[i*.002 for i in range(8)],timeout=3600,
                       estimated_seconds=900,calibration=True,warmup_count=2)
            write(ROOT/'configs'/f'{name}.json',cfg)
    write(ROOT/'configs/calibration_rules.json',dict(
        SLO_rule='TTFT threshold = 1.5 x baseline request TTFT p95. TPOT threshold = 1.5 x baseline request TPOT p95.',
        calibration_requests=8,baseline='PD_G1, disjoint complete ShareGPT prompts, controlled original output targets.',
        main_rate_rule='0.8 x baseline completed requests divided by cohort makespan. W2 factor1.0, W4 factor1.1.',
        burst='W3: Poisson background, central eight long-prompt arrivals at five times baseline rate.',
        caution='Small calibration is diagnostic, not a validated SLO guarantee or capacity estimate.',
        static_tuning='All candidate cuts use disjoint calibration inputs. Never select on held-out serving outputs.'))

def percentile(x,p):
    x=sorted(x);z=(len(x)-1)*p;a=int(z);b=min(a+1,len(x)-1)
    return x[a]+(x[b]-x[a])*(z-a)

def freeze():
    results={}
    for model in ('ouro','huginn'):
        for w in ('W1','W2','W3','W4'):
            name=f'calibration_{model}_{w.lower()}_pd_g1';directory=ROOT/'raw'/name
            if not (directory/'status.json').exists():continue
            if json.loads((directory/'status.json').read_text())['status']!='COMPLETE':continue
            rows=[json.loads(s) for s in (directory/'requests.jsonl').read_text().splitlines() if s]
            rows=[r for r in rows if not r['warmup']];window=json.loads((directory/'window.json').read_text())
            rate=len(rows)/(window['end']-window['start'])*{'W1':.8,'W2':1.,'W3':.8,'W4':1.1}[w]
            results[f'{model}_{w}']=dict(ttft_slo_ms=1.5*percentile([r['ttft_ms'] for r in rows],.95),
                tpot_slo_ms=1.5*percentile([r['tpot_ms'] for r in rows],.95),rate=rate,
                baseline_name=name,request_count=len(rows),freeze_time=time.time(),
                prefill_ms_per_token=percentile([r['ttft_ms']/r['input_tokens'] for r in rows],.5),
                input_lengths=[r['input_tokens'] for r in rows],output_lengths=[r['output_tokens'] for r in rows],
                arrival_pattern='Eight arrivals separated by2ms. Saturated diagnostic calibration, not Poisson main serving.',
                coefficient_scope='Host TTFT per input token, including queueing. Not pure GPU Prefill time.',
                W3_scope='Normal-branch calibration. The held-out W3 adds the explicitly defined long-input burst.' if w=='W3' else None)
    dest=ROOT/'slo_calibration.json'
    if dest.exists():raise FileExistsError('Frozen SLOs cannot be silently replaced')
    write(dest,results)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['make','freeze']);a=p.parse_args()
    globals()[a.action]()

if __name__=='__main__':main()
