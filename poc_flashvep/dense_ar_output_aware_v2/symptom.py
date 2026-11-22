"""A red-capable check against paired, actual continuous-serving starts."""
import argparse
import json
from statistics import mean
from .common import ROOT,PREVIOUS

def measure(root,jobs,left='HybridScoped',right='StockToken'):
    ratios=[];raw=[]
    for cell in jobs:
        values={}
        for arm in (left,right):
            name=cell['jobs'][arm]
            metric=json.loads((root/'analysis'/(name+'_metrics.json')).read_text())
            cfg=json.loads((root/'raw'/name/'configuration.json').read_text())
            assert not cfg.get('controlled_pipeline')
            values[arm]=metric['TTFT_p95_ms']
            assert json.loads((root/'raw'/name/'dispatcher_result.json').read_text())['complete']
            raw.append(dict(name=name,dataset=cfg['dataset_SHA256'],seed=cfg['seed'],gpus=cfg['gpus']))
        ratios.append(values[left]/values[right])
    for offset in range(0,len(raw),2):
        assert all(raw[offset][key]==raw[offset+1][key] for key in ('dataset','seed','gpus'))
    return dict(start_ratios=ratios,paired_start_mean_ratio=mean(ratios),
        ratio_of_start_means=mean(json.loads((root/'analysis'/(c['jobs'][left]+'_metrics.json')).read_text())['TTFT_p95_ms'] for c in jobs)/
            mean(json.loads((root/'analysis'/(c['jobs'][right]+'_metrics.json')).read_text())['TTFT_p95_ms'] for c in jobs),n=len(jobs),
        expectation='Hybrid TTFT p95 must not exceed strong compiled Stock Token. Decode safety is a separate mandatory gate.',
        scope='Actual measured C3 paired starts,not a simulator or a kernel-time proxy.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--historical',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    root=PREVIOUS if args.historical else ROOT
    plan=args.plan or 'eager_prefill_native_decode_confirmation_frozen_v2.json'
    cells=json.loads((root/'configs'/plan).read_text())['cells']
    result=measure(root,cells);print(json.dumps(result,indent=2))
    if result['ratio_of_start_means']>1:raise SystemExit('RED: Hybrid remains slower than strong Stock Token in TTFT p95')
    print('TTFT_SYMPTOM_GREEN; accuracy and both-Native decode gates are still required')

if __name__=='__main__':main()
