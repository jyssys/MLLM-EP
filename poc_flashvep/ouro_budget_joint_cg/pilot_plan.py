"""Freeze a balanced TRAIN-only budget dose pilot after the cost-model freeze."""
import json
import random
from collections import defaultdict
import numpy as np
from .common import ROOT,write,digest,BUDGETS,CHUNKS
from .cost_predictor import ServiceLookup
from .optimizer import enumerate_candidates
from .metrics import summarize

def trained_fixed():
    values=defaultdict(list)
    for folder in (ROOT/'raw').glob('cb_p2_P*train_v1'):
        assert (folder/'completed.json').exists()
        for file in folder.glob('*_T*.json'):
            d=json.loads(file.read_text())
            if 'requests' not in d:
                continue
            values[(d['P'],d['configured_N'],d['policy'],d['C'])].append(summarize(d))
    def score(key):
        rows=values[key];tail=np.mean([r['ITL_pooled_P99_ms'] for r in rows])
        return (max(0.,tail/200.-1.),np.mean([r['TTFT_P95_ms'] for r in rows]),
            -np.mean([r['drained_output_tokens_s'] for r in rows]))
    choices={}
    for P,N in ((2033,4),(8177,8)):
        for mode in ('atomic','mixed'):
            keys=[k for k in values if k[:3]==(P,N,mode)]
            assert keys
            choices[(P,N,mode)]=min(keys,key=score)[3]
    return choices,dict(rule='TRAIN-only lexicographic excess over 200ms pooled ITL P99, then TTFT P95, then drained tokens/s. Bounded diagnostic tuning, not qualified SLO capacity.',
        choices=[dict(P=k[0],N=k[1],mode=k[2],C=v) for k,v in choices.items()])

def choose_G_C(lookup,budget,P):
    state=dict(kind='NEW_CHUNK',C=0,depth=0,start=P//2,prompt_remaining=P-P//2,token_budget=2048,issued_decode=[dict(id='observed')])
    feasible=[]
    for C in CHUNKS:
        choices=enumerate_candidates(state,budget,lookup,'G_ONLY',C,'U2',0.)
        if choices:
            feasible.append((max(c.utility for c in choices),C))
    return max(feasible)[1] if feasible else 64

def main():
    target=ROOT/'p3_budget_pilot_plan.json';assert not target.exists()
    artifact=ROOT/'cost_model_artifacts/service_lookup_v1.json'
    freeze=json.loads((ROOT/'cost_model_artifacts/freeze_v1.json').read_text());assert digest(artifact)==freeze['sha256']
    lookup=ServiceLookup(json.loads(artifact.read_text()));fixed,selection=trained_fixed()
    write(ROOT/'pilot_training_selection.json',selection)
    jobs=[]
    for startup in range(2):
        for shape,(P,N) in enumerate(((2033,4),(8177,8))):
            for index,B in enumerate(BUDGETS):
                gpu=(index+startup*3+shape*2)%8
                base=dict(C=2048,budget_ms=B,fairness='completion',yield_penalty_ms=0.)
                cells=[dict(base,label='C1A',runtime='atomic',optimizer='C_ONLY'),
                    dict(base,label='C1S',runtime='split',optimizer='C_ONLY'),
                    dict(base,label='G1',runtime='split',optimizer='G_ONLY',fixed_C=choose_G_C(lookup,B,P)),
                    dict(base,label='J1',runtime='split',optimizer='JOINT'),
                    dict(label='M0',runtime='mixed',C=fixed[(P,N,'mixed')],fairness='stock'),
                    dict(label='M1',runtime='mixed',C=fixed[(P,N,'mixed')],fairness='completion'),
                    dict(label='F1A',runtime='atomic',C=fixed[(P,N,'atomic')],fairness='completion')]
                random.Random(20261023+startup*1000+shape*100+index).shuffle(cells)
                jobs.append(dict(id=f'cb_p3_budget_P{P}_N{N}_B{B}_S{startup}_v1',kind='live_budget',gpu=gpu,
                    shape=[P,N],split='TRAIN',trials=[9+2*startup,10+2*startup],cap=128,cells=cells,
                    cost_model=str(artifact.relative_to(ROOT)),cost_model_sha256=digest(artifact),
                    observe=True,budget=2048,timeout_s=3600,allow_any_gpu=False))
    write(target,dict(gpus=list(range(8)),jobs=jobs,
        scope='TRAIN-only fixed-budget natural pilot. Two device/process starts per shape/budget. Observer/shared-host lane. Not capacity or final held-out.',
        utility='U2 includes prelude and all residual R4 service. No additional yield penalty in this first physical-cost screen.',
        budget_grid=list(BUDGETS),SLO_grid=dict(TTFT_s=[1,2,5,10],every_gap_TBT_ms=[100,150,200,250,500]),
        important='Native fixed/Mixed retain native packing. All adaptive controls limit one prefill span per batch.'))
    print('Registered',len(jobs),'budget-pilot crossover jobs')

if __name__=='__main__':
    main()
