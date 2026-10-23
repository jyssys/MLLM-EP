"""Freeze correctness and fresh path-service pilot inputs before GPU execution."""
import json
import random
from .common import ROOT,write,CHUNKS

def main():
    assert not (ROOT/'p1_numerical_plan.json').exists()
    jobs=[dict(id=f'cb_p1_numerical_gpu{gpu}_v1',kind='numerical',gpu=gpu,
        lengths=[128,512] if gpu%2==0 else [1536,2048],timeout_s=1800) for gpu in range(8)]
    write(ROOT/'p1_numerical_plan.json',dict(gpus=list(range(8)),jobs=jobs,
        tolerance='Predeclared BF16 relative L2 <=0.03 for hidden/logits/KV. Discrete differences remain separately recorded. Full R4 and frontier integrity mandatory.'))
    service=[]
    for i,(P,N) in enumerate(((2033,4),(8177,8))):
        for j,C in enumerate(CHUNKS):
            for k,split in enumerate(('TRAIN','VALIDATION','CALIBRATION','NEW_HELDOUT')):
                cells=[dict(mode=mode,C=C,fairness='stock') for mode in ('atomic','staged4','split','grouped','grouped3','mixed')]
                random.Random(20261023+i*100+j*4+k).shuffle(cells)
                service.append(dict(id=f'cb_p2_P{P}_N{N}_C{C}_{split.lower()}_v1',kind='service',gpu=(i*3+j+k)%8,
                    shape=[P,N],split=split,cells=cells,trials=[0,1],cap=32,allow_any_gpu=True,budget=2048,timeout_s=1800))
    write(ROOT/'p2_service_plan.json',dict(gpus=list(range(8)),jobs=service,
        scope='Fresh source families and readiness-separated native service. Shared-host diagnostic, not clean serving headline.'))
    fairness=[]
    for gpu in range(8):
        cells=[dict(mode=mode,C=1024 if mode!='mixed' else 512,fairness=fair)
            for mode in ('atomic','staged4','grouped','split','mixed') for fair in ('stock','completion')]
        random.Random(20261023+gpu).shuffle(cells)
        fairness.append(dict(id=f'cb_p1_fairness_gpu{gpu}_v1',kind='fairness',gpu=gpu,
            shape=[2033,4] if gpu%2==0 else [8177,8],split='TRAIN',cells=cells,trials=[0,1],cap=64,
            budget=2048,timeout_s=1800))
    write(ROOT/'p1_fairness_plan.json',dict(gpus=list(range(8)),jobs=fairness,
        scope='Same device and fixed C/G. Completion fairness is independent from optimization.'))
    print('Registered',len(jobs),'numerical,',len(service),'fresh cost,',len(fairness),'fairness jobs')

if __name__=='__main__':
    main()
