"""Register same-process C-by-G interaction blocks before observing results."""
import random
from .common import ROOT,write

def main():
    target=ROOT/'p2_paired_service_plan.json'
    assert not target.exists()
    jobs=[]
    for gpu in range(8):
        cells=[dict(mode=mode,C=C,fairness='completion') for C in (128,512,2048)
            for mode in ('split','grouped','grouped3','staged4')]
        random.Random(20261023+700+gpu).shuffle(cells)
        jobs.append(dict(id=f'cb_p2_paired_gpu{gpu}_v1',kind='service',gpu=gpu,
            shape=[2033,4] if gpu%2==0 else [8177,8],split='TRAIN',cells=cells,trials=[2,3],cap=32,
            budget=2048,timeout_s=2400))
    write(target,dict(gpus=list(range(8)),jobs=jobs,
        scope='Every C/G cell crosses within one GPU/model-process block. Diagnostic shared host. No clean serving claim.'))
    print('Registered',len(jobs),'paired interaction blocks')

if __name__=='__main__':
    main()
