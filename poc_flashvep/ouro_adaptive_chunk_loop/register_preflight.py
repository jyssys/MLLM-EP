"""A train-only harness check, separate from the frozen factorial matrix."""
from .common import ROOT,write

def main():
    modes=['atomic','split','grouped','grouped3','staged4','mixed']
    jobs=[]
    for gpu in range(8):
        order=modes[gpu%6:]+modes[:gpu%6]
        jobs.append(dict(id=f'p2_harness_preflight_gpu{gpu}_v1',kind='surface',gpu=gpu,shape=[2033,4],
            cells=[[m,512] for m in order],split='TRAIN',trials=[0],cap=16,budget=2048,trace=True,timeout_s=1800,
            scope='TRAIN_ONLY_HARNESS_VALIDATION_NOT_HELDOUT_PERFORMANCE'))
    target=ROOT/'p2_preflight_plan.json'
    assert not target.exists()
    write(target,dict(gpus=list(range(8)),jobs=jobs))

if __name__=='__main__':
    main()
