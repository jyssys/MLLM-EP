"""Generate immutable staged plans. No unmeasured performance is encoded."""
from .common import ROOT,write

def main():
    jobs=[]
    for gpu in range(8):
        n=(1,2,4,8)[gpu%4]
        jobs.append(dict(id=f'p1_numeric_v2_g{gpu}',gpu=gpu,kind='numerical',shapes=[[256,n],[2048,n]],
            groups=[1,2,4,3],graphs=gpu<4,**{'async':gpu%2==0},blocks=18432,timeout_s=5400))
    write(ROOT/'p1_v2_plan.json',dict(gpus=list(range(8)),jobs=jobs))
    jobs=[]
    for gpu in range(8):
        shapes=[[256,n] for n in (1,2,4,8,16)]+[[2048,n] for n in (1,2,4,8,16)]+[[8192,n] for n in (1,2,4,8)]
        # All policies cross every GPU. Ordering rotates to avoid a fixed D/GPU effect.
        groups=[1,2,4];groups=groups[gpu%3:]+groups[:gpu%3]
        jobs.append(dict(id=f'p2_screen_v2_g{gpu}',gpu=gpu,kind='timing',shapes=shapes,
            groups=groups,graphs=True,**{'async':True},repeats=2,cap=64,blocks=18432,timeout_s=10800))
    write(ROOT/'p2_screen_v2_plan.json',dict(gpus=list(range(8)),jobs=jobs))

if __name__=='__main__':main()
