"""Predeclare quiet confirmation and causal component checks."""
from .common import ROOT,write

def main():
    jobs=[]
    for gpu in range(8):
        n=(1,2,4,8)[gpu%4]
        jobs.append(dict(id=f'p1_numeric_v3_g{gpu}',gpu=gpu,kind='numerical',shapes=[[256,n],[2048,n]],
            groups=[1,2,4,3],graphs=gpu<4,**{'async':gpu%2!=0},blocks=18432,timeout_s=3600))
    write(ROOT/'p1_crossed_plan.json',dict(gpus=list(range(8)),jobs=jobs))
    write(ROOT/'lifecycle_plan.json',dict(gpus=[0,4],jobs=[dict(id=f'p1_lifecycle_v1_g{gpu}',gpu=gpu,
        kind='lifecycle',graphs=True,**{'async':gpu==0},timeout_s=1800) for gpu in (0,4)]))
    # Every process crosses all D on the same GPU. Starts rotate devices/order.
    groups=[]
    for start in range(3):
        for gpu in (0,4):
            ds=[1,2,4];ds=ds[start:]+ds[:start]
            jobs=[dict(id=f'p2_quiet_v1_s{start}_g{gpu}',gpu=gpu,kind='timing',
                shapes=[[256,1],[2048,4],[8192,1],[2048,16]],groups=ds,graphs=True,**{'async':True},
                cap=128,repeats=3,lane='QUIET_CONFIRMATION',timeout_s=5400)]
            groups.append(dict(jobs=jobs))
    write(ROOT/'quiet_plan.json',dict(gpus=list(range(8)),groups=groups))
    write(ROOT/'barrier_plan.json',dict(gpus=list(range(8)),jobs=[dict(id=f'p2_barrier_v1_g{gpu}',gpu=gpu,
        kind='timing',shapes=[[2048,16],[256,4]],groups=[1,2,4],graphs=True,**{'async':True},
        decode_barrier=True,cap=64,repeats=3,lane='MATCHED_DECODE_FRONTIER',timeout_s=3600) for gpu in range(8)]))
    jobs=[]
    for gpu in range(8):
        jobs.append(dict(id=f'p2_intervene_v2_g{gpu}',gpu=gpu,kind='intervention',
            cells=[dict(P=(256,2048)[gpu//4],old=(1,2)[(gpu//2)%2],new=(1,2)[gpu%2])],graphs=True,blocks=18432,timeout_s=1800))
    write(ROOT/'intervention_v2_plan.json',dict(gpus=list(range(8)),jobs=jobs))
    write(ROOT/'profile_plan.json',dict(gpus=[0],jobs=[dict(id='p2_profile_v1_g0',gpu=0,kind='profile',
        shapes=[[256,1],[2048,4],[2048,16]],graphs=True,**{'async':True},timeout_s=1800)]))

if __name__=='__main__':main()
