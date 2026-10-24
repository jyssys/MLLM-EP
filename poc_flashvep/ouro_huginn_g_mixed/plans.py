"""Register balanced fixed-hybrid screens. No adaptive controller is enabled."""
import argparse
from .common import ROOT, write

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','screen','confirm','calibration','audit','no_decode','dp','mechanism','full_state','dp_confirm','dp_warm_confirm','dp_warm_cross']);a=p.parse_args()
    jobs=[]
    if a.stage in ['dp_confirm','dp_warm_confirm','dp_warm_cross']:
        groups=[]
        for block in ([2,3] if a.stage=='dp_confirm' else [7,8] if a.stage=='dp_warm_cross' else [4,5,6]):
            for arm in (['hybrid','mixed'] if block in [2,7,8] else ['mixed','hybrid']):
                barrier=f'dp_huginn_w8_{arm}_s0_r{block}'
                jobs=[dict(id=f'dp_huginn_w8_{arm}_s0_gpu{gpu}_r{block}',kind='dp',gpu=gpu,
                    model='huginn',arm=arm,G=2 if arm=='hybrid' else 1,width=8,
                    barrier=barrier,global_ids=list(range(gpu,32,8)),C=512,N=4,P=128,
                    cap=128,timeout_s=3600,chat_warmup=a.stage!='dp_confirm') for gpu in range(8)]
                groups.append(dict(jobs=jobs,barrier=barrier))
        write(ROOT/'plans'/f'{a.stage}.json',dict(gpus=list(range(8)),groups=groups));return
    if a.stage=='full_state':
        for i,(model,n) in enumerate([(m,n) for m in ['ouro','huginn'] for n in [1,4]]):
            jobs.append(dict(id=f'full_state_intervention_{model}_N{n}_s0',kind='intervene',model=model,gpu=i,
                P=2048,C=512,N=n,cap=24,inject_after=4,timeout_s=3600,full_state=True))
        for i,model in enumerate(['ouro','huginn']):
            jobs.append(dict(id=f'lifecycle_{model}_s0',kind='lifecycle',model=model,gpu=i+4,
                P=64,C=32,N=2,cap=8,timeout_s=1800))
        write(ROOT/'plans'/f'{a.stage}.json',dict(gpus=list(range(8)),jobs=jobs));return
    if a.stage=='mechanism':
        for i,(model,n) in enumerate([(m,n) for m in ['ouro','huginn'] for n in [1,4]]):
            jobs.append(dict(id=f'intervention_{model}_N{n}_s0',kind='intervene',model=model,gpu=i,
                P=2048,C=512,N=n,cap=40,inject_after=4,timeout_s=3600))
        for i,model in enumerate(['ouro','huginn']):
            jobs.append(dict(id=f'event_profile_{model}_s0',kind='profile',model=model,gpu=i+4,
                P=2048,C=512,N=4,cap=32,inject_after=4,timeout_s=3600,
                arms=[['mixed',1],['hybrid',2],['group',2],['group',4]]))
        for i,(model,chunk) in enumerate([(m,c) for m in ['ouro','huginn'] for c in [256,1024]]):
            jobs.append(dict(id=f'chunk_control_{model}_C{chunk}_s0',kind='screen',model=model,gpu=i,
                P=2048,C=chunk,N=2,cap=48,repeats=2,trace=True,rotation=i,timeout_s=3600,
                arms=[['mixed',1],['hybrid',2],['hybrid',4],['group',2],['group',4]]))
        for i,n in enumerate([1,4]):
            jobs.append(dict(id=f'long_control_ouro_P8192_N{n}_s0',kind='screen',model='ouro',gpu=i+6,
                P=8192,C=512,N=n,cap=64,repeats=2,trace=True,rotation=i,timeout_s=3600,
                arms=[['mixed',1],['hybrid',2],['hybrid',3],['hybrid',4],['group',2],['group',4]]))
        for model in ['ouro','huginn']:
            for start in range(3):
                jobs.append(dict(id=f'observer_{model}_s{start}',kind='observer',model=model,gpu=(start*3)%8,
                    P=2048,C=512,N=4,cap=32,rotation=start,timeout_s=3600))
        for i,model in enumerate(['ouro','huginn']):
            jobs.append(dict(id=f'width8_control_{model}_s0',kind='screen',model=model,gpu=i+4,
                P=2048,C=512,N=8,cap=48,repeats=2,trace=True,rotation=i,timeout_s=3600))
        write(ROOT/'plans'/f'{a.stage}.json',dict(gpus=list(range(8)),jobs=jobs));return
    if a.stage=='dp':
        groups=[]
        for model in ['ouro','huginn']:
            for width in [1,2,4,8]:
                for arm in ['mixed','hybrid']:
                    barrier=f'dp_{model}_w{width}_{arm}_s0'
                    registered=[]
                    for gpu in range(width):
                        registered.append(dict(id=f'{barrier}_gpu{gpu}',kind='dp',gpu=gpu,
                            model=model,arm=arm,G=2 if arm=='hybrid' else 1,
                            width=width,barrier=barrier,global_ids=list(range(gpu,32,width)),
                            C=512,N=4,P=128,cap=128,timeout_s=3600))
                    groups.append(dict(jobs=registered,barrier=barrier))
        write(ROOT/'plans'/f'{a.stage}.json',dict(gpus=list(range(8)),groups=groups));return
    if a.stage=='smoke':
        for model,gpu in [('ouro',0),('huginn',1)]:
            jobs.append(dict(id=f'mixed_live_smoke_{model}_s0',kind='screen',model=model,gpu=gpu,
                P=256,C=128,N=2,cap=12,repeats=1,trace=True,rotation=0,
                arms=[['mixed',1],['hybrid',2],['group',2]],timeout_s=1800))
    elif a.stage=='screen':
        # All arms remain paired on one GPU and share the chunk size.
        for i,(model,P,N) in enumerate([(m,p,n) for m in ['ouro','huginn']
                for p in [256,2048] for n in [1,4]]):
            jobs.append(dict(id=f'fixed_screen_{model}_P{P}_N{N}_s0',kind='screen',model=model,gpu=i,
                P=P,C=512,N=N,cap=48,repeats=2,rotation=i,trace=True,timeout_s=5400))
    elif a.stage=='audit':
        for gpu,model in enumerate(['ouro','huginn']):
            jobs.append(dict(id=f'batched_numerical_{model}_P512_N4_s0',kind='audit',model=model,gpu=gpu,
                P=512,C=256,N=4,cap=16,inject_after=3,timeout_s=3600))
    elif a.stage=='no_decode':
        for i,(model,P) in enumerate([(m,p) for m in ['ouro','huginn'] for p in [256,2048]]):
            jobs.append(dict(id=f'no_decode_{model}_P{P}_s0',kind='screen',model=model,gpu=i,
                P=P,C=512,N=0,cap=16,repeats=3,trace=True,rotation=i,timeout_s=3600))
    elif a.stage=='confirm':
        for model in ['ouro','huginn']:
            for s in range(3):
                for n in [1,4]:
                    i=len(jobs)
                    jobs.append(dict(id=f'quiet_confirm_{model}_P2048_N{n}_s{s}',kind='screen',
                        model=model,gpu=(i*3+s)%8,P=2048,C=512,N=n,cap=64,repeats=2,
                        rotation=s,trace=False,timeout_s=5400,
                        arms=[['mixed',1],['hybrid',2],['hybrid',4],['group',2],['group',4]]))
        write(ROOT/'plans'/f'{a.stage}.json',dict(gpus=list(range(8)),
            groups=[dict(jobs=[j]) for j in jobs]));return
    else:
        groups=[]
        for width in [1,2,4,8]:
            group=[]
            for gpu in range(width):
                model='ouro'
                group.append(dict(id=f'contention_w{width}_gpu{gpu}_s0',kind='screen',model=model,
                    gpu=gpu,P=256,C=128,N=2,cap=32,repeats=3,rotation=0,trace=False,
                    arms=[['mixed',1]],barrier=f'w{width}_s0',timeout_s=1800))
            groups.append(dict(jobs=group,barrier=f'w{width}_s0'))
        write(ROOT/'plans'/f'{a.stage}.json',dict(gpus=list(range(8)),groups=groups));return
    write(ROOT/'plans'/f'{a.stage}.json',dict(gpus=list(range(8)),jobs=jobs))

if __name__=='__main__':main()
