"""Prospective calibration, exact layout controls, and heavy profile plans."""
from .common import ROOT,write as atomic_write
import json

def write(path,value):
    if path.exists():
        assert json.loads(path.read_text())==value,('IMMUTABLE_PLAN_CHANGED',path)
        return
    atomic_write(path,value)

def main():
    groups=[]
    for repeat in range(2):
        for width in ((1,2,4,8) if repeat==0 else (8,4,2,1)):
            barrier=f'cal_w{width}_r{repeat}'
            groups.append(dict(barrier=barrier,jobs=[dict(id=f'{barrier}_g{g}',gpu=g,
                kind='timing',P=8192,N=4,cap=64,repeats=2,split='VALIDATION',barrier=barrier)
                for g in range(width)]))
    write(ROOT/'plans'/'calibration.json',dict(gpus=list(range(8)),groups=groups))
    jobs=[]
    for gpu,p in enumerate((2048,8192,32768,65408)):
        jobs.append(dict(id=f'fixture_P{p}',gpu=gpu,kind='capture',P=p,N=1,cap=2,
            graphs=False,split='FINAL' if p==65408 else 'TRAIN'))
    for gpu,storage in ((4,'native'),(5,'layer_major')):
        jobs.append(dict(id=f'correct_{storage}',gpu=gpu,kind='capture',P=2048,N=1,cap=2,
            graphs=False,split='VALIDATION',fixture_namespace=f'{storage}_',
            **(dict(layout=storage) if storage!='native' else {})))
    write(ROOT/'plans'/'capture.json',dict(gpus=list(range(8)),jobs=jobs))
    jobs=[]
    for restart in range(3):
        for p,n in ((2048,1),(8192,4),(32768,1),(65408,1)):
            gpu=(restart+p//2048)%8
            for storage in (('native','layer_major') if restart%2==0 else ('layer_major','native')):
                jobs.append(dict(id=f'layout_{storage}_P{p}_N{n}_s{restart}',gpu=gpu,kind='timing',
                    P=p,N=n,cap=64,repeats=3,split='VALIDATION' if restart==0 else 'FINAL',
                    **(dict(layout=storage) if storage!='native' else {})))
    # Size-aware pool control at identical valid payload. This saves empty reserve only.
    for gpu,blocks in ((6,18432),(7,4608)):
        jobs.append(dict(id=f'pool_{blocks}_P8192_N1',gpu=gpu,kind='timing',P=8192,N=1,
            cap=64,repeats=3,blocks=blocks,split='FINAL'))
    write(ROOT/'plans'/'layout.json',dict(gpus=list(range(8)),jobs=jobs))
    jobs=[]
    for p,n in ((2048,1),(8192,4),(32768,1),(65408,1)):
        jobs.append(dict(id=f'profile_P{p}_N{n}',gpu=0,kind='timing',P=p,N=n,cap=16,
            repeats=1,split='VALIDATION',nsys=True))
    write(ROOT/'plans'/'profiles.json',dict(gpus=list(range(8)),groups=[dict(jobs=[j]) for j in jobs]))
    layout_jobs=__import__('json').loads((ROOT/'plans'/'layout.json').read_text())['jobs']
    quiet=[]
    for j in layout_jobs:
        if j['id'].startswith('layout') and j['P'] in (2048,32768):
            quiet.append(dict(j,id=j['id']+'_quiet',sample_memory=False,cap=32))
    write(ROOT/'plans'/'quiet_layout.json',dict(gpus=list(range(8)),groups=[dict(jobs=[j]) for j in quiet]))
    controls=[dict(id='tiering_exact',gpu=1,kind='tiering',fixture='P32768_d1_l12.pt')]
    controls += [dict(id=f'observer_mem_{flag}',gpu=0,kind='timing',P=8192,N=4,cap=64,
        repeats=3,split='FINAL',sample_memory=flag) for flag in (False,True)]
    controls += [dict(id='capacity_admission_v1',gpu=0,kind='capacity',P=1024,N=1),
        dict(id='fixture_P16384',gpu=0,kind='capture',P=16384,N=1,cap=2,graphs=False,split='FINAL'),
        dict(id='profile_P16384_N1',gpu=0,kind='timing',P=16384,N=1,cap=16,repeats=1,split='VALIDATION',nsys=True)]
    controls += [dict(id=f'natural_EOS_P{p}_N{n}',gpu=0,kind='timing',P=p,N=n,cap=128,
        repeats=1,split='FINAL',natural=True,sample_memory=False) for p,n in ((2048,1),(8192,4))]
    write(ROOT/'plans'/'controls_extended.json',dict(gpus=list(range(8)),groups=[dict(jobs=[j]) for j in controls]))
    write(ROOT/'plans'/'local_segmentation.json',dict(gpus=list(range(8)),groups=[dict(jobs=[
        dict(id=f'local_segmentation_s{s}',gpu=s,kind='local_split',repeats=20)]) for s in range(3)]))

if __name__=='__main__':main()
