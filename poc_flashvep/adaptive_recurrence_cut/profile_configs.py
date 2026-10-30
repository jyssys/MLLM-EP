"""Small profiler-only runs. Their latency is not clean serving latency."""
import json
from .common import ROOT,write


def main():
    plans=[]
    for model,depth in (('ouro',4),('huginn',32)):
        cfg=json.loads((ROOT/'configs'/f'gate_{model}_L512_cut{depth}.json').read_text())
        for cut in (depth,depth//2):
            name=f'profile_{model}_L512_cut{cut}'
            value=dict(cfg,name=name,cut=cut,count=2,cap=4,verify_transfer=False,state_probe=False,
                       profile=True,arrival_offsets=[0.,0.],gpus=[0,1],warmup=True,
                       regime='profiler_only',timeout=900)
            dest=ROOT/'configs'/f'{name}.json'
            if dest.exists():raise FileExistsError(dest)
            write(dest,value);plans.append(dest.name)
    write(ROOT/'configs/profiling_plan.json',dict(configs=plans,clean_latency_use=False,scope='N2 kernel and host-path diagnosis only.'))
    print(' '.join(plans))


if __name__=='__main__':main()
