"""Run a separate heavy lane after clean experiments finish."""
import argparse
import os
import subprocess
import time
from .contract import ROOT,PROJECT,write,sha

PYTHON='/home/esjung/.venvs/llada2-mini-reference/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--depth',type=int,default=4);p.add_argument('--concurrency',type=int,default=4)
    args=p.parse_args()
    directory=ROOT/'profiles';directory.mkdir(exist_ok=True)
    for arm in ['layer_pp4','loop_pp4']:
        name=f'{arm}_R{args.depth}_C{args.concurrency}_v5'
        target=directory/name
        assert not target.with_suffix('.nsys-rep').exists()
        cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
             '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
             '--force-overwrite=false','-o',str(target),PYTHON,'-u','-m','poc_flashvep.ouro_request_pp.benchmark',
             '--mode','benchmark','--placement',arm,'--label','nsys_'+name,'--depth',str(args.depth),
             '--concurrency',str(args.concurrency),'--prompt-len','128','--output-len','8','--graphs','--trace','--nsys']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                                         HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        write(ROOT/'commands'/f'profile_{name}.json',dict(argv=cmd,cwd=str(PROJECT),heavy_lane=True))
        begin=time.monotonic()
        with (directory/f'{name}.log').open('w') as log:
            code=subprocess.run(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        write(ROOT/'commands'/f'profile_{name}_completion.json',dict(returncode=code,wall_s=time.monotonic()-begin))
        assert code==0, f'Profiler failed: {name}'
        export=['nsys','export','--type=sqlite','--output',str(target.with_suffix('.sqlite')),str(target.with_suffix('.nsys-rep'))]
        subprocess.run(export,check=True,cwd=PROJECT)
        print('PROFILE_CAPTURED',name,flush=True)


if __name__=='__main__':main()
