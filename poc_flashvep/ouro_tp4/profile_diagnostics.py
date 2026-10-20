"""Physical timeline evidence only. No clean serving claim after the C2 failure."""
import subprocess
import argparse
from .contract import ROOT,PYTHON
from .launch import execute


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='')
    parser.add_argument('--only',choices=['all','TP'],default='all');args=parser.parse_args()
    target=ROOT/'profiles';target.mkdir(exist_ok=True)
    for arm in ([] if args.only=='TP' else ['layer_pp4','loop_pp4']):
        name=args.prefix+'mp_correctness_'+arm
        argv=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none',
            '-o',str(target/name),PYTHON,'-u','-m','poc_flashvep.ouro_tp4.mp_pp','--arm',arm,
            '--label',name,'--correctness','--trace','--nsys']
        execute(name,argv,timeout=300)
        subprocess.run(['nsys','export','--type=sqlite','--output',str(target/f'{name}.sqlite'),str(target/f'{name}.nsys-rep')],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    name=args.prefix+'tp_operator_diagnostic'
    execute(name,['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
        '--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none','-o',str(target/name),
        PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',
        '-m','poc_flashvep.ouro_tp4.tp_driver','--mode','operators','--nsys','--label',name],timeout=300)
    subprocess.run(['nsys','export','--type=sqlite','--output',str(target/f'{name}.sqlite'),str(target/f'{name}.nsys-rep')],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)


if __name__=='__main__':main()
