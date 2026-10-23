"""Register remaining legal G frontier cases before adaptive execution."""
from .common import ROOT,write

def main():
    target=ROOT/'p1_dynamic_frontier_plan.json';assert not target.exists()
    jobs=[dict(id=f'cb_p1_dynamic_gpu{gpu}_v1',kind='numerical',gpu=gpu,lengths=[512,2048],
        modes=['atomic','cycled'],timeout_s=1800) for gpu in range(8)]
    write(target,dict(gpus=list(range(8)),jobs=jobs,
        scope='Pretrained G1-to-G3 and G1-to-G2-to-G1 continuations. Immutable token spans, all R4.'))

if __name__=='__main__':
    main()
