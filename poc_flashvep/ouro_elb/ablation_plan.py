"""Same-GPU execution-mode and graph ablations, not new backends."""
from .common import ROOT,write

def main():
    jobs=[]
    for start,gpu in enumerate((0,4,7)):
        modes=[(True,True),(False,True),(True,False),(False,False)]
        modes=modes[start:]+modes[:start]
        for asynchronous,graphs in modes:
            jobs.append(dict(id=f'p2_ablation_v1_g{gpu}_a{int(asynchronous)}_q{int(graphs)}',gpu=gpu,kind='timing',
                shapes=[[256,1],[2048,4]],groups=[1,2,4],graphs=graphs,**{'async':asynchronous},
                cap=32,repeats=3,lane=f'ABLATION_ASYNC{int(asynchronous)}_GRAPH{int(graphs)}',timeout_s=3600))
    write(ROOT/'ablation_plan.json',dict(gpus=[0,4,7],jobs=jobs))

if __name__=='__main__':main()
