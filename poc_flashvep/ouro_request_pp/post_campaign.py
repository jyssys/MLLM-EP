"""Run bounded validation and observer controls after the primary campaign."""
import argparse
from .campaign import execute
from .contract import ROOT,write


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['validation','controls']);a=p.parse_args()
    if a.phase=='validation':
        execute('correctness_P2048_C1_v5',['--mode','correctness','--graphs',
            '--correctness-prompt','2048','--correctness-count','1','--correctness-depths','4'])
        for arm in ['layer_pp4','loop_pp4']:
            execute(f'eos_{arm}_v5',['--mode','benchmark','--placement',arm,'--label',f'eos_{arm}_v5',
                '--depth','4','--concurrency','4','--prompt-len','128','--output-len','64','--graphs','--stop-eos'])
    else:
        cases=[dict(depth=4,C=4,P=128,O=8,repeat=i) for i in range(3)]
        plan=ROOT/'observer_plan_v5.json';write(plan,dict(cases=cases))
        for arm in ['layer_pp4','loop_pp4']:
            for trace in [False,True]:
                name=f'observer_{arm}_{"trace" if trace else "clean"}_v5'
                execute(name,['--mode','benchmark','--placement',arm,'--label',name,
                    '--case-file',str(plan),'--graphs']+(['--trace'] if trace else []))
        # Short fixed-depth traces expose unused late Loop-PP devices.
        plan=ROOT/'depth_trace_plan_v5.json'
        write(plan,dict(cases=[dict(depth=d,C=8,P=128,O=8,repeat=0) for d in [2,3]]))
        for arm in ['layer_pp4','loop_pp4']:
            name=f'depth_trace_{arm}_v5'
            execute(name,['--mode','benchmark','--placement',arm,'--label',name,
                '--case-file',str(plan),'--graphs','--trace'])


if __name__=='__main__':main()
