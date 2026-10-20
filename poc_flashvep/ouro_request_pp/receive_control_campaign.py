"""Check whether per-handoff destination allocation explains the arm ranking."""
from .campaign import execute
from .contract import ROOT,write


def main():
    execute('correctness_receive_pool',['--mode','correctness','--graphs','--preallocated-recv',
        '--correctness-prompt','128','--correctness-count','4','--correctness-depths','4',
        '--runtime-label','receive_pool'])
    cases=[]
    for repeat in range(3):
        for R,C in [(4,1),(4,4),(4,16),(2,16)]:
            cases.append(dict(depth=R,C=C,P=128,O=32,repeat=repeat))
    plan=ROOT/'receive_pool_plan.json';write(plan,dict(cases=cases))
    for arm in ['layer_pp4','loop_pp4']:
        name=f'receive_pool_{arm}'
        execute(name,['--mode','benchmark','--placement',arm,'--label',name,
            '--case-file',str(plan),'--graphs','--preallocated-recv'])
    execute('correctness_receive_pool_rope',['--mode','correctness','--graphs','--preallocated-recv','--precompute-rotary',
        '--correctness-prompt','128','--correctness-count','4','--correctness-depths','4',
        '--runtime-label','receive_pool_rope'])
    plan=ROOT/'receive_pool_rope_plan.json'
    write(plan,dict(cases=[dict(depth=4,C=C,P=128,O=32,repeat=repeat)
                          for repeat in range(3) for C in [4,16]]))
    for arm in ['loop_pp4','layer_pp4']:
        name=f'receive_pool_rope_{arm}'
        execute(name,['--mode','benchmark','--placement',arm,'--label',name,
            '--case-file',str(plan),'--graphs','--preallocated-recv','--precompute-rotary'])


if __name__=='__main__':main()
