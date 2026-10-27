"""Counterbalanced source-policy checks. Each process start keeps its own results."""
from .common import append
from .workloads import inputs
from .measure import timed,generate

def verify(a,ep,target,config):
    ids,requests=inputs(a,1,128)
    ep.request_ids=[r['id'] for r in requests]
    order=(['source_owner','static','static','source_owner'] if config['start']==1 else
        ['static','source_owner','source_owner','static'])
    reference=None
    for block,policy in enumerate(order):
        ep.source_policy=policy
        for _ in range(10):out=a.forward(ids);del out
        times=[]
        for _ in range(20):duration,out=timed(lambda:a.forward(ids));times.append(duration);del out
        warm,_=generate(a,ids,32)
        if reference is None:reference=warm['generated_ids']
        assert warm['generated_ids']==reference,'Source policy changed the observed greedy output'
        for repeat in range(3):
            result,_=generate(a,ids,32)
            assert result['generated_ids']==reference
            append(target/f'paired_verify.rank{ep.rank}.jsonl',dict(result,model=a.name,ep=ep.world,
                block=block,policy=policy,repeat=repeat,start=config['start'],prefill_ms=times,
                order=order,evidence='LIVE_COUNTERBALANCED_WARMED_GENERATION'))
        print('VERIFIED',a.name,ep.world,config['start'],block,policy,flush=True)
    ep.source_policy='static'
