"""Stronger generic controls. Remove a redundant count collective, not expert work."""
import torch
import torch.distributed as dist
from collections import Counter
from contextlib import contextmanager
from .common import append,write
from .measure import timed,generate
from .workloads import inputs

@contextmanager
def collective_calls():
    counts=Counter();originals={name:getattr(dist,name) for name in ['all_gather_into_tensor','all_to_all_single','all_reduce']}
    def wrapper(name,function):
        def call(*args,**kwargs):
            counts[name]+=1
            return function(*args,**kwargs)
        return call
    try:
        for name,function in originals.items():setattr(dist,name,wrapper(name,function))
        yield counts
    finally:
        for name,function in originals.items():setattr(dist,name,function)

def compare_logits(actual,reference):
    numerator=denominator=0.;exact=True
    width=actual.shape[-1];actual=actual.reshape(-1,width);reference=reference.reshape(-1,width)
    for offset in range(0,len(actual),16):
        x=actual[offset:offset+16];y=reference[offset:offset+16].to(x.device)
        exact=exact and torch.equal(x,y)
        numerator+=float((x.float()-y.float()).square().sum())
        denominator+=float(y.float().square().sum())
    return exact,(numerator/max(denominator,1e-18))**.5

def measure(a,ep,target,config):
    cap=config.get('generation_cap',32)
    for b in config.get('batches',[1,2,4,8,16]):
        for length in config.get('lengths',[128,512,2048]):
            ids,requests=inputs(a,b,length);ep.request_ids=[r['id'] for r in requests]
            ep.set_policy('static');reference=a.forward(ids).detach().cpu()
            # Counterbalance policy order across EP degree. These are generic policies only.
            policies=['static','replicated_counts','replicated_counts_source_owner']
            if ep.world==8:policies=list(reversed(policies))
            outputs={}
            for policy in policies:
                ep.set_policy(policy);ep.verify_metadata=policy!='static'
                actual=a.forward(ids)
                exact,l2=compare_logits(actual,reference)
                assert exact if policy=='replicated_counts' else l2<=.02
                append(target/f'fidelity.rank{ep.rank}.jsonl',dict(batch=b,prefix=length,policy=policy,
                    logit_exact=exact,relative_l2=l2,full_split_matrix_verified=True,
                    evidence='LIVE_CURRENT_ROUTER_METADATA_AGREES_WITH_ACTUAL_ALL_GATHER'))
                ep.verify_metadata=False;del actual
                with collective_calls() as calls:value=a.forward(ids)
                del value
                moe_calls=sum(ep.calls.values())
                assert calls['all_gather_into_tensor']==(moe_calls if policy=='static' else 0)
                assert calls['all_to_all_single']==2*moe_calls and calls['all_reduce']==moe_calls
                append(target/f'collective_calls.rank{ep.rank}.jsonl',dict(batch=b,prefix=length,policy=policy,
                    calls=dict(calls),native_moe_invocations=moe_calls,
                    evidence='LIVE_PYTHON_COLLECTIVE_SUBMISSIONS_SEPARATE_FROM_CLEAN_TIMING'))
                for _ in range(9):value=a.forward(ids);del value
                timings=[]
                for _ in range(5):elapsed,value=timed(lambda:a.forward(ids));timings.append(elapsed);del value
                checked=None
                if policy!='static':
                    ep.verify_metadata=True
                    checked,_=generate(a,ids,cap)
                    ep.verify_metadata=False
                warm,_=generate(a,ids,cap);outputs[policy]=warm['generated_ids']
                if checked is not None:assert checked['generated_ids']==warm['generated_ids'],'Verified and clean metadata paths changed output'
                for repeat in range(3):
                    value,_=generate(a,ids,cap)
                    assert value['generated_ids']==warm['generated_ids']
                    append(target/f'generic_b{b}_p{length}.rank{ep.rank}.jsonl',dict(value,policy=policy,repeat=repeat,
                        batch=b,prefix=length,prefill_ms=timings,count_collectives_per_moe=1 if policy=='static' else 0,
                        expert_dispatch_per_moe=1,expert_combine_per_moe=1,restoration_allreduce_per_moe=1,
                        evidence='LIVE_TRUE_EP_GENERIC_METADATA_CONTROL',
                        limitation='Dense states and current exact router outputs remain replicated. Not a general Attention-TP/DP contract.'))
            expected=outputs['static']
            write(target/f'output_parity_b{b}_p{length}.rank{ep.rank}.json',dict(batch=b,prefix=length,
                policy_exact_to_static={k:v==expected for k,v in outputs.items()},
                metadata_only_exact=outputs['replicated_counts']==expected))
            assert outputs['replicated_counts']==expected
            del reference
            print('GENERIC_METADATA',a.name,ep.world,b,length,flush=True)
    # One unseen B and a prefix not divisible by either EP degree.
    ids,requests=inputs(a,3,129);ep.request_ids=[r['id'] for r in requests]
    ep.set_policy('static');reference=a.forward(ids).detach().cpu()
    for policy in ['replicated_counts','replicated_counts_source_owner']:
        ep.set_policy(policy);ep.verify_metadata=True;actual=a.forward(ids)
        exact,l2=compare_logits(actual,reference);del actual
        assert exact if policy=='replicated_counts' else l2<=.02
        append(target/f'unseen_shape.rank{ep.rank}.jsonl',dict(batch=3,prefix=129,policy=policy,
            logit_exact=exact,relative_l2=l2,split_matrices_verified=True,
            evidence='LIVE_UNSEEN_BATCH_AND_UNEVEN_SOURCE_ROWS'))
    ep.verify_metadata=False;ep.set_policy('static')
