"""CPU formula checks. They do not replace actual multi-rank model validation."""
import ast
import copy
from types import SimpleNamespace
import torch
from .ep import ExpertParallel
from .generic_controls import compare_logits
from .common import ROOT,write
from ..looped_moe_ep.runtime import assign_name

def main():
    tree=copy.deepcopy(ExpertParallel._replicated_run._ep_observer_ast)
    statements=[node for node in tree.body[0].body if assign_name(node) in ['sources','matrix_tensor','matrix']]
    assert len(statements)==3
    probe=ast.parse('def probe(self,ids,e,k,b,l):\n    pass')
    probe.body[0].body=statements+[ast.parse('return matrix').body[0]]
    scope=dict(torch=torch)
    exec(compile(ast.fix_missing_locations(probe),'<actual-generated-count-formula-cpu-check>','exec'),scope)
    torch.manual_seed(20261027);torch.set_num_threads(1);cases=0
    for world in [4,8]:
        for b in [1,2,3,4,8,16]:
            for length in [128,129,512,2048]:
                for e in [8,25,64]:
                    first=torch.randint(e,(b,length));second=(first+torch.randint(1,e,(b,length)))%e
                    ids=torch.stack([first,second],dim=-1)
                    for policy in ['static','source_owner']:
                        obj=SimpleNamespace(world=world,source_policy=policy)
                        actual=scope['probe'](obj,ids,e,2,b,length)
                        if policy=='static':sources=torch.arange(b*length)%world
                        else:sources=ids.reshape(-1,2)[:,0]%world
                        expected=[torch.bincount(ids.reshape(-1,2)[sources==rank].flatten(),minlength=e).tolist() for rank in range(world)]
                        assert actual==expected,(world,b,length,e,policy)
                        cases+=1
    logits=torch.randn(2,128,25,dtype=torch.bfloat16)
    assert compare_logits(logits,logits.clone())==(True,0.)
    changed=logits.clone();changed[0,0,0]+=1
    exact,error=compare_logits(logits,changed);assert not exact and error>0
    result=dict(status='PASS',count_formula_cases=cases,logit_comparison_cases=2,
        evidence='CPU_GENERATED_FORMULA_AND_BOUNDED_COMPARISON_UNIT_CHECK',
        limitation='No native checkpoint, GPU execution, dispatch latency, or downstream accuracy is measured here.')
    write(ROOT/'quality/generic_metadata_cpu_check.json',result);print(result)

if __name__=='__main__':main()
