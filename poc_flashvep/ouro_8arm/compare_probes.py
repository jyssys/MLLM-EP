"""Compare real model outputs and every owner-local KV value for bounded probes."""
import torch
from .contract import *
from .numerical_trace import errors

def compare(reference,candidate):
    torch.set_num_threads(1)
    base=ROOT/'raw'/reference;other=ROOT/'raw'/candidate;checks=[]
    gold=torch.load(ROOT/'.tensor_cache'/f'{reference}_rank0.pt',map_location='cpu',weights_only=False)
    observed={}
    for file in (ROOT/'.tensor_cache').glob(candidate+'_rank*.pt'):
        caches=torch.load(file,map_location='cpu',weights_only=False)
        for rid,slots in caches.items():
            observed.setdefault(rid,{}).update(slots)
    for rid,slots in gold.items():
        a=read(base/(rid+'.json'));b=read(other/(rid+'.json'))
        KV_bitwise=all(torch.equal(x,y) for index,kv in slots.items() for x,y in zip(kv,observed[rid][index]))
        worst=max(errors(y,x)['relative_L2'] for index,kv in slots.items() for x,y in zip(kv,observed[rid][index]))
        checks.append(dict(id=rid,reference=reference,candidate=candidate,token_exact=a['output_ids']==b['output_ids'],
            depths_exact=a['depths']==b['depths'],KV_slots=len(observed[rid]),KV_bitwise=KV_bitwise,KV_max_slot_L2=worst))
    write(ROOT/'numerical'/f'probe_{candidate}.json',checks)
    assert all(r['token_exact'] and r['depths_exact'] and r['KV_bitwise'] and r['KV_slots']==96 for r in checks),checks
    print('PROBE_PASS',reference,candidate,len(checks),flush=True)
    return checks

if __name__=='__main__':
    import sys
    compare(*sys.argv[1:])
