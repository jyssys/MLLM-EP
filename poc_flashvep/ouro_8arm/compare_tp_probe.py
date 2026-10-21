"""Reassemble true TP4 head ownership before a bounded KV comparison."""
import torch
from .contract import *
from .numerical_trace import errors


def compare(reference,candidate):
    torch.set_num_threads(1)
    gold=torch.load(ROOT/'.tensor_cache'/f'{reference}_rank0.pt',weights_only=False,map_location='cpu')
    ranks=[torch.load(ROOT/'.tensor_cache'/f'{candidate}_rank{rank}.pt',weights_only=False,map_location='cpu') for rank in range(4)]
    result=[]
    for rid,slots in gold.items():
        a=read(ROOT/'raw'/reference/(rid+'.json'));b=read(ROOT/'raw'/candidate/(rid+'.json'))
        assert a['prompt_ids']==b['prompt_ids']
        worst=0.
        for slot,kv in slots.items():
            for which,x in enumerate(kv):
                y=torch.cat([rank[rid][slot][which] for rank in ranks],dim=1)
                assert y.shape==x.shape and torch.isfinite(y).all(),(rid,slot,y.shape,x.shape)
                worst=max(worst,errors(y,x)['relative_L2'])
        result.append(dict(id=rid,reference=reference,candidate=candidate,token_exact=a['output_ids']==b['output_ids'],
            depth_exact=a['depths']==b['depths'],max_KV_slot_relative_L2=worst,KV_slots=len(slots),
            TP_local_head_count=4,assembled_head_count=16,
            scope='Bounded sharded KV comparison. Every rank separately checks bitwise current-row last_exited copies. Historical full-model TP qualification remains unresolved.'))
    write(ROOT/'numerical'/f'{candidate}_vs_SD.json',result)
    print('TP_HEAD_OWNERSHIP_PROBE',candidate,len(result),'requests;',sum(r['token_exact'] for r in result),'token matches',flush=True)
    return result


if __name__=='__main__':
    import sys
    compare(*sys.argv[1:])
