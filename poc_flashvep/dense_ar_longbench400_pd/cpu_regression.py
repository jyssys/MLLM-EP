"""Exercise the actual row-aware forward seam with deterministic CPU operators."""
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch
from .adapter import Adapter
from .geometry import Partition,tests
from .common import ROOT,write


class Layer:
    def __init__(self,number):self.number=number
    def __call__(self,positions,hidden,residual):return hidden+self.number+1,hidden.clone()


def main():
    tests()
    from vllm.forward_context import override_forward_context
    for hybrid in (False,True):
        me=Adapter.__new__(Adapter);me.role='decode';me.cfg={};me.rows=[];me.states={}
        part=Partition(24,4,8 if hybrid else 0)
        start=12 if hybrid else 0;count=8
        batch=SimpleNamespace(req_ids=['partial-x','decode-y'],num_tokens=count+1,num_reqs=2,
            query_start_loc_np=np.array([0,count,count+1]),num_computed_tokens_np=np.array([start,128]),
            is_prefilling_np=np.array([True,False]),num_tokens_after_padding=count+1)
        me.specs={'partial':part,'decode':Partition(128,4,0)}
        me.imports={'partial-x':'source'}
        incoming=torch.arange(part.prefix,dtype=torch.float32).unsqueeze(1)+sum(range(1,part.lower+1))
        me.transport=SimpleNamespace(get=lambda *_:(incoming,torch.zeros_like(incoming)),retire=lambda *_:None)
        me.active=dict(batch=batch,runner=None)
        def metadata(spans):
            flat=[j for _,left,right in spans for j in range(left,right)]
            return torch.tensor(flat,dtype=torch.long),None
        me.metadata=metadata
        # override_forward_context accepts None and restores the caller's prior context.
        module=SimpleNamespace(embed_input_ids=lambda ids:ids.to(torch.float32).unsqueeze(1),
            layers=[Layer(i) for i in range(36)],norm=lambda h,r:(h,r))
        ids=torch.tensor(list(range(start,start+count))+[128]);positions=ids.clone()
        actual=me.partial_forward(module,ids,positions)
        expected=ids.to(torch.float32).unsqueeze(1)+sum(range(1,37))
        assert torch.equal(actual,expected),(hybrid,actual,expected)
        coverage={}
        for row in me.rows:
            for token in range(row['start'],row['start']+row['count']):
                for layer in row['layers']:
                    key=(row['request_id'],token,layer);coverage[key]=coverage.get(key,0)+1
        for token in range(start,start+count):
            for layer in range(36):
                wanted=int(token>=part.prefix or layer>=part.lower)
                assert coverage.get(('partial',token,layer),0)==wanted
        assert all(coverage.get(('decode',128,layer))==1 for layer in range(36))
    result=dict(passed=True,scope='Actual Adapter.partial_forward and active_spans execute on CPU surrogate operators. '
        'This catches row ownership and merge errors. It does not qualify GPU attention,KV transfer or model numerics.')
    write(ROOT/'analysis/C3_row_aware_CPU_regression.json',result);print(result)


if __name__=='__main__':main()
