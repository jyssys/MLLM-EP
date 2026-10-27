"""Observe authoritative router outputs. Never change model tensors or scores."""
from collections import Counter
import numpy as np
import torch

class RouterTrace:
    def __init__(self,adapter):
        self.adapter=adapter;self.handles=[];self.records=[];self.calls=Counter()
        self.request_ids=[];self.domain=[];self.phase='prefill';self.step=0
        self.frontier_only=False
        self.handles.append(adapter.model.register_forward_pre_hook(self.begin))
        for site,block in adapter.sites.items():
            self.handles.append(block.router.register_forward_hook(self.hook(site)))

    def begin(self,module,inputs):
        self.calls.clear()

    def hook(self,site):
        def collect(module,inputs,output):
            if self.adapter.name.startswith('ifm'):
                scores,ids=output[:2]
            else:
                scores,ids=output[2:4]
                scores=scores.to(inputs[0].dtype)
            b,length,k=ids.shape
            assert k==2
            loop=self.calls[site];self.calls[site]+=1
            # The seed does not depend on a loop or a route. Token identities match across passes.
            rng=np.random.default_rng(20261026+length)
            positions=np.sort(rng.choice(length,min(length,64),replace=False)).tolist()
            if self.frontier_only:positions=[length-1]
            if self.phase=='decode' and length-1 not in positions:positions.append(length-1)
            selected=ids[:,positions].detach().cpu().tolist()
            weights=scores[:,positions].detach().float().cpu().tolist()
            for request in range(b):
                self.records.append(dict(site=site,loop=loop,recurrent=site in self.adapter.recurrent,
                    request_id=self.request_ids[request] if self.request_ids else str(request),
                    domain=self.domain[request] if self.domain else 'unknown',batch=b,batch_row=request,
                    context=length,phase=self.phase,step=self.step,positions=positions,
                    top2=selected[request],weights=weights[request],
                    roles=['new_token_frontier' if self.phase=='decode' and p==length-1 else
                        'prefix_recompute' if self.phase=='decode' else 'prefill' for p in positions],
                    evidence='LIVE_AUTHORITATIVE_ROUTER_OUTPUT_SAMPLE'))
        return collect

    def close(self):
        for handle in self.handles:handle.remove()

