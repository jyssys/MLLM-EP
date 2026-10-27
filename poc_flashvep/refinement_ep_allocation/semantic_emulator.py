"""Phase-2 intervention. Full expert kernels still run. This is not acceleration."""
import math
import torch

def select_edges(weights,masked,prior,kind,fraction,rank,layer,forward_index):
    """Select a fixed number of current edges. Preserve one deterministic slot per row."""
    assert weights.ndim==2 and weights.shape[1]==8
    assert kind in ('P1','P2','P3') and 0<=fraction<=.25
    n=weights.shape[0]
    count=min(int(math.floor(weights.numel()*fraction)),n*7)
    removed=torch.zeros_like(weights,dtype=torch.bool)
    if not count:
        return removed
    row=torch.arange(n,device=weights.device,dtype=torch.long)
    slots=torch.arange(8,device=weights.device,dtype=torch.long)
    protected=(row+rank+layer+forward_index)%8
    if kind=='P1':
        generator=torch.Generator(device=weights.device)
        generator.manual_seed(20261028+layer*49979687+forward_index*67867967+rank*86028121)
        score=torch.rand(weights.shape,device=weights.device,dtype=torch.float32,generator=generator)
    else:
        score=weights.float()/weights.float().sum(-1,keepdim=True).clamp_min(1e-12)
        if kind=='P3':
            available=torch.isfinite(prior)
            c=torch.nan_to_num(prior,nan=0.)
            frontier=torch.sigmoid((c-.95+.25)/.05)*torch.sigmoid((.95-c)/.05)
            frontier=torch.where(available,frontier,torch.zeros_like(frontier))
            risk=torch.where(masked,1.+frontier,torch.full_like(frontier,.75))
            score=score*risk[:,None]
    score=score.clone()
    score[row,protected]=float('inf')
    selected=torch.topk(score.flatten(),count,largest=False,sorted=False).indices
    removed[selected//8,selected%8]=True
    return removed

class SemanticEmulator:
    def __init__(self,kind,fraction,rank):
        self.kind=kind
        self.fraction=fraction
        self.rank=rank
        self.active=False
        self.history=torch.full((8192,),float('nan'),device=f'cuda:{rank}',dtype=torch.float32)
        self.context=None
        self.reset(False)

    def reset(self,active):
        self.active=active
        self.history.fill_(float('nan'))
        self.forward_index=0
        self.completed_decodes=0
        self.edges=0
        self.zeroed=0
        self.calls=0
        self.keepalive=[]
        self.weight_strides=set()
        self.actual_zeroed=torch.zeros((),device=f'cuda:{self.rank}',dtype=torch.int64)
        self.mass=torch.zeros(2,device=f'cuda:{self.rank}',dtype=torch.float32)

    def install(self,model,dllm):
        original_forward=model.forward
        def forward(*args,**kwargs):
            if self.active:
                token_ids=args[0] if args else kwargs['input_ids']
                self.context=dict(ids=token_ids,positions=kwargs['position_ids'])
                self.forward_index+=1
            return original_forward(*args,**kwargs)
        model.forward=forward
        original_decode=dllm.decoder.decode
        def decode(logits,block_start,block_end,x,*args,**kwargs):
            result=original_decode(logits,block_start,block_end,x,*args,**kwargs)
            if self.active:
                assert logits.shape[-1]==157184
                confidence=torch.softmax(logits.float(),dim=-1).amax(-1).flatten()
                assert confidence.numel()==block_end-block_start
                self.history[block_start:block_end].copy_(confidence)
                self.completed_decodes+=1
            return result
        dllm.decoder.decode=decode
        for layer in model.model.layers:
            block=layer.mlp
            if not hasattr(block,'deepep_dispatcher'):
                continue
            original=block.deepep_dispatcher.dispatch
            def dispatch(*args,_original=original,_layer=block.layer_id,**kwargs):
                if not self.active:
                    return _original(*args,**kwargs)
                ids=kwargs['topk_idx']
                weights=kwargs['topk_weights']
                n=ids.shape[0]
                start=self.rank*(self.context['ids'].numel()//4)
                tokens=self.context['ids'].flatten()[start:start+n]
                positions=self.context['positions'].flatten()[start:start+n].long()
                assert len(tokens)==n and len(positions)==n
                assert torch.isfinite(weights).all().item(),'Non-finite current router weights'
                mask=select_edges(weights,tokens==156895,self.history.index_select(0,positions),
                    self.kind,self.fraction,self.rank,_layer,self.forward_index)
                self.edges+=weights.numel()
                self.zeroed+=min(int(math.floor(weights.numel()*self.fraction)),n*7)
                self.actual_zeroed.add_(mask.sum())
                self.weight_strides.add(tuple(weights.stride()))
                self.calls+=1
                self.mass[0].add_(weights.float().sum())
                self.mass[1].add_(weights.float().masked_fill(~mask,0.).sum())
                modified_weights=weights.masked_fill(mask,0.)
                # DeepEP can consume inputs on a communication stream. Keep these
                # buffers until the request's existing end synchronization completes.
                self.keepalive.append(modified_weights)
                updated=dict(kwargs,topk_weights=modified_weights)
                # Global expert IDs stay unchanged. All dispatch and expert work remains.
                return _original(*args,**updated)
            block.deepep_dispatcher.dispatch=dispatch

    def summary(self):
        mass=self.mass.detach().cpu().tolist()
        actual=int(self.actual_zeroed)
        assert actual==self.zeroed,'Semantic mask count does not match its recorded budget'
        return dict(kind=self.kind,requested_fraction=self.fraction,
            diagnostic='DENSE_SEMANTIC_EMULATION_NOT_SPARSE_ACCELERATION',
            full_routed_edges_executed=self.edges,semantically_zeroed_edges=self.zeroed,
            semantic_edge_fraction=self.zeroed/self.edges if self.edges else 0.,
            omitted_original_weight_mass_fraction=mass[1]/mass[0] if mass[0] else 0.,
            actual_dispatch_edge_reduction=0.,actual_expert_kernel_work_reduction=0.,
            current_router_ids_unchanged=True,retained_weights_renormalized=False,
            minimum_retained_k=1,dispatch_calls=self.calls,completed_decode_observations=self.completed_decodes,
            actual_zeroed_mask_count=actual,current_weight_strides=sorted(self.weight_strides),
            refinement_rule='P3 uses current mask state and prior completed-forward confidence. Missing history falls back to router scores.',
            reference_method={'P1':'UNIFORM_CONTROL','P2':'CURRENT_ROUTER_ONLY_CONTROL',
                'P3':'REFLEX_INSPIRED_PROXY_NOT_PAPER_IMPLEMENTATION'}[self.kind])
