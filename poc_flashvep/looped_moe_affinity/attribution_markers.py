"""CPU markers label launches. They do not replace GPU timestamps or alter tensors."""
import ast
import copy
from contextlib import contextmanager
import inspect
import os
import textwrap
import time
import types

class Markers:
    def __init__(self,gpu_events=False):
        self.records=[];self.handles=[];self.step=0;self.enabled=False;self.pid=os.getpid()
        self.gpu_events=gpu_events;self.event_records=[];self.origin=None;self.endpoint=None
        self.count_records=[]
        self.depth=0

    def record_counts(self,site,loop,b,length,width,matrix,policy,recurrent,count_policy='collective'):
        # The exact CPU count matrix already exists in the original EP expression.
        # Retain its immutable list. Do not add a GPU readback or routing operation.
        self.count_records.append(dict(step=self.step,site=site,loop=loop,batch=b,length=length,width=width,
            source_counts=matrix,source_policy=policy,recurrent=recurrent,count_policy=count_policy))

    def event(self):
        if not self.gpu_events:return None
        import torch
        event=torch.cuda.Event(enable_timing=True);event.record();return event

    @contextmanager
    def region(self,label,site=''):
        if not self.enabled:
            yield
            return
        start=time.time_ns()
        event=self.event()
        depth=self.depth;self.depth+=1
        try:yield
        finally:
            self.depth-=1
            finish=self.event();self.records.append((start,time.time_ns(),label,site,self.step,depth))
            if event is not None:self.event_records.append((event,finish,label,site,self.step,depth))

    def hook(self,module,label,site):
        stack=[]
        def begin(mod,args):
            if self.enabled:
                stack.append((time.time_ns(),self.event(),self.depth));self.depth+=1
        def end(mod,args,value):
            if self.enabled:
                start,event,depth=stack.pop();self.depth-=1;finish=self.event()
                self.records.append((start,time.time_ns(),label,site,self.step,depth))
                if event is not None:self.event_records.append((event,finish,label,site,self.step,depth))
        self.handles.extend([module.register_forward_pre_hook(begin),module.register_forward_hook(end)])

    def install_modules(self,a,detail='fine'):
        router_ids={id(block.router) for block in a.sites.values()}
        expert_ids={id(block.experts) for block in a.sites.values() if hasattr(block,'experts')}
        for site,module in a.model.named_modules():
            kind=type(module).__name__.lower()
            label=None
            if id(module) in router_ids:label='router'
            elif id(module) in expert_ids:label='expert_compute'
            elif 'attention' in kind and 'rotary' not in kind:label='attention_qkv_rope'
            elif site in a.sites:label='moe_block_other'
            elif site in ['lm_head','output','output.output'] or site.endswith('.lm_head'):label='lm_head'
            elif 'embedding' in kind:label='embedding'
            elif kind in ['rmsnorm','layernorm']:label='normalization'
            if detail=='coarse' and label in ['normalization','moe_block_other','embedding','lm_head']:label=None
            if label:self.hook(module,label,site)

    def close(self):
        for h in self.handles:h.remove()

    def resolve_events(self):
        import torch
        torch.cuda.synchronize()
        assert self.depth==0,'Observer nesting must close before attribution.'
        boundaries=[end for start,end,label,site,step,depth in self.event_records if label=='sampling_output' and step==0]
        assert len(boundaries)==1,'The complete rollout requires one first-token boundary.'
        return dict(window_ms=self.origin.elapsed_time(self.endpoint),first_output_complete_ms=self.origin.elapsed_time(boundaries[0]),intervals=[
            (self.origin.elapsed_time(start),self.origin.elapsed_time(end),label,site,step,depth)
            for start,end,label,site,step,depth in self.event_records])

def instrument_ep(ep,markers,detail='fine'):
    """Wrap coarse lexical regions of the unchanged TRUE EP expression."""
    from ..looped_moe_ep import runtime as original
    replicated=getattr(ep,'count_policy','collective')=='replicated'
    if replicated:
        method=ep._replicated_run
        tree=copy.deepcopy(getattr(method,'__func__',method)._ep_observer_ast)
    else:tree=ast.parse(textwrap.dedent(inspect.getsource(original.PhysicalEP._run)))
    class Sources(ast.NodeTransformer):
        def visit_Subscript(self,node):
            node=self.generic_visit(node)
            if isinstance(node.slice,ast.Slice) and ast.unparse(node.slice)=='self.rank::self.world':
                node.slice=ast.Name(id='source_rows',ctx=ast.Load())
            return node
    tree=Sources().visit(tree);body=tree.body[0].body
    flat=next(i for i,n in enumerate(body) if original.assign_name(n)=='flat')
    if not replicated:
        body[flat+1:flat+1]=ast.parse("source_rows = slice(self.rank, None, self.world) if self.source_policy == 'static' else self.source_rows(ids)").body
    matrix=next(i for i,n in enumerate(body) if original.assign_name(n)=='matrix')
    body[matrix+1:matrix+1]=ast.parse("self.attribution.record_counts(site, loop, b, l, d, matrix, self.source_policy, site in self.a.recurrent, self.count_policy)").body
    def call_name(node):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):return ast.unparse(node.value.func)
        return ''
    gather=next((i for i,n in enumerate(body) if call_name(n)=='dist.all_gather_into_tensor'),None)
    aa=[i for i,n in enumerate(body) if call_name(n)=='dist.all_to_all_single']
    compute=next(i for i,n in enumerate(body) if original.assign_name(n)=='computed')
    restore=next(i for i,n in enumerate(body) if call_name(n)=='dist.all_reduce')
    metadata=([(0,gather,'metadata_pack'),(gather,gather+1,'count_allgather'),(gather+1,aa[0],'metadata_pack')]
        if gather is not None else [(0,aa[0],'metadata_pack')])
    cuts=metadata+[(aa[0],aa[0]+1,'dispatch'),
        (aa[0]+1,compute,'receive_layout'),(compute,compute+1,'expert_compute'),
        (compute+1,aa[1],'return_layout'),(aa[1],aa[1]+1,'combine'),
        (aa[1]+1,restore,'weighting_layout'),(restore,restore+1,'restore_allreduce'),
        (restore+1,len(body),'other_ep')]
    if detail=='coarse':
        cuts=[(0,aa[0],'metadata_and_count_sync'),(aa[0],aa[0]+1,'dispatch'),
            (aa[0]+1,aa[1],'expert_and_local_layout'),(aa[1],aa[1]+1,'combine'),
            (aa[1]+1,len(body),'reweight_and_restore')]
    wrapped=[]
    for first,last,label in cuts:
        context=ast.parse(f"with self.attribution.region('{label}', site):\n    pass").body[0]
        context.body=body[first:last];wrapped.append(context)
    tree.body[0].body=wrapped
    scope=dict(vars(original))
    exec(compile(ast.fix_missing_locations(tree),'<timing-only-EP-host-regions>','exec'),scope)
    old=ep._run;ep.attribution=markers;ep._run=types.MethodType(scope['_run'],ep)
    return old

def instrument_native_foil(a,markers):
    """Add one lexical marker around the official inline expert expression."""
    from ..looped_moe_ep.runtime import assign_name
    patched={}
    for site,block in a.sites.items():
        block._attribution=markers;block._attribution_site=site
        cls=type(block)
        if cls in patched:continue
        original=cls.forward;tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
        body=tree.body[0].body
        first=next(i for i,n in enumerate(body) if assign_name(n)=='x_flat')
        last=next(i for i,n in enumerate(body) if assign_name(n)=='experts_out')
        context=ast.parse("with self._attribution.region('expert_native_including_layout', self._attribution_site):\n    pass").body[0]
        context.body=body[first:last+1];body[first:last+1]=[context]
        scope=dict(vars(__import__('sys').modules[cls.__module__]))
        exec(compile(ast.fix_missing_locations(tree),'<timing-only-native-Foil-expert-region>','exec'),scope)
        cls.forward=scope['forward'];patched[cls]=original
    return patched
