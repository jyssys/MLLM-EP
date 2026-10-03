"""Deferred, bounded traces around the already validated physical DeepEP bridge."""
import functools
import json
import types
import numpy as np
import torch

class Recorder:
    def __init__(self,rank):
        self.rank=rank;self.mode='bypass';self.calls=[];self.stages=[];self.routes=[]
        self.acceptances=[];self.call=-1;self.context=None;self.iteration=None

    def begin(self,mode,request_id,prompt_length):
        self.mode=mode;self.request_id=request_id;self.prompt_length=prompt_length
        self.calls=[];self.stages=[];self.routes=[];self.acceptances=[];self.iteration=None;self.final_state=None

    @staticmethod
    def event():
        e=torch.cuda.Event(enable_timing=True);e.record();return e

    def wrapped_stage(self,layer,stage,fn,wait=False):
        @functools.wraps(fn)
        def call(*args,**kwargs):
            if self.mode not in ('trace','timing'):return fn(*args,**kwargs)
            a=self.event();result=fn(*args,**kwargs)
            # Native bridge waits on this event immediately after _combine_core.
            # Move the same dependency before timing-end, so it measures completion.
            if wait:result[1].current_stream_wait()
            b=self.event();self.stages.append((self.call,layer,stage,a,b))
            if stage=='dispatch' and self.mode=='trace':
                ids=kwargs['topk_idx'];weights=kwargs['topk_weights'];hidden=kwargs['hidden_states']
                assert hidden.dtype==torch.bfloat16 and hidden.shape[-1]==2048
                start=self.rank*(self.context['positions'].numel()//4)
                positions=self.context['positions'].flatten()[start:start+ids.shape[0]]
                before=self.context['input_ids'].flatten()[start:start+ids.shape[0]]==156895
                self.routes.append(dict(call_id=self.call,layer=layer,positions=positions.detach().clone(),
                    ids=ids.detach().clone(),weights=weights.detach().clone(),before=before.detach().clone(),
                    recv_ids=result[1].detach().clone(),recv_rows=result[0].shape[0],
                    hidden_row_bytes=hidden.shape[-1]*hidden.element_size()))
            return result
        return call

    def install(self,model,dllm):
        original=model.forward
        def forward(_model,input_ids=None,*args,**kwargs):
            if self.mode not in ('trace','timing'):return original(input_ids,*args,**kwargs)
            self.call=len(self.calls)
            self.context=dict(call_id=self.call,phase='denoise' if self.iteration is not None else 'prefill',
                iteration=self.iteration,positions=kwargs['position_ids'],input_ids=input_ids)
            a=self.event();out=original(input_ids,*args,**kwargs);b=self.event()
            context={k:v for k,v in self.context.items() if k not in ('input_ids','positions')}
            self.calls.append((context,a,b))
            return out
        model.forward=types.MethodType(forward,model)
        for layer in model.model.layers:
            block=layer.mlp
            if not hasattr(block,'deepep_dispatcher'):continue
            dispatcher=block.deepep_dispatcher
            dispatcher.dispatch=self.wrapped_stage(block.layer_id,'dispatch',dispatcher.dispatch)
            dispatcher._normal_dispatcher._combine_core=self.wrapped_stage(block.layer_id,'combine',dispatcher._normal_dispatcher._combine_core,True)
            block.experts.quant_method.apply=self.wrapped_stage(block.layer_id,'expert',block.experts.quant_method.apply)
            block.gate.forward=self.wrapped_stage(block.layer_id,'router',block.gate.forward)
        iteration=dllm.diff_iteration;original_iteration=iteration.forward
        def iteration_forward(_self,model_arg,decoder,x,kv_cache,block,block_loc,block_id,*args,**kwargs):
            if self.mode not in ('trace','timing','acceptance'):
                return original_iteration(model_arg,decoder,x,kv_cache,block,block_loc,block_id,*args,**kwargs)
            self.iteration=dict(index=len(self.acceptances),native_iter=_self.iter_no,block=block_id,
                start=block_loc.start,end=block_loc.end,is_cross_block=bool(kwargs.get('is_cross_block',False)))
            first_call=len(self.calls);before=x.data.detach().clone() if self.mode in ('trace','acceptance') else None
            result=original_iteration(model_arg,decoder,x,kv_cache,block,block_loc,block_id,*args,**kwargs)
            after=x.data.detach().clone() if before is not None else None
            self.acceptances.append(dict(**self.iteration,call_ids=list(range(first_call,len(self.calls))),before=before,after=after))
            if self.mode=='trace':
                for r in self.routes:
                    if r['call_id']>=first_call and 'after' not in r:
                        r['after']=(after.flatten().index_select(0,r['positions'].flatten())==156895).clone()
            self.iteration=None
            return result
        iteration.forward=types.MethodType(iteration_forward,iteration)
        from dinfer.decoding.utils import TokenArray
        original_return=TokenArray.get_generated_tokens
        def returned(x):
            if self.mode in ('trace','acceptance'):self.final_state=x.data.detach().clone()
            return original_return(x)
        TokenArray.get_generated_tokens=returned

    def finish(self,outdir,mode,token_ids):
        torch.cuda.synchronize(self.rank)
        events=[dict(**c,forward_ms=float(a.elapsed_time(b))) for c,a,b in self.calls]
        stages=[dict(call_id=c,layer=l,stage=s,ms=float(a.elapsed_time(b))) for c,l,s,a,b in self.stages]
        denoise=[(a,b) for c,a,b in self.calls if c['phase']=='denoise']
        denoise_span_ms=float(denoise[0][0].elapsed_time(denoise[-1][1])) if denoise else None
        def cpu(v):return v.detach().cpu().numpy()
        arrays={};route_meta=[]
        for n,r in enumerate(self.routes):
            tag=f'r{n}'
            before=cpu(r['before']);after=cpu(r.get('after',r['before']))
            for key,value in [('ids',cpu(r['ids']).astype(np.uint16)),('weights',cpu(r['weights'].float())),
                ('positions',cpu(r['positions']).astype(np.uint16)),('before',before),('after',after),
                ('recv_ids',cpu(r['recv_ids']).astype(np.int16))]:arrays[tag+'_'+key]=value
            route_meta.append(dict(tag=tag,**{k:r[k] for k in ('call_id','layer','recv_rows','hidden_row_bytes')}))
        acceptance=[]
        for a in self.acceptances:
            row={k:v for k,v in a.items() if k not in ('before','after')}
            if a['before'] is not None:
                before=cpu(a['before']).flatten();after=cpu(a['after']).flatten()
                live=(before==156895);new=live&(after!=156895)
                assert not np.any((before!=156895)&(after==156895)),'native monotone threshold acceptance required'
                row.update(live_positions=np.flatnonzero(live).tolist(),new_positions=np.flatnonzero(new).tolist(),
                    still_live_positions=np.flatnonzero(after==156895).tolist(),accepted_ids=after[new].tolist())
            acceptance.append(row)
        from poc_dllm_ep_comm.contract import write
        payload=dict(rank=self.rank,request_id=self.request_id,mode=mode,prompt_length=self.prompt_length,
            events=events,stages=stages,denoise_span_ms=denoise_span_ms,routes=route_meta,acceptances=acceptance,output_ids=token_ids,
            final_state=None if self.final_state is None else cpu(self.final_state).flatten().tolist())
        write(outdir/f'{mode}.rank{self.rank}.json',payload)
        if arrays:np.savez_compressed(outdir/f'{mode}.rank{self.rank}.npz',**arrays)
        self.mode='bypass'
