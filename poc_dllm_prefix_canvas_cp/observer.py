"""Deferred observations. Clean lane has no operator-level hooks/events."""
import functools
import inspect
import torch


class Observer:
    def __init__(self,model,dllm,rank,mode,prefix,runner=None):
        self.model=model;self.dllm=dllm;self.rank=rank;self.mode=mode;self.prefix=prefix
        self.patches=[];self.events=[];self.phases=[];self.updates=[];self.calls=[]
        self.call=-1;self.layer=None;self.iteration=None;self.snapshots=[];self.first_update=None
        self.runner=runner;self.graph_keys={}

    @staticmethod
    def event():
        event=torch.cuda.Event(enable_timing=True);event.record();return event

    def patch(self,obj,name,value):
        self.patches.append((obj,name,getattr(obj,name)));setattr(obj,name,value)

    def timed(self,fn,name,layer=None):
        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            a=self.event();out=fn(*args,**kwargs);b=self.event()
            self.events.append((dict(call=self.call,layer=self.layer if layer is None else layer,stage=name),a,b))
            return out
        return wrapped

    def __enter__(self):
        self.origin=self.event()
        # These two phase boundaries and retained native token-update outputs
        # are the only clean-lane observations. No per-layer hooks are installed.
        for name in ['prefill','decode']:
            original=getattr(self.dllm.block_runner,name)
            def phase(*args,_fn=original,_name=name,**kwargs):
                a=self.event();out=_fn(*args,**kwargs);b=self.event()
                self.phases.append((dict(stage=_name),a,b));return out
            self.patch(self.dllm.block_runner,name,phase)
        from dinfer.decoding.utils import TokenArray
        original=TokenArray.__setitem__
        def update(x,idx,values,_original=original):
            result=_original(x,idx,values)
            # values is the newly allocated native torch.where output. Native
            # code never mutates it after assignment. Retain, without a clone.
            if isinstance(idx,tuple) and isinstance(idx[1],slice) and isinstance(values,torch.Tensor):
                self.updates.append((idx[1].start,idx[1].stop,values))
                if self.first_update is None:self.first_update=self.event()
            return result
        self.patch(TokenArray,'__setitem__',update)
        if self.mode=='clean':return self
        iteration=self.dllm.diff_iteration.forward
        def iteration_forward(*args,**kwargs):
            block=args[6];loc=args[5]
            self.iteration=dict(block=int(block),start=loc.start,end=loc.end,
                cross=bool(kwargs.get('is_cross_block',False)))
            try:return iteration(*args,**kwargs)
            finally:self.iteration=None
        self.patch(self.dllm.diff_iteration,'forward',iteration_forward)
        target=self.runner or self.model
        forward=target.forward
        def model_forward(*args,**kwargs):
            self.call+=1
            ids=kwargs.get('input_ids',args[0] if args else None);cache=kwargs.get('past_key_values')
            if cache is not None and hasattr(cache,'_data'):cache=cache._data
            c=dict(call=self.call,query_rows=ids.shape[1],cache_keys=0 if cache is None else cache.shape[4],
                phase='prefill' if self.iteration is None else ('commit_cross' if self.iteration['cross'] else 'refine'),
                iteration=None if self.iteration is None else self.iteration.copy())
            key=(ids.shape[0],cache is not None,ids.shape[1],0 if cache is None else cache.shape[4])
            self.graph_keys[key]=self.graph_keys.get(key,0)+1
            selected=(self.mode=='trace' and cache is not None and self.iteration and
                not self.iteration['cross'] and self.iteration['block']==0)
            captures=[]
            if selected:
                for lid in [0,9,19]:
                    if sum(s['layer']==lid for s in self.snapshots)>=3:continue
                    old=cache[lid]
                    captures.append(dict(layer=lid,call=self.call,cache_shape=list(old[0].shape),
                        cache_storage=old[0].untyped_storage().data_ptr(),cache_version=old[0]._version,
                        mask_none=kwargs.get('attention_mask') is None,
                        positions=kwargs['position_ids'].detach().clone(),
                        prefix=[v[:,:,:self.prefix].detach().clone() for v in old]))
            a=self.event();out=forward(*args,**kwargs);b=self.event()
            self.calls.append((c,a,b))
            # Native graph outputs are safe on the consumer stream after replay.
            for snapshot in captures:
                lid=snapshot['layer'];present=out.past_key_values
                values=present[lid] if isinstance(present,torch.Tensor) else present[2*lid:2*lid+2]
                snapshot['returned_prefix']=[v[:,:,:self.prefix].detach().clone() for v in values]
                snapshot['returned_canvas']=[v[:,:,-32:].detach().clone() for v in values]
                self.snapshots.append(snapshot)
            return out
        self.patch(target,'forward',model_forward)
        if self.runner is not None:
            graph=self.runner.graph_runner
            self.patch(graph,'replay_prepare',self.timed(graph.replay_prepare,'graph_input_KV_prepare'))
        if self.runner is not None and self.mode=='coarse':
            # Operator events already live inside the marked native graphs.
            return self
        for layer in self.model.model.layers:
            lid=layer.layer_id;attn=layer.attention
            if self.mode=='collectives':
                fn=layer.forward
                def layer_call(*args,_fn=fn,_lid=lid,**kwargs):
                    self.layer=_lid
                    try:return _fn(*args,**kwargs)
                    finally:self.layer=None
                self.patch(layer,'forward',layer_call)
                continue
            fn=attn.forward
            def attention(*args,_fn=fn,_lid=lid,**kwargs):
                self.layer=_lid
                snapshot=None
                if self.mode=='trace' and self.iteration and not self.iteration['cross'] and self.iteration['block']==0 and _lid in [0,9,19]:
                    old=kwargs.get('past_key_values')
                    if old is not None and sum(s['layer']==_lid for s in self.snapshots)<3:
                        snapshot=dict(layer=_lid,call=self.call,cache_shape=list(old[0].shape),
                            cache_storage=old[0].untyped_storage().data_ptr(),cache_version=old[0]._version,
                            mask_none=kwargs.get('attention_mask') is None,
                            positions=kwargs['positions'].detach().clone(),
                            prefix=[v[:,:,:self.prefix].detach().clone() for v in old])
                        self.snapshots.append(snapshot)
                a=self.event();result=_fn(*args,**kwargs);b=self.event()
                self.events.append((dict(call=self.call,layer=_lid,stage='attention_path'),a,b))
                if snapshot is not None:
                    snapshot['returned_prefix']=[v[:,:,:self.prefix].detach().clone() for v in result[1]]
                    snapshot['returned_canvas']=[v[:,:,-32:].detach().clone() for v in result[1]]
                self.layer=None;return result
            self.patch(attn,'forward',attention)
            if self.mode=='trace':
                for obj,name,label in [(attn.query_key_value,'forward','qkv'),(attn.dense,'forward','attn_output_projection'),
                    (attn.rotary_emb,'forward','RoPE'),(attn,'_apply_qk_norm','qk_norm'),(attn,'_apply_repeat','KV_repeat')]:
                    self.patch(obj,name,self.timed(getattr(obj,name),label,lid))
        if self.mode=='trace':
            import torch.nn.functional as F
            fn=F.scaled_dot_product_attention
            def sdpa(q,k,v,*args,_fn=fn,**kwargs):
                if self.layer is None:return _fn(q,k,v,*args,**kwargs)
                meta=dict(call=self.call,layer=self.layer,stage='SDPA',q_shape=list(q.shape),k_shape=list(k.shape),
                    dtype=str(q.dtype),mask_none=kwargs.get('attn_mask') is None,is_causal=kwargs.get('is_causal',False),
                    physical_KV_read_bytes_proxy=(k.numel()+v.numel())*k.element_size())
                a=self.event();result=_fn(q,k,v,*args,**kwargs);b=self.event();self.events.append((meta,a,b));return result
            self.patch(F,'scaled_dot_product_attention',sdpa)
            fn=torch.Tensor.slice_scatter
            def scatter(tensor,*args,_fn=fn,**kwargs):
                if self.layer is None:return _fn(tensor,*args,**kwargs)
                return self.timed(_fn,'KV_slice_scatter')(tensor,*args,**kwargs)
            self.patch(torch.Tensor,'slice_scatter',scatter)
        if self.mode=='collectives':
            from sglang.srt.distributed.parallel_state import GroupCoordinator
            for name in ['all_reduce','all_gather','reduce_scatter']:
                if not hasattr(GroupCoordinator,name):continue
                original=getattr(GroupCoordinator,name)
                def collective(group,tensor,*args,_fn=original,_name=name,**kwargs):
                    a=self.event();result=_fn(group,tensor,*args,**kwargs);b=self.event()
                    self.events.append((dict(call=self.call,layer=self.layer,stage='TP_'+_name,
                        input_payload_bytes=tensor.numel()*tensor.element_size()),a,b));return result
                self.patch(GroupCoordinator,name,collective)
        return self

    def __exit__(self,*exc):
        for obj,name,fn in reversed(self.patches):setattr(obj,name,fn)

    def finish(self):
        # Call only after final request completion. No event resolutions or
        # host-device transfers occur inside any measured native operator.
        def record(entries):return [dict(**meta,start_ms=self.origin.elapsed_time(a),
            end_ms=self.origin.elapsed_time(b),ms=a.elapsed_time(b)) for meta,a,b in entries]
        updates=[]
        for start,end,values in self.updates:
            updates.append(dict(start=start,end=end,ids=values.cpu().tolist()))
        captures=[];previous={}
        for snap in self.snapshots:
            lid=snap['layer'];old=previous.get(lid)
            captures.append(dict(layer=lid,call=snap['call'],cache_shape=snap['cache_shape'],
                cache_storage=snap['cache_storage'],cache_version=snap['cache_version'],mask_none=snap['mask_none'],
                positions=snap['positions'].cpu().tolist(),
                prefix_copy_equal=[torch.equal(a,b) for a,b in zip(snap['prefix'],snap['returned_prefix'])],
                prefix_equal_previous=None if old is None else [torch.equal(a,b) for a,b in zip(old['prefix'],snap['prefix'])],
                canvas_equal_previous=None if old is None else [torch.equal(a,b) for a,b in zip(old['returned_canvas'],snap['returned_canvas'])]))
            previous[lid]=snap
        return dict(events=record(self.events),calls=record(self.calls),phases=record(self.phases),
            native_updates=updates,KV_captures=captures,
            first_token_update_ms=None if self.first_update is None else self.origin.elapsed_time(self.first_update),
            clock='rank-local CUDA stream; do not subtract timestamps across ranks')
