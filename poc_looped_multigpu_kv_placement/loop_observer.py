"""Source-labelled eager diagnostic. Never substitute these times for clean graphs."""
import torch


class LoopObserver:
    def __init__(self,model):
        self.model=model;self.active=None;self.patches=[];self.records=[];self.wrapped=set()
        self.events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(18000)]
        self.origin=torch.cuda.Event(enable_timing=True)

    def wrap(self,obj,name,category):
        # Native RoPE modules can be shared by all physical layers.
        # One hook uses the current layer context. Repeated hooks multiply observations.
        key=(id(obj),name)
        if key in self.wrapped:return
        self.wrapped.add(key)
        original=getattr(obj,name)
        def forward(*args,**kwargs):
            if self.active is None:return original(*args,**kwargs)
            assert len(self.records)<len(self.events)
            a,b=self.events[len(self.records)];self.records.append((self.active,category,a,b))
            a.record()
            try:return original(*args,**kwargs)
            finally:b.record()
        self.patches.append((obj,name,original));setattr(obj,name,forward)

    def __enter__(self):
        self.origin.record()
        for layer_id,layer in enumerate(self.model.model.layers):
            original=layer.forward
            def forward(*args,_fn=original,_lid=layer_id,**kwargs):
                ut=kwargs.get('current_ut',args[2] if len(args)>2 else None)
                pos=kwargs.get('positions',args[0] if args else None)
                self.active=(int(ut),_lid,int(pos.numel()))
                try:return _fn(*args,**kwargs)
                finally:self.active=None
            self.patches.append((layer,'forward',original));layer.forward=forward
            self.wrap(layer.self_attn.qkv_proj,'forward','QKV_PROJECTION')
            self.wrap(layer.self_attn.rotary_emb,'forward','ROPE')
            self.wrap(layer.self_attn.o_proj,'forward','O_PROJECTION_INCLUDES_REDUCE')
            self.wrap(layer.mlp,'forward','MLP_INCLUDES_REDUCE')
            for module in layer.self_attn.attn:self.wrap(module,'forward','ATTENTION_AND_KV_FUSED')
        import vllm.model_executor.layers.linear as linear
        self.wrap(linear,'tensor_model_parallel_all_reduce','TP_ALL_REDUCE_API_INTERVAL_NESTED')
        return self

    def __exit__(self,*unused):
        for obj,name,fn in reversed(self.patches):setattr(obj,name,fn)

    def finish(self):
        torch.cuda.synchronize()
        return [dict(loop=context[0],physical_layer=context[1],query_rows=context[2],category=cat,
            start_ms=self.origin.elapsed_time(a),end_ms=self.origin.elapsed_time(b),duration_ms=a.elapsed_time(b),
            validity='SOURCE_LABELLED_EAGER_DIAGNOSTIC_ONLY. GPU event intervals can include host gaps. Nested all-reduce is not additive.')
            for context,cat,a,b in self.records]
