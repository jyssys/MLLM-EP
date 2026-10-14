"""External CUDA-event nodes in native graphs. No math/layout replacement.

Events retain ONLY the most recent replay for each graph key. Analysis must
not pretend these are timestamps for all earlier denoising iterations.
"""
import torch


class GraphEvents:
    def __init__(self,detail=False):
        self.key=None;self.layer=None;self.records={};self.patches=[]
        self.detail=detail
        self.patched=set()

    def patch(self,obj,name,fn):
        if (id(obj),name) in self.patched:return
        self.patched.add((id(obj),name))
        self.patches.append((obj,name,getattr(obj,name)));setattr(obj,name,fn)

    def wrap(self,fn,stage,lid=None,metadata=None):
        def call(*args,**kwargs):
            if self.key is None or not torch.cuda.is_current_stream_capturing():return fn(*args,**kwargs)
            a=torch.cuda.Event(enable_timing=True,external=True)
            b=torch.cuda.Event(enable_timing=True,external=True)
            a.record();out=fn(*args,**kwargs);b.record()
            meta=dict(stage=stage,layer=self.layer if lid is None else lid)
            if metadata:meta.update(metadata(args,kwargs))
            self.records.setdefault(self.key,[]).append((meta,a,b))
            return out
        return call

    def install(self,model):
        from dinfer.decoding.diffusion_runner import CudaGraphRunner
        original=CudaGraphRunner.capture_one_batch_size
        def capture(runner,bs,forward,is_decode_phase=True,length=0,cache_length=0,use_mask=False):
            self.key=(bs,is_decode_phase,length,cache_length)
            try:return original(runner,bs,forward,is_decode_phase,length,cache_length,use_mask)
            finally:self.key=None
        self.patch(CudaGraphRunner,'capture_one_batch_size',capture)
        for layer in model.model.layers:
            lid=layer.layer_id;fn=layer.forward
            def layer_call(*args,_fn=fn,_lid=lid,**kwargs):
                self.layer=_lid
                try:return _fn(*args,**kwargs)
                finally:self.layer=None
            self.patch(layer,'forward',layer_call)
            a=layer.attention
            self.patch(a,'forward',self.wrap(a.forward,'attention_path',lid))
            if not self.detail:continue
            for obj,name,label in [(a.query_key_value,'forward','qkv'),(a.dense,'forward','output_projection'),
                (a,'_apply_repeat','KV_repeat'),(a.rotary_emb,'forward','RoPE'),(a,'_apply_qk_norm','qk_norm')]:
                self.patch(obj,name,self.wrap(getattr(obj,name),label))
            self.patch(layer.mlp,'forward',self.wrap(layer.mlp.forward,'MLP_MoE',lid))
        if not self.detail:return
        import torch.nn.functional as F
        self.patch(F,'scaled_dot_product_attention',self.wrap(F.scaled_dot_product_attention,'SDPA',
            metadata=lambda a,k:dict(q_shape=list(a[0].shape),key_shape=list(a[1].shape),
                KV_read_bytes_proxy=(a[1].numel()+a[2].numel())*a[1].element_size(),
                mask_none=k.get('attn_mask') is None)))
        self.patch(torch.Tensor,'slice_scatter',self.wrap(torch.Tensor.slice_scatter,'KV_slice_scatter'))
        from sglang.srt.distributed.parallel_state import GroupCoordinator
        self.patch(GroupCoordinator,'all_reduce',self.wrap(GroupCoordinator.all_reduce,'TP_all_reduce',
            metadata=lambda a,k:dict(input_payload_bytes=a[1].numel()*a[1].element_size())))
        self.patch(GroupCoordinator,'all_gather',self.wrap(GroupCoordinator.all_gather,'TP_all_gather',
            metadata=lambda a,k:dict(input_payload_bytes=a[1].numel()*a[1].element_size())))

    def restore(self):
        for obj,name,fn in reversed(self.patches):setattr(obj,name,fn)

    def read(self,keys):
        result=[]
        for key,count in keys.items():
            if key not in self.records:continue
            rows=[dict(**meta,ms=a.elapsed_time(b)) for meta,a,b in self.records[key]]
            result.append(dict(key=list(key),actual_native_calls=count,
                scope='LAST_REPLAY_FOR_THIS_GRAPH_KEY_ONLY',operators=rows))
        return result
