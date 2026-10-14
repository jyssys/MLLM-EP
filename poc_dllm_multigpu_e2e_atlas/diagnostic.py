"""One source-correlated eager forward. Never a headline timing lane."""
import contextlib
import functools
import torch
from poc_dllm_multigpu_e2e_atlas.contract import *


class Labels:
    def __init__(self,model):self.model=model;self.patches=[];self.records=[];self.layer=None;self.seen=set()

    @contextlib.contextmanager
    def label(self,kind):
        torch.cuda.nvtx.range_push(f'DIAG|layer={self.layer}|{kind}')
        try:yield
        finally:torch.cuda.nvtx.range_pop()

    def patch(self,obj,name,kind):
        if (id(obj),name) in self.seen:return
        self.seen.add((id(obj),name));original=getattr(obj,name)
        @functools.wraps(original)
        def call(*args,**kwargs):
            with self.label(kind):return original(*args,**kwargs)
        self.patches.append((obj,name,original));setattr(obj,name,call)

    def __enter__(self):
        for layer in self.model.model.layers:
            fn=layer.forward;lid=layer.layer_id
            def forward(*a,_fn=fn,_lid=lid,**kw):
                self.layer=_lid
                try:
                    with self.label('LAYER'):return _fn(*a,**kw)
                finally:self.layer=None
            self.patches.append((layer,'forward',fn));layer.forward=forward
            attn=layer.attention
            for obj,name,kind in [(attn.query_key_value,'forward','QKV'),(attn,'_apply_qk_norm','QK_NORM'),
                (attn.rotary_emb,'forward','ROPE'),(attn,'_apply_repeat','KV_REPEAT'),(attn.dense,'forward','OUTPUT_PROJECTION'),
                (layer.mlp,'forward','MLP_MOE'),(layer.layer_communicator,'prepare_attn','NORM_PREPARE_ATTN'),
                (layer.layer_communicator,'prepare_mlp','ATTN_TP_AND_NORM'),
                (layer.layer_communicator,'postprocess_layer','MLP_TP_AND_RESIDUAL')]:self.patch(obj,name,kind)
            if hasattr(layer.mlp,'gate'):self.patch(layer.mlp.gate,'forward','ROUTER')
            if hasattr(layer.mlp,'experts'):self.patch(layer.mlp.experts,'forward','ROUTED_EXPERTS')
        import torch.nn.functional as F
        self.patch(F,'scaled_dot_product_attention','SDPA')
        original=torch.Tensor.slice_scatter
        def scatter(tensor,src,*a,_fn=original,**kw):
            with self.label('KV_SLICE_SCATTER'):
                result=_fn(tensor,src,*a,**kw)
                self.records.append(dict(layer=self.layer,operation='slice_scatter',
                    destination_shape=list(tensor.shape),destination_strides=list(tensor.stride()),
                    destination_storage=tensor.untyped_storage().data_ptr(),source_shape=list(src.shape),
                    source_strides=list(src.stride()),source_storage=src.untyped_storage().data_ptr(),
                    output_storage=result.untyped_storage().data_ptr(),full_tensor_bytes=tensor.numel()*tensor.element_size(),
                    source_bytes=src.numel()*src.element_size(),args=list(a),kwargs=kw))
                return result
        self.patches.append((torch.Tensor,'slice_scatter',original));torch.Tensor.slice_scatter=scatter
        self.patch(self.model.lm_head,'forward','LM_HEAD')
        from sglang.srt.distributed.parallel_state import GroupCoordinator
        for name in ['all_reduce','all_gather','reduce_scatter','broadcast']:
            if hasattr(GroupCoordinator,name):self.patch(GroupCoordinator,name,'TP_'+name.upper())
        return self

    def __exit__(self,*unused):
        for obj,name,fn in reversed(self.patches):setattr(obj,name,fn)


class EagerProbe:
    def __init__(self,runner,model,output,rank):
        self.runner=runner;self.model=model;self.output=Path(output);self.rank=rank;self.done=False
        self.labels=None;self.pending=None

    def __enter__(self):
        self.original=self.runner.forward
        def forward(*args,**kwargs):
            result=self.original(*args,**kwargs)
            kv=kwargs.get('past_key_values')
            if not self.done and kv is not None and kwargs.get('attention_mask') is None:
                self.done=True
                if hasattr(kv,'_data'):kv=kv._data
                self.labels=Labels(self.model)
                torch.cuda.nvtx.range_push('DIAGNOSTIC_EXTRA_EAGER_FORWARD_NOT_NATIVE_REQUEST_WORK')
                try:
                    with self.labels:
                        diagnostic=self.runner.forward_normal(*args,**dict(kwargs,past_key_values=kv))
                    self.pending=((result.logits.float()-diagnostic.logits.float()).abs().amax(),
                        (result.logits.argmax(-1)==diagnostic.logits.argmax(-1)).all())
                finally:torch.cuda.nvtx.range_pop()
            return result
        self.runner.forward=forward
        return self

    def __exit__(self,*unused):self.runner.forward=self.original

    def finish(self):
        write(self.output/f'eager_source_diagnostic.rank{self.rank}.json',dict(executed=self.done,
            scope='ONE EXTRA EAGER FORWARD. Source correlation only. Never subtract or call its wall native serving latency.',
            tensor_operations=[] if self.labels is None else self.labels.records,
            logit_max_abs=None if self.pending is None else float(self.pending[0]),
            greedy_logit_parity=None if self.pending is None else bool(self.pending[1])))
