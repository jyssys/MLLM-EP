"""Compile fixed D layer intervals with original weights and exact metadata."""
import copy
import hashlib
from pathlib import Path
import time
import torch
from vllm.compilation.decorators import support_torch_compile

class _Stage(torch.nn.Module):
    def __init__(self,layers):
        super().__init__();self.layers=torch.nn.ModuleList(layers)
    def run(self,positions,hidden,residual):
        for layer in self.layers:hidden,residual=layer(positions,hidden,residual)
        return hidden,residual

@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DLower34(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DLower32(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DLower28(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DLower24(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DUpper34(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DUpper32(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DUpper28(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class DUpper24(_Stage):
    def forward(self,positions,hidden,residual):return self.run(positions,hidden,residual)

CLASSES={(0,34):DLower34,(0,32):DLower32,(0,28):DLower28,(0,24):DLower24,
    (34,36):DUpper34,(32,36):DUpper32,(28,36):DUpper28,(24,36):DUpper24}

def compiled_D_segment(adapter,module,positions,hidden,residual,begin,end):
    from vllm.config import set_current_vllm_config,CUDAGraphMode
    from vllm.config.kernel import IrOpPriorityConfig
    from vllm.forward_context import get_forward_context,override_forward_context
    assert adapter.role=='decode' and adapter.cfg['D_compile_partial'] and adapter.cfg['D_scoped_prompt_custom_ops']
    assert (begin,end) in CLASSES,(begin,end)
    native=adapter.active['runner'].vllm_config;assert int(native.compilation_config.mode)==3
    cache=getattr(adapter,'_compiled_D_intervals',None)
    if cache is None:cache={};adapter._compiled_D_intervals=cache
    key=(begin,end,residual is None)
    if key not in cache:
        local=copy.copy(native);local.compilation_config=copy.copy(native.compilation_config)
        cc=local.compilation_config;cc.custom_ops=['all'];cc.local_cache_dir=None;cc.traced_files=set()
        sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        cc.cache_dir=str(Path('/home/esjung/.cache/vllm/dense_ar_output_aware_v2/D_partial_v12')/sha/native.compute_hash()/f'{begin}_{end}_none{residual is None}')
        with set_current_vllm_config(local,check_compile=False):cache[key]=CLASSES[begin,end](list(module.layers[begin:end]))
        assert not cache[key].do_not_compile
    block=cache[key];context=copy.copy(get_forward_context());native_cache=native.compilation_config.cache_dir
    context.skip_compiled=False;context.cudagraph_runtime_mode=CUDAGraphMode.NONE;context.batch_descriptor=None
    context.v2_compiled_prompt_cuda_ops=True
    assert block.compilation_config is not native.compilation_config and block.compilation_config.cache_dir!=native_cache
    with override_forward_context(context),set_current_vllm_config(block.vllm_config,check_compile=False),IrOpPriorityConfig.with_default(['vllm_c','native']).set_priority():
        result=block(positions,hidden,residual)
    compiled=bool(block.compiled or getattr(block,'aot_compiled_fn',None) is not None)
    assert compiled,'Do not call an eager fallback a compiled D interval'
    assert native.compilation_config.cache_dir==native_cache
    adapter.rows.append(dict(event='D_COMPILED_PARTIAL_SEGMENT',time=time.monotonic(),begin=begin,end=end,
        query_rows=int(hidden.shape[0]),compiled=True,CUDA_graph=False,cache=block.compilation_config.cache_dir,
        scope='Original fixed layer interval and unchanged causal metadata. Original CUDA prompt primitives. Native decode-only path is unchanged.'))
    return result
