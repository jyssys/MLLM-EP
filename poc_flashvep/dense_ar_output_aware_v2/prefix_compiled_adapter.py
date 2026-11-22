"""Experimental P-only compiled lower-layer segment with original weights."""
import copy
import hashlib
from pathlib import Path
import time
import torch
from vllm.compilation.decorators import support_torch_compile
from poc_flashvep.dense_ar_longbench400_pd.compiled_full_adapter import CompiledFullAdapter

class _PrefixBase(torch.nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers=torch.nn.ModuleList(layers)

    def run_layers(self,positions,hidden,residual):
        for layer in self.layers:hidden,residual=layer(positions,hidden,residual)
        return hidden,residual

# Separate forward identities also prevent action constants from sharing an
# AOT artifact whose guards were intentionally removed by the serving compiler.
@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class PrefixLower34(_PrefixBase):
    def forward(self,positions,hidden,residual):return self.run_layers(positions,hidden,residual)

@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class PrefixLower32(_PrefixBase):
    def forward(self,positions,hidden,residual):return self.run_layers(positions,hidden,residual)

@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class PrefixLower28(_PrefixBase):
    def forward(self,positions,hidden,residual):return self.run_layers(positions,hidden,residual)

@support_torch_compile(dynamic_arg_dims={'positions':0,'hidden':0,'residual':0})
class PrefixLower24(_PrefixBase):
    def forward(self,positions,hidden,residual):return self.run_layers(positions,hidden,residual)

def isolated_prefix_config(original,end):
    config=copy.copy(original);config.compilation_config=copy.copy(original.compilation_config)
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    cc=config.compilation_config
    factors=original.compute_hash()
    cc.cache_dir=str(Path('/home/esjung/.cache/vllm/dense_ar_output_aware_v2/prefix_compiler_v3')/sha/factors/f'lower{end}')
    cc.local_cache_dir=None;cc.traced_files=set()
    return config

class PrefixCompiledAdapter(CompiledFullAdapter):
    def compiled_prefix_segment(self,module,positions,hidden,residual,begin,end):
        from vllm.config import set_current_vllm_config,CUDAGraphMode
        from vllm.config.kernel import IrOpPriorityConfig
        from vllm.forward_context import get_forward_context,override_forward_context
        assert self.role=='prefill' and self.cfg.get('P_compile_prefix')
        assert begin==0 and end==36-int(self.cfg['axis_cut'])
        runner=self.active['runner'];config=runner.vllm_config
        assert int(config.compilation_config.mode)==3
        cache=getattr(self,'_compiled_prefix_segments',None)
        if cache is None:cache={};self._compiled_prefix_segments=cache
        key=(begin,end)
        if key not in cache:
            local=isolated_prefix_config(config,end)
            cls={34:PrefixLower34,32:PrefixLower32,28:PrefixLower28,24:PrefixLower24}[end]
            with set_current_vllm_config(local,check_compile=False):cache[key]=cls(list(module.layers[begin:end]))
            assert not cache[key].do_not_compile
        block=cache[key];context=copy.copy(get_forward_context())
        native_cache=config.compilation_config.cache_dir
        assert block.compilation_config is not config.compilation_config
        assert block.compilation_config.cache_dir!=native_cache
        context.skip_compiled=False;context.cudagraph_runtime_mode=CUDAGraphMode.NONE;context.batch_descriptor=None
        # Preserve the explicitly qualified eager CUDA RMSNorm provider. This
        # changes graph execution,not weights,causal metadata or residual state.
        with override_forward_context(context),set_current_vllm_config(block.vllm_config,check_compile=False),IrOpPriorityConfig.with_default(['vllm_c','native']).set_priority():
            result=block(positions,hidden,residual)
        assert config.compilation_config.cache_dir==native_cache
        compiled=bool(block.compiled or getattr(block,'aot_compiled_fn',None) is not None)
        assert compiled,'Do not label an eager fallback as a compiled prefix'
        self.rows.append(dict(event='P_COMPILED_PREFIX_SEGMENT',time=time.monotonic(),begin=begin,end=end,
            query_rows=int(hidden.shape[0]),compiled=compiled,CUDA_graph=False,
            native_cache=native_cache,prefix_cache=block.compilation_config.cache_dir,
            scope='Original lower-layer weights and causal metadata. Returns hidden plus residual before final norm.'))
        return result
