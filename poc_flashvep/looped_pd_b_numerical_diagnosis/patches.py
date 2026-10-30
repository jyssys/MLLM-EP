"""Independent process-local ablations. The installed engine stays read-only."""
import os
import types


def install():
    variant=os.environ.get('B_NUMERICAL_VARIANT','original')
    import torch
    from vllm_rlt.models import AutoModelForCausalLM
    if getattr(AutoModelForCausalLM,'_numerical_diagnosis',False): return
    AutoModelForCausalLM._numerical_diagnosis=True
    accepted={'original','down_fixed','sensitive_fixed','fixed_core','coda_guard',
              'down_core','sensitive_core','sensitive_core_coda','canonical_all',
              'norm_fixed','down_norm','down_norm_core_coda','down_norm_adapter'}
    if variant not in accepted: raise ValueError(variant)
    original=AutoModelForCausalLM.from_pretrained
    def load(cls,*args,**kwargs):
        model=original(*args,**kwargs)
        selected=[]
        for name,module in model.named_modules():
            if not isinstance(module,torch.nn.Linear): continue
            down=name.endswith('down_proj') or '.mlp.proj' in name
            sensitive=down or name=='lm_head' or name.endswith('adapter')
            if ((variant.startswith('down') and (down or
                    (variant=='down_norm_adapter' and name.endswith('adapter')))) or
                (variant.startswith('sensitive') and sensitive)):
                base=module.forward
                def forward(self,x,_base=base):
                    if x.ndim!=2 or len(x)==0: return _base(x)
                    # Fixed 256-row tiles cover both packed prompt and decode calls.
                    # Real output rows retain their original order. This adds work.
                    outputs=[]
                    for left in range(0,len(x),256):
                        real=x[left:left+256]
                        pad=x.new_zeros((256,x.shape[-1]))
                        pad[:len(real)].copy_(real)
                        outputs.append(_base(pad)[:len(real)])
                    return torch.cat(outputs,dim=0) if len(outputs)>1 else outputs[0]
                module.forward=types.MethodType(forward,module)
                selected.append(name)
        from .common import ROOT,write,digest
        from .norm_probe import fixed_norm
        write(ROOT/'raw'/os.environ['B_NUMERICAL_RUN']/f'projection_patch_{os.getpid()}.json',
              dict(variant=variant,selected=selected,geometry=256,source=digest(__file__),
                   fixed_norm_code_sha256=__import__('hashlib').sha256(__import__('marshal').dumps(fixed_norm.__code__)).hexdigest(),
                   scope='Process-local projection ablation. BF16 weights and fixed depth unchanged.'))
        return model
    AutoModelForCausalLM.from_pretrained=classmethod(load)
    if 'norm' in variant:
        from vllm_rlt.layers.normalization import RMSNorm
        from .norm_probe import fixed_norm
        RMSNorm.forward=fixed_norm
    if 'core' in variant or variant=='fixed_core':
        from vllm_rlt.worker import cuda_graph
        from vllm_rlt.config import cuda_graph_rows
        select=cuda_graph.RecurrentGraphs.select
        def fixed(self,hidden,batch):
            if batch.cu_seqlens_q is not None: return select(self,hidden,batch)
            self.capture_sizes=(cuda_graph_rows(self.execution),)
            return cuda_graph.DeviceRecurrentGraphs.select(self,hidden,batch)
        cuda_graph.RecurrentGraphs.select=fixed
    if 'coda' in variant:
        from ..looped_pd_b_slo_serving.repair_coda import install as coda
        coda(dict(boundary_bucket_calls=0))
    if variant=='canonical_all':
        from ..looped_pd_b_slo_serving.repair_hooks import install as old
        os.environ['B_REPAIR_VARIANT']='canonical_all'
        from .common import ROOT
        os.environ['B_REPAIR_RUN_DIR']=str(ROOT/'raw'/os.environ['B_NUMERICAL_RUN'])
        old()
