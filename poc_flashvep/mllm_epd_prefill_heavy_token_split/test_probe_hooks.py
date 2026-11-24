"""Attention must retain its callable after separate GEMM hooks are installed."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ProbeHookTest(unittest.TestCase):
    def test_attention_and_gemm_hooks_keep_independent_originals(self):
        from torch._inductor.select_algorithm import extern_kernels
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from . import forward_probe as module
        called=[]
        def attention(*args,**kwargs):called.append('attention');return 'ORIGINAL_ATTENTION'
        def gemm(a,b,c=None,*,out=None):called.append('gemm');return 'ORIGINAL_GEMM'
        class Model:
            def compute_logits(self,*args,**kwargs):return None
        class Runner:
            def __init__(self):self.model=Model();self.kv_caches=[None]*28
            def execute_model(self,*args,**kwargs):return None
            def _model_forward(self,*args,**kwargs):return None
        with TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)), \
                patch.dict(os.environ,MLLM_EPD_ROLE='prefill'), \
                patch.object(FlashAttentionImpl,'forward',attention), \
                patch.object(extern_kernels,'mm',gemm),patch.object(extern_kernels,'addmm',gemm):
            probe=module.ForwardProbe(SimpleNamespace(model_runner=Runner()),'coexisting_hooks',4,
                include_attention=True,include_gemm=True)
            probe.begin()
            try:
                result=FlashAttentionImpl.forward(SimpleNamespace(),SimpleNamespace(layer_name='not-a-selected-layer'),
                    None,None,None,None,None,None)
                self.assertEqual(result,'ORIGINAL_ATTENTION')
                self.assertEqual(called,['attention'])
            finally:
                probe.close()

    def test_norm_inputs_are_captured_before_generated_code_deletes_weight(self):
        import torch
        from torch._inductor.select_algorithm import extern_kernels
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from . import forward_probe as module
        def gemm(a,b,c=None,*,out=None):return out.copy_(a@b)
        class Model:
            def compute_logits(self,*args,**kwargs):return None
        class Runner:
            def __init__(self):self.model=Model();self.kv_caches=[None]*28
            def execute_model(self,*args,**kwargs):return None
            def _model_forward(self,*args,**kwargs):return None
        def generated_call():
            arg3_1=torch.ones(4);arg4_1=torch.ones(2,4)
            extern_kernels.mm(torch.ones(2,4),torch.eye(4),out=torch.empty(2,4))
            del arg3_1
            extern_kernels.mm(torch.ones(2,4),torch.eye(4),out=torch.empty(2,4))
        with TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)), \
                patch.dict(os.environ,MLLM_EPD_ROLE='prefill'), \
                patch.object(FlashAttentionImpl,'forward',lambda *a,**k:None), \
                patch.object(extern_kernels,'mm',gemm),patch.object(extern_kernels,'addmm',gemm), \
                patch.dict(generated_call.__globals__,triton_red_fused_fused_add_rms_norm_0=SimpleNamespace(
                    configs=None,launchers=[SimpleNamespace(config=SimpleNamespace(kwargs={'R0_BLOCK':4096},
                        num_warps=16,num_stages=1))])):
            probe=module.ForwardProbe(SimpleNamespace(model_runner=Runner()),'generated_local_lifetime',4,
                include_attention=True,include_gemm=True)
            probe.begin();probe.events.append(dict(index=0,query_tokens=2));probe.last_attention_layer=0
            try:
                generated_call()
                self.assertTrue(torch.equal(probe.tensors['norm_0_weight_0'],torch.ones(4)))
                self.assertTrue(torch.equal(probe.tensors['norm_0_residual_0'],torch.ones(2,4)))
                kernels=probe.events[0]['norm_runtime']['kernels']
                self.assertEqual(kernels[0]['configs'],[])
                self.assertEqual(kernels[0]['launcher_configs'][0]['kwargs'],{'R0_BLOCK':4096})
            finally:probe.close()


if __name__=='__main__':unittest.main()
