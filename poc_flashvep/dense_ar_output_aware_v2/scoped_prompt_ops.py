"""Use original CUDA primitives only in explicitly eager prompt contexts."""
import atexit
from .common import ROOT,write

def install(cfg,role):
    active=(role=='decode' and cfg.get('D_scoped_prompt_custom_ops')) or (role=='single' and cfg.get('teacher_scoped_prompt_custom_ops'))
    if not active:return
    from vllm.model_executor.custom_op import CustomOp
    from vllm.forward_context import is_forward_context_available,get_forward_context
    original=CustomOp.forward
    def forward(self,*args,**kwargs):
        if is_forward_context_available() and get_forward_context().skip_compiled and type(self).forward_cuda is not CustomOp.forward_cuda:
            return self.forward_cuda(*args,**kwargs)
        return original(self,*args,**kwargs)
    CustomOp.forward=forward
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/f'prompt_custom_op_scope_{role}.json',dict(
        installed=True,scope='Only explicit eager prompt contexts use available original CUDA CustomOp methods. '
            'Compiled decode retains the original native dispatch. No weights,tensors,tokens or timestamps are corrected. '
            'This source policy is not an exclusive-kernel measurement or a decode-safety result.')))
