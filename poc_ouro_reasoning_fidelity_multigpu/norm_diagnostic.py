"""Calibration-only operator control. Never use this as the quality baseline."""
import types
from poc_ouro_reasoning_fidelity_multigpu.worker_probe import FidelityProbe
from poc_ouro_reasoning_fidelity_multigpu.contract import MODEL

class NormDiagnosticProbe(FidelityProbe):
    def check_loaded_weights(self):
        import hashlib
        import torch
        from safetensors import safe_open
        records=[]
        def digest(value):
            return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        with safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as source:
            for name,value in self.get_model().named_parameters():
                if '.qkv_proj.' in name:
                    originals=[name.replace('.qkv_proj.','.'+part+'_proj.') for part in ['q','k','v']]
                elif '.gate_up_proj.' in name:
                    originals=[name.replace('.gate_up_proj.','.'+part+'_proj.') for part in ['gate','up']]
                else:
                    originals=[name]
                expected=[source.get_tensor(part) for part in originals]
                expected=expected[0] if len(expected)==1 else torch.cat(expected,dim=0)
                a,b=digest(value),digest(expected)
                records.append(dict(parameter=name,source_keys=originals,shape=list(value.shape),
                    loaded_sha256=a,source_sha256=b,equal=a==b))
        return dict(records=records,all_equal=all(r['equal'] for r in records),
            scope='Every loaded TP1 parameter, including stacked QKV and gate/up tensors. Outside timing.')

    def norm_mode(self,mode):
        import torch
        assert mode in ['native','hf_operator']
        if not hasattr(self,'norm_originals'):
            self.norm_originals=[(module,module.forward) for module in self.get_model().modules()
                if type(module).__name__=='RMSNorm']
        def hf_operator(module,x,residual=None):
            dtype=x.dtype
            if residual is not None:
                x=(x+residual).to(dtype)
                residual=x
            value=x.float()
            value=value*torch.rsqrt(value.pow(2).mean(-1,keepdim=True)+module.variance_epsilon)
            value=module.weight*value.to(dtype)
            return value if residual is None else (value,residual)
        for module,original in self.norm_originals:
            module.forward=original if mode=='native' else types.MethodType(hf_operator,module)
        return dict(mode=mode,physical_norm_modules=len(self.norm_originals),
            scope='Calibration operator ablation only. No performance or benchmark-quality claim.')
