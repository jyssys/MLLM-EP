"""Use actual checkpoint weights and live operands. This is not a serving candidate."""
from .common import *

def run_probe(engine,model,barrier,folder,cfg):
    import torch
    import torch.nn.functional as F
    from .capacity_worker import cohort
    from .telemetry import Recorder
    captured={};handles=[]
    targets=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and
        (n.endswith('down_proj') or '.mlp.proj' in n or n.endswith('adapter'))]
    # One physical down projection and the Huginn injection adapter are sufficient geometry diagnostics.
    down=[(n,m) for n,m in targets if not n.endswith('adapter')]
    selected=down[:1]+[(n,m) for n,m in targets if n.endswith('adapter')]
    for name,module in selected:
        def capture(mod,args,n=name):
            if n not in captured and barrier.mode=='decode':captured[n]=args[0].detach().clone()
        handles.append(module.register_forward_pre_hook(capture))
    rows=json.loads((ROOT/'data/test_L512.json').read_text())[:4]
    cohort(engine,[r['input_ids'][cfg['model']] for r in rows],4,cfg['seed'],'geometry_operand',barrier,Recorder(False))
    for handle in handles:handle.remove()
    results=[]
    for name,module in selected:
        if name not in captured:continue
        full=captured[name]
        for n in (1,2,4):
            if len(full)<n:continue
            x=full[:n];pad=x.new_zeros((256,x.shape[-1]));pad[:n].copy_(x)
            operations={'native_rows':lambda:F.linear(x,module.weight,module.bias),
                        'fixed_256_rows_prepared':lambda:F.linear(pad,module.weight,module.bias)[:n]}
            reference=None;values={}
            for kind,fn in operations.items():
                for _ in range(5):fn()
                torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):result=fn()
                end.record();end.synchronize();values[kind]=start.elapsed_time(end)/100
                if reference is None:reference=result.detach().clone()
                else:
                    delta=(reference.float()-result.float())
                    values.update(exact_element_fraction=float((reference==result).float().mean()),
                        max_abs=float(delta.abs().max()),relative_L2=float(delta.norm()/reference.float().norm().clamp_min(1e-20)))
            results.append(dict(module=name,input_rows=n,input_features=x.shape[-1],output_features=module.weight.shape[0],
                dtype='BF16',**values,scope='Standalone GEMM using actual live operand and checkpoint weight. Prepared padding excludes copy/allocation.',
                evidence='MEASURED_MICROBENCHMARK_NOT_SERVING_SPEEDUP',candidate_runtime_changed=False))
    write(folder/'projection_geometry_microbenchmark.json',results)
