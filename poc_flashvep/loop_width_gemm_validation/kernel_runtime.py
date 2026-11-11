"""Optional fixed operator policy. No scheduler, state, or attention changes."""
import os,types
from collections import Counter
from .common import *


def install(cfg):
    if cfg.get('projection_kernel_policy') not in ('tuned_s1','tuned_v2'):return
    import torch
    import torch.nn.functional as F
    from vllm_rlt.models import AutoModelForCausalLM
    from vllm_rlt.engine.stage_engine import StageEngine
    from .lt import Plan,library
    policy_path=Path(cfg['projection_kernel_policy_file'])
    assert digest(policy_path)==cfg['projection_kernel_policy_SHA256']
    policy=json.loads(policy_path.read_text());data=policy['models'][cfg['model']]
    assert library().lt_version()==data['cublasLt_version']
    entries={(r['role'],r['M'],r['N'],r['K']):r for r in data['entries']}
    load,close=AutoModelForCausalLM.from_pretrained,StageEngine.close
    def loaded(cls,*args,**kwargs):
        model=load(*args,**kwargs);plans={};modules=[];counts=Counter()
        device=next(model.parameters()).device
        # Only large-row recurrent projections use this scratch buffer.
        # Decode shapes below 16 rows retain the qualified original function.
        # Qualified runtimes submit recurrent prompt work on one ordered lane.
        shared=torch.empty(64*1024**2,device=device,dtype=torch.uint8)
        for name,module in model.named_modules():
            core=('transformer.core_block.' in name or name=='transformer.adapter') if cfg['model']=='huginn' else 'model.layers.' in name
            if not core or not isinstance(module,torch.nn.Linear):continue
            assert module.bias is None,'Bias requires its own exact operator qualification'
            role=('down' if name.endswith('down_proj') or '.mlp.proj' in name else 'adapter' if name.endswith('adapter') else
                'up' if '.mlp.' in name else 'attention')
            n,k=module.weight.shape
            eligible={m:r for (kind,m,a,b),r in entries.items() if kind==role and a==n and b==k and r['method']!='qualified_reference'}
            if not eligible:continue
            for m,choice in eligible.items():
                key=(m,n,k)
                if choice['method']=='cublasLt' and key not in plans:
                    plan=Plan(m,n,k,workspace_buffer=shared)
                    index=choice['algorithm']['index']
                    assert plan.metadata(index)==choice['algorithm'],('Algorithm order or attributes changed',key)
                    plans[key]=plan
            base=module.forward
            def forward(self,x,_base=base,_eligible=eligible,_n=n,_k=k,_name=name):
                if x.ndim!=2 or len(x) not in _eligible:return _base(x)
                assert x.is_contiguous() and x.dtype==torch.bfloat16
                choice=_eligible[len(x)]
                counts[(_name,len(x),choice['method'],'capture' if torch.cuda.is_current_stream_capturing() else 'eager')]+=1
                if choice['method']=='unpadded':return F.linear(x,self.weight)
                output=torch.empty((len(x),_n),device=x.device,dtype=x.dtype)
                return plans[(len(x),_n,_k)].run(choice['algorithm']['index'],x,self.weight,output)
            module.forward=types.MethodType(forward,module)
            modules.append(dict(name=name,role=role,N=n,K=k,rows=sorted(eligible)))
        model._width_kernel_state=dict(plans=plans,workspace=shared,modules=modules,counts=counts,
            policy_SHA256=cfg['projection_kernel_policy_SHA256'])
        write(ROOT/'raw'/cfg['name']/f'kernel_model_installation_{os.getpid()}.json',dict(
            policy_SHA256=cfg['projection_kernel_policy_SHA256'],modules=modules,shared_workspace_bytes=shared.numel(),
            scope='Fixed operator policy for recurrent projections. Attention, numerical normalization, request lifecycle, and Native handoff remain unchanged. Small Decode shapes retain the original implementation.'))
        return model
    def closed(engine):
        state=getattr(engine.model,'_width_kernel_state',None)
        if state:
            write(ROOT/'raw'/cfg['name']/f'kernel_dispatch_{os.getpid()}.json',dict(role=engine.stage_config.role,
                policy_SHA256=state['policy_SHA256'],host_calls=[dict(module=k[0],rows=k[1],method=k[2],execution=k[3],calls=v) for k,v in state['counts'].items()],
                scope='Host projection calls include graph capture. Graph replays do not call Python and are not counted here. No hardware kernel counter is implied.'))
        # Plans stay alive until Native graph teardown completes.
        return close(engine)
    AutoModelForCausalLM.from_pretrained=classmethod(loaded)
    StageEngine.close=closed
