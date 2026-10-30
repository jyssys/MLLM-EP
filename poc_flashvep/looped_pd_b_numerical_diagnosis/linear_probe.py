"""Real checkpoint GEMMs with identical operands across batch geometry."""
import argparse
import importlib.metadata
import json
import time
import torch
from .common import ROOT, MODELS, setup, write


def metrics(a, b):
    if torch.equal(a,b):
        return dict(exact=True,agreement=1.0,different_elements=0,max_absolute=0.0,relative_L2=0.0)
    x, y = a.float(), b.float()
    delta = x-y
    different=int((a!=b).sum())
    return dict(exact=bool(torch.equal(a,b)), agreement=(a.numel()-different)/a.numel(),
                different_elements=different, max_absolute=float(delta.abs().max()),
                relative_L2=float(delta.norm()/x.norm().clamp_min(1e-30)))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', choices=('ouro','huginn'), required=True)
    parser.add_argument('--name', required=True)
    args=parser.parse_args()
    setup()
    dest=ROOT/'raw'/args.name
    if (dest/'result.json').exists():
        print('REUSE',args.name)
        return
    dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    from vllm_rlt.models import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODELS[args.model],device='cuda:0',dtype=torch.bfloat16)
    # Cover every distinct projection geometry and every role in the model.
    chosen={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):
            role = ('down' if name.endswith('down_proj') or '.mlp.proj' in name else
                    'gate_up' if name.endswith(('gate_proj','up_proj','.fc')) else
                    'head' if name=='lm_head' else
                    'adapter' if name.endswith('adapter') else 'attention')
            key=(role,tuple(module.weight.shape))
            chosen.setdefault(key,(name,module))
    result=[]
    generator=torch.Generator(device='cuda').manual_seed(20261030)
    for (role,shape),(name,module) in chosen.items():
        operands=torch.randn((256,shape[1]),device='cuda',dtype=torch.bfloat16,generator=generator)
        # This uses real checkpoint weights and seeded operands, not captured failure activations.
        reference=torch._C._nn.linear(operands[:1],module.weight,module.bias)[0].clone()
        for size in (1,2,3,4,8,16,32,64,128,256):
            for reverse in (False,True):
                x=operands[:size].flip(0) if reverse else operands[:size]
                row=size-1 if reverse else 0
                actual=torch._C._nn.linear(x,module.weight,module.bias)[row]
                padded=torch.zeros((256,shape[1]),device='cuda',dtype=torch.bfloat16)
                padded[:size].copy_(x)
                fixed=torch._C._nn.linear(padded,module.weight,module.bias)[row]
                fixed_ref=torch._C._nn.linear(operands,module.weight,module.bias)[0]
                item=dict(role=role,module=name,shape=shape,rows=size,reverse=reverse,
                          native=metrics(reference,actual),fixed_to_fixed=metrics(fixed_ref,fixed),
                          evidence='MEASURED_OPERATOR_DIAGNOSTIC',operands='seeded BF16, resident checkpoint weights')
                if role=='head':
                    top,ids=actual.float().topk(2)
                    item.update(top1=int(actual.argmax()),topk_first=int(ids[0]),topk_second=int(ids[1]),margin=float(top[0]-top[1]))
                result.append(item)
        timings={}
        for size in (1,8,16):
            x=operands[:size]
            for _ in range(10): torch._C._nn.linear(x,module.weight,module.bias)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100): torch._C._nn.linear(x,module.weight,module.bias)
            end.record(); end.synchronize()
            timings[size]=start.elapsed_time(end)/100
        for item in result:
            if item['module']==name: item['mean_GEMM_ms_by_rows']=timings
        print(name,shape,'nonexact',sum(not r['native']['exact'] for r in result if r['module']==name),flush=True)
    write(dest/'result.json',dict(model=args.model,torch=torch.__version__,CUDA=torch.version.cuda,
        GPU=torch.cuda.get_device_name(),UUID=str(torch.cuda.get_device_properties(0).uuid),
        matmul_reduced_BF16=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        projections=result,time=time.time(),scope='Operator numerical susceptibility. Not historical earliest-operation proof or serving speedup.'))


if __name__=='__main__': main()
