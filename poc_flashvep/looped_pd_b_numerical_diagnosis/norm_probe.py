"""Isolate FP32 RMSNorm sub-operations with a captured real failure operand."""
import argparse
import json
import torch
from .common import ROOT,MODELS,setup,write
from .linear_probe import metrics


def fixed_norm(self,hidden):
    """Keep small FP32 reduction geometry fixed. Do not pad KV or Linear calls."""
    value=hidden.float()
    square=value.square()
    if hidden.ndim==2 and 0<len(hidden)<16:
        staged=square.new_zeros((16,square.shape[1]))
        staged[:len(hidden)].copy_(square)
        variance=staged.mean(-1,keepdim=True)[:len(hidden)]
    else:
        variance=square.mean(-1,keepdim=True)
    value=value*torch.rsqrt(variance+self.variance_epsilon)
    return self.weight*value.to(hidden.dtype)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--name',required=True)
    args=p.parse_args()
    setup();dest=ROOT/'raw'/args.name
    if (dest/'result.json').exists(): print('REUSE',args.name);return
    from vllm_rlt.models import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODELS[args.model],device='cuda',dtype=torch.bfloat16)
    saved=torch.load(ROOT/'raw'/args.source/'isolated_same_operand_differences.pt',map_location='cpu',weights_only=True)
    tensor=next(r for r in saved if '.norm_' in r['key'][3] or 'layernorm' in r['key'][3] or r['key'][3].endswith('norm'))
    module=model.get_submodule(tensor['key'][3])
    a=tensor['input'].to('cuda')
    def stages(x):
        value=x.float()
        square=value.square()
        variance=square.mean(-1,keepdim=True)
        inverse=torch.rsqrt(variance+module.variance_epsilon)
        normalized=value*inverse
        return dict(square=square,variance=variance,inverse=inverse,
                    normalized=normalized,cast=normalized.to(x.dtype),output=module.weight*normalized.to(x.dtype))
    reference=stages(a[None,:])
    stable=stages(a[None,:].repeat(16,1))
    rows=[]
    for size in (*range(1,17),31,32,63,64,127,128,255,256):
        x=a[None,:].repeat(size,1)
        actual=stages(x)
        fixed_output=fixed_norm(module,x)
        rows.append(dict(rows=size,
            stages={key:metrics(reference[key][0],actual[key][0]) for key in reference},
            fixed_output_to_M16=metrics(stable['output'][0],fixed_output[0]),
            native_row_position_agreement=metrics(actual['output'][0].expand_as(actual['output']),actual['output']),
            fixed_all_rows_to_M16=metrics(stable['output'][0].expand_as(fixed_output),fixed_output),
            variance_first=float(actual['variance'][0]),inverse_first=float(actual['inverse'][0])))
    write(dest/'result.json',dict(source=args.source,captured_key=tensor['key'],
        rows=rows,evidence='MEASURED_REAL_OPERAND_OPERATOR_DIAGNOSTIC',
        scope='Same BF16 operand and checkpoint norm weight. FP32 mean/rsqrt/cast are observed separately.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
