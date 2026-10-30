"""Hot operator CUDA-graph timings. These are not whole-model speedups."""
import argparse
import time
import numpy as np
import torch
from .common import ROOT,MODELS,write
from .norm_probe import fixed_norm


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('ouro','huginn'),required=True);p.add_argument('--name',required=True)
    args=p.parse_args();dest=ROOT/'raw'/args.name
    if (dest/'result.json').exists():print('REUSE',args.name);return
    torch.set_num_threads(1)
    torch.manual_seed(20261030)
    from vllm_rlt.models import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODELS[args.model],device='cuda',dtype=torch.bfloat16)
    source=('teacher_ouro_sample3_down_norm_v3' if args.model=='ouro' else 'teacher_huginn_sample92_norm_v3')
    saved=torch.load(ROOT/'raw'/source/'isolated_same_operand_differences.pt',map_location='cpu',weights_only=True)
    record=next(r for r in saved if '.norm_' in r['key'][3] or 'layernorm' in r['key'][3])
    norm=model.get_submodule(record['key'][3]);operand=record['input'].to('cuda')
    down=model.get_submodule('model.layers.0.mlp.down_proj' if args.model=='ouro' else 'transformer.core_block.0.mlp.proj')
    results=[]
    for role in ('norm','down'):
        for rows in (1,2,3,4,8,16):
            x=(operand[None,:].repeat(rows,1) if role=='norm' else
               torch.randn((rows,down.weight.shape[1]),device='cuda',dtype=torch.bfloat16))
            if role=='norm':
                native=lambda:norm(x)
                fixed=lambda:fixed_norm(norm,x)
            else:
                native=lambda:down(x)
                def fixed():
                    staged=x.new_zeros((256,x.shape[-1]));staged[:len(x)].copy_(x)
                    return down(staged)[:len(x)]
            graphs={};outputs={};capture={}
            for label,call in (('native',native),('fixed',fixed)):
                for _ in range(3):call()
                torch.cuda.synchronize()
                begin=time.perf_counter();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output=call()
                capture[label]=1000*(time.perf_counter()-begin)
                graphs[label]=graph;outputs[label]=output
                for _ in range(10):graph.replay()
            samples={'native':[],'fixed':[]}
            for repeat in range(10):
                for label in ('native','fixed') if repeat%2==0 else ('fixed','native'):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):graphs[label].replay()
                    end.record();end.synchronize()
                    samples[label].append(1000*start.elapsed_time(end)/100)
            summary={label:dict(mean_us=float(np.mean(values)),p50_us=float(np.percentile(values,50)),
                               p95_us=float(np.percentile(values,95)),samples_us=values) for label,values in samples.items()}
            results.append(dict(role=role,rows=rows,timing=summary,capture_host_ms=capture,
                 fixed_over_native_mean_ratio=summary['fixed']['mean_us']/summary['native']['mean_us'],
                 operands='captured failure operand' if role=='norm' else 'seeded-shape BF16 with resident real checkpoint weight'))
    write(dest/'result.json',dict(model=args.model,results=results,torch=torch.__version__,CUDA=torch.version.cuda,
         UUID=str(torch.cuda.get_device_properties(0).uuid),physical_mask=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
         evidence='MEASURED_HOT_OPERATOR_CUDA_GRAPH_TIMING',
         scope='10 groups of100 replays. Native/fixed order alternates. Hot operator caches differ from whole-model execution. Not a serving latency or E2E speedup. Other GPU pairs may run numerical qualification.'))
    print([(r['role'],r['rows'],r['timing']['native']['mean_us'],r['timing']['fixed']['mean_us']) for r in results])


if __name__=='__main__':main()
