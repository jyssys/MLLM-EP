"""Measure exact BF16 operator shapes without claiming serving speedup."""
import json
import math
import statistics
import sys
import time
import traceback
from .common import ROOT,PREVIOUS,write,digest


def main():
    cfg=json.loads(open(sys.argv[1]).read())
    folder=ROOT/'raw'/cfg['name']
    write(folder/'status.json',dict(status='RUNNING',time=time.time()))
    import torch
    from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
    from ..dense_ar_slo_layer_pd.watch import GPUWatch
    from ..dense_ar_slo_layer_pd.analyze import percentile
    torch.set_num_threads(1)
    enable_batch_invariant_mode()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(2026111780)
    witness=PREVIOUS/'raw/TP1_first_tensor_witness_req4_v5'
    tensors=torch.load(witness/'tensors_rank0.pt',weights_only=True)
    weights=torch.load(witness/'weights_rank0.pt',weights_only=True)
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name'])
    monitor.start()
    results=[]
    profiles=[]
    normal=False

    def timed(operation,samples=30,warm=5):
        for _ in range(warm):
            operation()
        torch.cuda.synchronize()
        pairs=[]
        for _ in range(samples):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();operation();b.record()
            pairs.append((a,b))
        torch.cuda.synchronize()
        return [a.elapsed_time(b) for a,b in pairs]

    def probe(operation,label):
        """Keep A/B/A enclosures and the complete kernel trace."""
        before=timed(operation)
        path=folder/(label+'.trace.json')
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as observer:
                observed=timed(operation,samples=10,warm=0)
            observer.export_chrome_trace(str(path))
            after=timed(operation)
            control=(statistics.median(before)+statistics.median(after))/2
            ratio=statistics.median(observed)/control
            stable=abs(statistics.median(after)/statistics.median(before)-1)<=.05
            gate=abs(ratio-1)<=.05 and stable
            kernels=[r for r in json.loads(path.read_text())['traceEvents'] if r.get('cat')=='kernel']
            kernel_totals={}
            for event in kernels:
                item=kernel_totals.setdefault(event['name'],dict(count=0,duration_us=0.,streams=[]))
                item['count']+=1
                item['duration_us']+=event['dur']
                stream=event.get('args',{}).get('stream')
                if stream not in item['streams']:
                    item['streams'].append(stream)
            value=dict(label=label,trace=str(path),trace_SHA256=digest(path),
                before_ms=before,observed_ms=observed,after_ms=after,
                observed_over_control=ratio,controls_within_5pct=stable,
                observer_gate_5pct=gate,kernels=kernel_totals,
                evidence='MEASURED' if gate else 'DIAGNOSTIC',measurement_class='ISOLATED_KERNEL_TRACE',
                scope='Isolated operator only. A/B/A checks event-enclosure perturbation. This is not an online serving profiler gate. No physical serving idle or SM/HBM counter is measured.')
        except BaseException as error:
            value=dict(label=label,error=repr(error),traceback=traceback.format_exc(),
                evidence='FAILED',scope='Preserve clean event samples. Do not infer a kernel time from this failed capture.')
        profiles.append(value)
        return before

    try:
        widths=sorted(set([8000]+[s for s in (880,896,1776,1792,2656,2672)]
            +[8*s for s in (880,896)]+[4*s for s in (1776,1792)]
            +[3*s for s in (2656,2672)]+[8000-s for s in (880,896,1776,1792,2656,2672)]))
        if cfg.get('isolated_widths'):
            widths=sorted(set(cfg['isolated_widths']))
            assert all(isinstance(m,int) and 0<m<=8192 for m in widths)
        for device in range(2):
            torch.cuda.set_device(device)
            gpu_uuid=str(torch.cuda.get_device_properties(device).uuid)
            for suffix in ('self_attn.qkv_proj','self_attn.o_proj','mlp.gate_up_proj','mlp.down_proj'):
                name='model.layers.0.'+suffix
                weight=weights[name+'.weight'].to(device='cuda',dtype=torch.bfloat16)
                operand=tensors['0:'+name+':input.0']
                for rows in widths:
                    x=operand.repeat((rows+len(operand)-1)//len(operand),1)[:rows].contiguous().cuda()
                    operation=lambda x=x,weight=weight:torch.nn.functional.linear(x,weight)
                    label=f'GPU{device}_{suffix.replace(".","_")}_M{rows}'
                    values=probe(operation,label) if rows in (8000,2672) else timed(operation)
                    flops=2*rows*weight.shape[0]*weight.shape[1]
                    results.append(dict(kind='GEMM',GPU=device,GPU_UUID=gpu_uuid,
                        projection=name,M=rows,N=weight.shape[0],K=weight.shape[1],
                        dtype=str(x.dtype),samples_ms=values,p50_ms=percentile(values,.5),
                        p95_ms=percentile(values,.95),structural_FLOPs=flops,
                        enclosure_effective_TFLOPs_s=flops/(percentile(values,.5)*1e9),
                        minimum_read_write_bytes=2*(x.numel()+weight.numel()+rows*weight.shape[0]),
                        scope='Actual original BF16 weights and repeated captured activation rows. Tensor rows set M only. CUDA event enclosure includes launch gaps. This is not exclusive GEMM kernel time.'))
                    del x
                del weight
            # Same exact FA3 API as the local H100 backend. These are shape
            # diagnostics, not outputs from an extra approximate model.
            shapes=[(batch,query,8000) for batch in (1,3,4,8)
                for query in (8000,880,896,1776,1792,2656,2672) if query*batch<=8192]
            shapes.extend((1,8000-suffix,8000-suffix) for suffix in (880,896,1776,1792,2656,2672))
            for batch,query,key_tokens in ([] if cfg.get('isolated_skip_attention') else shapes):
                    q=torch.randn(batch*query,32,128,device='cuda',dtype=torch.bfloat16)
                    k=torch.randn(batch*key_tokens,8,128,device='cuda',dtype=torch.bfloat16)
                    v=torch.randn_like(k)
                    cuq=torch.arange(batch+1,device='cuda',dtype=torch.int32)*query
                    cuk=torch.arange(batch+1,device='cuda',dtype=torch.int32)*key_tokens
                    operation=lambda q=q,k=k,v=v,cuq=cuq,cuk=cuk,query=query,key_tokens=key_tokens:flash_attn_varlen_func(q,k,v,max_seqlen_q=query,
                        cu_seqlens_q=cuq,max_seqlen_k=key_tokens,cu_seqlens_k=cuk,
                        causal=True,softmax_scale=1/math.sqrt(128),fa_version=3)
                    label=f'GPU{device}_FA3_B{batch}_Q{query}_K{key_tokens}'
                    values=probe(operation,label) if batch==1 and query in (8000,896,1792,2672) else timed(operation)
                    pairs=batch*(query*(key_tokens-query)+query*(query+1)//2)
                    results.append(dict(kind='ATTENTION',GPU=device,GPU_UUID=gpu_uuid,
                        batch=batch,query_tokens=query,KV_tokens=key_tokens,query_heads=32,KV_heads=8,
                        head_dimension=128,dtype='bfloat16',fa_version=3,causal='bottom_right',
                        cache_layout='CONTIGUOUS_NONPAGED',operand_source='SEEDED_RANDOM_SHAPE_DIAGNOSTIC',
                        samples_ms=values,p50_ms=percentile(values,.5),p95_ms=percentile(values,.95),
                        structural_causal_pairs=pairs,structural_attention_FLOPs=4*pairs*32*128,
                        scope='Exact causal FA3 shape. Synthetic BF16 operands isolate geometry. Serving uses paged KV and real prompts. This microbenchmark excludes KV append, RoPE, norm, transfer, queueing and decode contention.'))
                    del q,k,v,cuq,cuk
            torch.cuda.empty_cache()
        write(folder/'operator_geometry.json',dict(results=results,profiles=profiles,
            original_weights_SHA256=digest(witness/'weights_rank0.pt'),
            captured_activations_SHA256=digest(witness/'tensors_rank0.pt'),
            TF32_enabled=False,SM_counters='NOT_MEASURED',HBM_counters='NOT_MEASURED',
            evidence='MEASURED',measurement_class='ISOLATED_CUDA_EVENT_OPERATOR_ENCLOSURES',
            scope='No serving speedup is inferred. Every clean sample remains separate from profiled samples. Attention inputs are synthetic shape diagnostics.'))
        normal=True
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        monitor.stop()
        write(folder/'worker_exits.json',dict(passed=normal))


if __name__=='__main__':
    main()
