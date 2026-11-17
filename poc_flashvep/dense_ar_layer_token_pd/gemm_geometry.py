"""Time isolated, exact BF16 projections at the realized axis widths."""
import json
import sys
import time
from .common import ROOT, write


def main():
    cfg=json.load(open(sys.argv[1]))
    folder=ROOT/'raw'/cfg['name']
    write(folder/'status.json',dict(status='RUNNING',time=time.time()))
    import torch
    from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    from ..dense_ar_slo_layer_pd.watch import GPUWatch
    from ..dense_ar_slo_layer_pd.analyze import percentile
    torch.set_num_threads(1)
    enable_batch_invariant_mode()
    torch.backends.cuda.matmul.allow_tf32=False
    witness=ROOT/'raw/TP1_first_tensor_witness_req4_v5'
    tensors=torch.load(witness/'tensors_rank0.pt',weights_only=True)
    weights=torch.load(witness/'weights_rank0.pt',weights_only=True)
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start()
    normal=False
    results=[]
    try:
        widths=sorted(set([4*8000,8*8000]+[4*s for s in (512,1024,1776,1792,2048,2656,2672,4096)]
            +[4*(8000-s) for s in (512,1024,2048,2656,2672,4096)]))
        for device in range(2):
            torch.cuda.set_device(device)
            for suffix in ('self_attn.qkv_proj','self_attn.o_proj','mlp.gate_up_proj','mlp.down_proj'):
                name='model.layers.0.'+suffix
                w=weights[name+'.weight'].cuda()
                operand=tensors['0:'+name+':input.0']
                for rows in widths:
                    x=operand.repeat((rows+len(operand)-1)//len(operand),1)[:rows].contiguous().cuda()
                    for _ in range(5):
                        y=torch.nn.functional.linear(x,w)
                    torch.cuda.synchronize()
                    times=[]
                    for _ in range(30):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record();y=torch.nn.functional.linear(x,w);end.record();end.synchronize()
                        times.append(begin.elapsed_time(end))
                    flops=2*rows*w.shape[0]*w.shape[1]
                    results.append(dict(device=device,GPU_UUID=str(torch.cuda.get_device_properties(device).uuid),
                        projection=name,M=rows,N=w.shape[0],K=w.shape[1],dtype='bfloat16',samples=len(times),
                        p50_ms=percentile(times,.5),p95_ms=percentile(times,.95),
                        structural_FLOPs=flops,operator_enclosure_effective_TFLOPs_s=flops/(percentile(times,.5)*1e9)))
                    del x,y
                del w
        write(folder/'gemm_geometry_costs.json',dict(results=results,
            evidence='MEASURED_ISOLATED_CUDA_EVENT_OPERATOR_ENCLOSURES',
            operand_source=str(witness),actual_original_BF16_weights=True,TF32_enabled=False,
            scope='Repeat captured actual rows to select M. The original batch-invariant BF16 linear implementation is unchanged. Single-operator stream event enclosures include launch gaps and are not exclusive kernel times. Model, KV, attention, transfer, host queues and online decode contention are absent. Effective TFLOPs/s is FLOPs divided by this enclosure, not serving speedup or a clean physical-idle measurement.'))
        normal=True
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    finally:
        monitor.stop()
        write(folder/'worker_exits.json',dict(passed=normal))


if __name__=='__main__':
    main()
