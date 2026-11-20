"""Isolate actual serving Linear APIs with CUDA-graph submission control."""
import json
import os
import sys
import time
import numpy as np
from .common import ROOT,write,MODEL_REVISION
from .engine_options import options


def main():
    cfg=json.load(open(sys.argv[1]));folder=ROOT/'raw'/cfg['name']
    assert os.environ['VLLM_BATCH_INVARIANT']=='0'
    import torch
    torch.set_num_threads(1)
    from vllm import LLM
    from .activity import Activity
    engine=None;collector=None;rows=[]
    try:
        write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
        opts=options(cfg);opts.update(compilation_config=dict(mode=0,cudagraph_mode='NONE'),async_scheduling=False)
        llm=LLM(**opts);engine=llm.llm_engine
        runner=engine.model_executor.driver_worker.worker.model_runner;model=runner.model.model
        sub=folder/'activity_all';sub.mkdir();collector=Activity(sub);collector.pause()
        write(folder/'actual_runtime.json',dict(options=opts,BI=False,model_revision=MODEL_REVISION,
            GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
            scope='Serving Linear module methods with resident original BF16 weights. Projection outputs are not model outputs.'))
        with torch.inference_mode():
            for length in (4608,8064):
                for cut in (1,2,4,6,8,12):
                    for axis in ('Layer','Token'):
                        query=length if axis=='Layer' else length*cut//36
                        indices=list(range(36-cut,36)) if axis=='Layer' else list(range(36))
                        modules=[];shapes=[];inputs={}
                        for index in indices:
                            layer=model.layers[index]
                            for component,module in (('QKV',layer.self_attn.qkv_proj),('O',layer.self_attn.o_proj),
                                ('gate_up',layer.mlp.gate_up_proj),('down',layer.mlp.down_proj)):
                                n,k=map(int,module.weight.shape)
                                key=(query,k)
                                if key not in inputs:inputs[key]=torch.zeros((query,k),device='cuda',dtype=module.weight.dtype)
                                modules.append((module,inputs[key]));shapes.append(dict(layer=index,component=component,M=query,N=n,K=k,
                                    module=type(module).__name__,quant_method=type(module.quant_method).__name__))
                        def action():
                            for module,x in modules:module(x)
                        for _ in range(3):action()
                        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):action()
                        samples={}
                        for label in ('clean_before','profile','clean_after'):
                            measured=[]
                            if label=='profile':assert collector.lib.activity_enable(0)==0
                            try:
                                for repeat in range(32):
                                    torch.cuda.synchronize()
                                    left=collector.stamp() if label=='profile' else None
                                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                    begin.record();graph.replay();end.record();end.synchronize()
                                    right=collector.stamp() if label=='profile' else None
                                    measured.append(dict(CUDA_enclosure_ms=begin.elapsed_time(end),warmup=repeat<2,
                                        CUPTI_left=left,CUPTI_right=right))
                            finally:
                                if label=='profile':collector.pause()
                            samples[label]=measured
                        rows.append(dict(input_tokens=length,cut=cut,axis=axis,logical_rows=length*cut,
                            actual_linear_calls=len(modules),GEMM_shapes=shapes,samples=samples,
                            scope='CUDA graph contains only the actual serving Linear calls. No attention, norm, activation, KV, transport or scheduling. '
                            'Zero input is a shape diagnostic, not a workload request. Repeated calls are not independent starts.'))
                        write(folder/'gemm_API_partial.json',dict(rows=rows))
                        print('GEMM_API',length,cut,axis,flush=True)
                        del graph,modules,inputs
        write(folder/'gemm_API_costs.json',dict(rows=rows,evidence='ISOLATED_GEMM_API_MEASUREMENT',
            actual_batch=1,BI=False,logical_rows_exact=True,model_revision=MODEL_REVISION))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error)));raise
    finally:
        if collector is not None:collector.close()
        if engine is not None:engine.engine_core.shutdown()


if __name__=='__main__':main()
