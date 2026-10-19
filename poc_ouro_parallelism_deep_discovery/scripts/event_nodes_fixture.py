"""Primitive event-retention diagnostic. Not native Ouro pass timing or a method."""
import datetime as dt
import os
import statistics
import time
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(output/'inductor_cache')
    os.environ['TRITON_CACHE_DIR']=str(output/'triton_cache')
    configure_native_imports()
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    mask=validate_mask();assert len(mask)==1
    device=torch.cuda.get_device_properties(0)
    assert str(device.uuid).removeprefix('GPU-')==UUIDS[GPUS.index(mask[0])]
    info=MODELS['1.4B']
    source=ROOT/'outputs/capture_base14_v2/1.4B/actual_mlp_inputs.pt'
    values=torch.load(source,map_location='cpu',weights_only=True)
    x=values['fixed_0_layer0_call5'].to('cuda')
    assert x.shape==(1,2048)
    with safe_open(info['path']/'model.safetensors',framework='pt',device='cpu') as checkpoint:
        gate=checkpoint.get_tensor('model.layers.0.mlp.gate_proj.weight').to('cuda')
        up=checkpoint.get_tensor('model.layers.0.mlp.up_proj.weight').to('cuda')
        down=checkpoint.get_tensor('model.layers.0.mlp.down_proj.weight').to('cuda')
        norm=checkpoint.get_tensor('model.norm.weight').to('cuda')
    events=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(4)]
    for pair in events:
        for event in pair:event.record()
    torch.cuda.synchronize()
    # This artificial four-MLP chain only tests compiler/event mechanics.
    # It omits native Attention, KV, residual rules, collectives, and gate work.
    def clean(value):
        for _ in range(4):
            value=F.linear(F.silu(F.linear(value,gate))*F.linear(value,up),down)
            value=F.rms_norm(value,(2048,),norm,1e-6)
        return value
    def marked(value):
        for index in range(4):
            events[index][0].record()
            value=F.linear(F.silu(F.linear(value,gate))*F.linear(value,up),down)
            value=F.rms_norm(value,(2048,),norm,1e-6)
            events[index][1].record()
        return value
    metadata=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),physical_gpu=mask[0],logical_cuda=0,
                  UUID=str(device.uuid),model=info['id'],revision=info['revision'],source_sha256=sha(source),
                  source_key='fixed_0_layer0_call5',Torch=torch.__version__,evidence_class='GPU_MICROBENCH',
                  scope='Artificial four-MLP chain with actual pretrained weights and input. No native Ouro pass, attention, KV, collective, or E2E result.',
                  parity_rule='Bitwise equality of this fixture output. No tolerance relaxation.',
                  timer_rule='One actual replay per readout. No last-replay multiplication.',
                  native_instrumentation='NOT_VALIDATED_BY_THIS_FIXTURE')
    write(output/'fixture_contract.json',metadata)
    try:
        with torch.inference_mode():
            functions=[torch.compile(clean,fullgraph=True),torch.compile(marked,fullgraph=True)]
            for fn in functions:
                for _ in range(7):fn(x)
            torch.cuda.synchronize()
            graphs=[];results=[]
            for arm,fn in zip(['clean','marked'],functions):
                graph=torch.cuda.CUDAGraph();graph.enable_debug_mode()
                with torch.cuda.graph(graph):result=fn(x)
                graph.debug_dump(str(output/(arm+'.dot')))
                graphs.append(graph);results.append(result)
            torch.cuda.synchronize()
            assert torch.equal(results[0].view(torch.int16),results[1].view(torch.int16))
            timing=[[],[]];internal=[]
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for trial in range(31):
                for index in ([0,1] if trial%2==0 else [1,0]):
                    start.record();graphs[index].replay();end.record();end.synchronize()
                    timing[index].append(start.elapsed_time(end))
                    if index==1:
                        internal.append([a.elapsed_time(b) for a,b in events])
            assert torch.equal(results[0].view(torch.int16),results[1].view(torch.int16))
            median=[statistics.median(v) for v in timing]
            write(output/'fixture_results.json',dict(**metadata,output_bitwise_equal=True,
                  clean_replay_ms=timing[0],marked_replay_ms=timing[1],marked_internal_intervals_ms=internal,
                  clean_median_ms=median[0],marked_median_ms=median[1],observer_fraction=median[1]/median[0]-1,
                  graph_debug_paths=['clean.dot','marked.dot'],
                  interpretation='Fixture mechanics only. Native graph-boundary ordering, identical lowering, and native observer overhead remain unvalidated.'))
    except Exception as error:
        write(output/'status.json',dict(success=False,classification='EVENT_FIXTURE_UNSUPPORTED_OR_INVALID',
              error_type=type(error).__name__,error=str(error),scope=metadata['scope']))
        raise
    write(output/'status.json',dict(success=True,scope=metadata['scope'],
          native_pass_timing='NOT_RUN',candidate_speedup='NOT_RUN'))


if __name__=='__main__':main()
