"""Whole-rollout attribution. Percentages partition one endpoint CUDA timeline."""
from contextlib import nullcontext
import gzip
import json
import time
import torch
import torch.distributed as dist
from .common import append,write
from .workloads import inputs
from .attribution_markers import Markers,instrument_ep,instrument_native_foil

def generation(a,ep,ids,cap,markers=None):
    if ep is not None:dist.barrier()
    torch.cuda.synchronize()
    before=time.time_ns();start=time.perf_counter_ns();after=time.time_ns()
    absolute=(before+after)//2
    if markers is not None and markers.gpu_events:markers.origin=markers.event()
    with markers.region('input_staging') if markers is not None else nullcontext():prefix=ids.clone()
    tokens=[];times=[]
    for step in range(cap):
        if markers is not None:markers.step=step
        with markers.region('model_other') if markers is not None else nullcontext():
            logits=a.forward(prefix)
        with markers.region('sampling_output') if markers is not None else nullcontext():
            token=logits[:,-1].argmax(-1,keepdim=True)
            torch.cuda.synchronize()
            tokens.append(token.flatten().cpu().tolist())
        end=time.perf_counter_ns();end_check=time.time_ns();times.append((end-start)/1e6)
        if step==cap-1 and markers is not None and markers.gpu_events:markers.endpoint=markers.event()
        with markers.region('input_update') if markers is not None else nullcontext():
            prefix=torch.cat([prefix,token],dim=-1)
        del logits
    close=end_check-(absolute+end-start)
    return dict(e2e_ms=times[-1],ttft_ms=times[0],tpot_ms=(times[-1]-times[0])/(cap-1),
        generated_ids=tokens,timestamps_ms=times,start_realtime_ns=absolute,end_realtime_ns=absolute+end-start,
        origin_uncertainty_ns=(after-before)/2,clock_closure_ns=close,output_cap=cap,eos_enabled=False,
        contract='Full-prefix greedy rollout. CUDA-only observer uses this exact timed interval.')

def measure(a,ep,target,config):
    rank=ep.rank if ep else 0
    cells=[]
    for b in config.get('batches',[1,2,4,8,16]):
        for length in config.get('lengths',[128,512,2048]):
            if [b,length] in config.get('skip_cells',[]):continue
            ids,requests=inputs(a,b,length)
            for policy in config.get('policies',['static']):
                if ep is not None:ep.set_policy(policy);ep.request_ids=[r['id'] for r in requests]
                warm=generation(a,ep,ids,32)
                clean=[]
                for repeat in range(3):
                    value=generation(a,ep,ids,32);clean.append(value['e2e_ms'])
                    assert value['generated_ids']==warm['generated_ids']
                    append(target/f'clean.rank{rank}.jsonl',dict(value,batch=b,prefix=length,policy=policy,repeat=repeat))
                traced=[];alignment=[]
                for repeat in range(config.get('trace_repeats',1)):
                    markers=None;old_ep=None;native_patch={};observer=None
                    if rank==0:
                        markers=Markers(gpu_events=config.get('attribution_mode','events')=='events')
                        markers.install_modules(a,detail=config.get('detail','fine'))
                        if ep is not None:old_ep=instrument_ep(ep,markers,detail=config.get('detail','fine'))
                        elif a.name=='foil1':native_patch=instrument_native_foil(a,markers)
                        markers.enabled=True
                        if not markers.gpu_events:
                            observer=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False,profile_memory=False,with_stack=False)
                            observer.start()
                    try:
                        value=generation(a,ep,ids,32,markers=markers)
                        assert value['generated_ids']==warm['generated_ids'],'Attribution changed greedy token IDs'
                        traced.append(value['e2e_ms'])
                        if rank==0:
                            stem=f'attribution_b{b}_p{length}_{policy}_rep{repeat}'
                            if markers.gpu_events:
                                event_data=markers.resolve_events()
                                from .attribution_partition import partition,split_phases
                                event_data['exclusive_components_ms']=partition(event_data['intervals'],event_data['window_ms'])
                                event_data['phase_components']=split_phases(event_data['intervals'],event_data['window_ms'],
                                    event_data['first_output_complete_ms'])
                                event_data['host_e2e_ms']=value['e2e_ms']
                                event_data['event_to_host_ratio']=event_data['window_ms']/value['e2e_ms']
                                alignment.append(event_data['event_to_host_ratio'])
                                with gzip.open(target/f'{stem}.events.json.gz','wt') as output:json.dump(event_data,output)
                                with gzip.open(target/f'{stem}.counts.json.gz','wt') as output:json.dump(markers.count_records,output)
                            else:
                                observer.stop()
                                path=target/f'{stem}.json'
                                observer.export_chrome_trace(str(path))
                                with path.open('rb') as source,gzip.open(str(path)+'.gz','wb') as output:
                                    while data:=source.read(8<<20):output.write(data)
                                path.unlink()  # Exact compressed copy is retained.
                            write(target/f'{stem}.markers.json',dict(pid=markers.pid,regions=markers.records,
                                window=value,batch=b,prefix=length,policy=policy,repeat=repeat,
                                scope='Rank 0 endpoint timeline. Collective kernels include rank waits, not just wire transfer.'))
                    finally:
                        if rank==0:
                            if markers is not None:markers.close()
                            if ep is not None and old_ep is not None:ep._run=old_ep
                            for cls,original in native_patch.items():cls.forward=original
                    if ep is not None:dist.barrier()
                if rank==0:
                    ratio=sum(traced)/len(traced)/(sum(clean)/len(clean))
                    cell=dict(batch=b,prefix=length,policy=policy,clean_e2e_ms=clean,traced_e2e_ms=traced,
                        overhead_ratio=ratio,low_overhead_timing_gate=abs(ratio-1)<=.05 and bool(alignment) and all(abs(v-1)<=.01 for v in alignment),
                        event_to_host_ratios=alignment,
                        detail=config.get('detail','fine'),
                        evidence='WHOLE_ROLLOUT_CUDA_EVENT_CRITICAL_STREAM_SPANS' if config.get('attribution_mode','events')=='events' else 'WHOLE_ROLLOUT_CUDA_ONLY_OBSERVER',
                        caveat='Shares always refer to observed E2E. The 5% overhead gate does not prove unbiased clean component shares. Collective waits are not pure wire communication.')
                    cells.append(cell);write(target/'attribution_status.json',cells)
                    print('ATTRIBUTED',a.name,ep.world if ep else 1,b,length,policy,ratio,flush=True)

