"""Run the original engine with legal barriers before and after first sampling."""
import gc
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from .common import *

class LocalBarrier:
    """Keep actual KV and state resident. Do not synthesize or reuse activations."""
    def __init__(self):self.mode='prefill'
    def exports(self,request):return False
    def can_schedule(self,request):
        from vllm_rlt.request import RequestPhase
        if self.mode=='prefill':return request.phase!=RequestPhase.CODA
        if self.mode=='first_token':return request.num_scheduled_outputs==0
        return True
    def prefill_submitted(self,*args,**kwargs):raise RuntimeError('The local barrier must not export request state')

def install():
    redirect()
    os.environ.update(LTP_WORKER='1',ABC_ALLOW_BOUNDARY_PD='1',ABC_ALLOW_STAGED_BOUNDARY='1',
        ABC_RUN_DIR=str(ROOT/'raw'/config()['name']))
    from ..adaptive_loop_token_pd_2gpu import worker as qualified
    qualified.install()
    from .telemetry import install as measure
    measure()

def drain(engine):
    engine.model_runner.synchronize()
    limit=time.monotonic()+20
    while getattr(engine,'_inflight',None):
        assert time.monotonic()<limit,'Inflight drain watchdog'
        outputs=engine.step()
        assert not outputs,'A barrier emitted an unexpected output while draining'
    engine.model_runner.synchronize()

def cohort(engine,tokens,cap,seed,tag,barrier,recorder,use_barrier=None):
    import torch
    from vllm_rlt import SamplingParams
    from vllm_rlt.request import RequestPhase
    cfg=config();ids=[f'{tag}_q{i}' for i in range(len(tokens))]
    starts={};events={rid:[] for rid in ids};outputs={};last={};first={};depth=cfg['depth']
    for i,(rid,values) in enumerate(zip(ids,tokens)):
        engine.add_request(rid,values,SamplingParams(max_tokens=cap,temperature=0.,seed=seed+i,
            min_loops=depth,max_loops=depth,exit_threshold=1.,ignore_eos=True))
    staged=cfg.get('barrier',True) if use_barrier is None else use_barrier
    barrier.mode='prefill' if staged else 'decode'
    recorder.phase('prefill');before=engine.stats();p_start=time.monotonic();limit=p_start+cfg.get('timeout',1800)
    def step():
        assert time.monotonic()<limit,'Capacity cohort watchdog'
        for out in engine.step():
            now=time.monotonic();rid=out.request_id
            if len(out.token_ids)>last.get(rid,0):
                first.setdefault(rid,now);events[rid].append(dict(time=now,count=len(out.token_ids)))
                last[rid]=len(out.token_ids)
            if out.finished:outputs[rid]=dict(output_ids=list(out.token_ids),depths=list(out.exit_depths),finish_reason=str(out.finish_reason))
    if not staged:
        while engine.has_unfinished_requests():step()
        engine.model_runner.synchronize()
        observed=engine.stats()['issued_loop_rows']-before['issued_loop_rows']
        required=(sum(map(len,tokens))+len(ids)*(cap-1))*depth
        assert observed==required,'Native full-depth accounting'
        assert len(outputs)==len(ids) and all(len(v['output_ids'])==cap and set(v['depths'])=={depth} for v in outputs.values())
        return dict(tag=tag,outputs=outputs,events=events,normal_reference=True,elapsed_s=time.monotonic()-p_start,
            input_ids=tokens,full_depth_rows=required,full_depth_rows_observed=observed)
    phase_range=torch.profiler.record_function('bc_phase_prefill') if os.environ.get('BC_ACTIVE_PROFILE')=='1' else nullcontext()
    phase_range.__enter__()
    while not all(engine.scheduler.requests[rid].phase==RequestPhase.CODA for rid in ids):step()
    drain(engine);p_end=time.monotonic();phase_range.__exit__(None,None,None)
    p_stats=engine.stats();p_measure=recorder.summary()
    assert p_stats['issued_loop_rows']-before['issued_loop_rows']==sum(map(len,tokens))*depth,'Prefill full-depth accounting'
    assert all(engine.cache_manager.has_allocation(rid) for rid in ids)
    # The same checkpoint state remains resident for first sampling and Decode.
    barrier.mode='first_token';recorder.phase('first_token');coda_start=time.monotonic()
    phase_range=torch.profiler.record_function('bc_phase_first_token') if os.environ.get('BC_ACTIVE_PROFILE')=='1' else nullcontext()
    phase_range.__enter__()
    while len(first)<len(ids):step()
    drain(engine);coda_end=time.monotonic();phase_range.__exit__(None,None,None);coda_measure=recorder.summary()
    assert all(last[rid]==1 for rid in ids),'Decode ran before the prepared-cache timer'
    barrier.mode='decode';recorder.phase('decode');d_start=time.monotonic()
    phase_range=torch.profiler.record_function('bc_phase_decode') if os.environ.get('BC_ACTIVE_PROFILE')=='1' else nullcontext()
    phase_range.__enter__()
    while engine.has_unfinished_requests():step()
    engine.model_runner.synchronize();d_end=time.monotonic();phase_range.__exit__(None,None,None)
    d_stats=engine.stats();d_measure=recorder.summary()
    required=(sum(map(len,tokens))+len(ids)*(cap-1))*depth
    assert d_stats['issued_loop_rows']-before['issued_loop_rows']==required,'Total fixed-depth accounting'
    assert len(outputs)==len(ids) and all(len(v['output_ids'])==cap and set(v['depths'])=={depth} for v in outputs.values())
    itls=[b['time']-a['time'] for values in events.values() for a,b in zip(values[1:],values[2:])]
    return dict(tag=tag,batch=len(ids),input_lengths=list(map(len,tokens)),output_target=cap,
        input_ids=tokens,outputs=outputs,token_events=events,
        prefill_host_service_s=p_end-p_start,first_coda_host_service_s=coda_end-coda_start,
        prefill_plus_first_token_service_s=(p_end-p_start)+(coda_end-coda_start),
        decode_rest_service_s=d_end-d_start,decode_native_boundary_service_s=(coda_end-coda_start)+(d_end-d_start),
        intervals=dict(prefill=[p_start,p_end],first_token=[coda_start,coda_end],decode=[d_start,d_end]),
        timed_decode_tokens_per_request=cap-1,decode_ITL_s=itls,
        prefill_measure=p_measure,coda_measure=coda_measure,decode_measure=d_measure,
        full_depth_rows=required,full_depth_rows_observed=d_stats['issued_loop_rows']-before['issued_loop_rows'],
        KV_preserved_at_boundary=True,weight_precision='BF16',KV_precision='BF16',
        timing_scope='P excludes first CODA. D-rest excludes prefill and first token. Native-boundary D includes first CODA. CUDA spans are not kernel busy.')

def main():
    import torch
    cfg=json.loads(Path(sys.argv[1]).read_text());setup();folder=ROOT/'raw'/cfg['name']
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    install();torch.set_num_threads(1);torch.cuda.set_device(0)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[cfg['gpus'][0]].removeprefix('GPU-')
    from vllm_rlt.models import AutoModelForCausalLM
    from vllm_rlt.engine.stage_engine import StageEngine
    from ..looped_pd_abc.worker import configuration
    from ..looped_pd_abc.common import MODELS
    from ..looped_pd_abc.nvml_watch import GPUWatch
    from . import telemetry
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start()
    engine=None;cache=None;barrier=None
    try:
        model=AutoModelForCausalLM.from_pretrained(MODELS[cfg['model']],device='cuda:0',dtype=torch.bfloat16)
        for group_index,group in enumerate(cfg['groups']):
            if engine is not None:
                engine.close();engine=None;cache=None;barrier=None
                gc.collect();torch.cuda.empty_cache()
            write(folder/f'group_memory_before_{group_index}.json',dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),device_free_bytes=torch.cuda.mem_get_info()[0],
                scope='After the previous engine and all local cache references are released. Before the new pool allocation.'))
            conf=configuration(cfg['model'],cfg.get('slots',4),group['chunk'],cfg.get('graphs',True),cfg['model']=='ouro')
            conf['scheduler_config']=replace(conf['scheduler_config'],max_num_batched_tokens=group['chunk'],wavefront_chunked_prefill=True)
            conf['cache_config']=replace(conf['cache_config'],num_blocks=group['kv_blocks'])
            barrier=LocalBarrier();engine=StageEngine(model,connector=barrier,**conf)
            cache=engine.cache_manager
            write(folder/f'geometry_{group_index}.json',dict(chunk=group['chunk'],slots=cfg.get('slots',4),KV_blocks=cache.num_blocks,
                KV_bytes=cache.key_cache.numel()*cache.key_cache.element_size()*2,KV_block_bytes=cache.bytes_per_block,
                layers=cache.num_layers,storage_depths=cache.storage_depths,
                block_size=cache.block_size,KV_heads=cache.num_kv_heads,head_dim=cache.head_dim,
                recurrent_kv_layers=list(getattr(model,'recurrent_kv_layers',range(cache.num_layers))),
                coda_kv_layers=list(getattr(model,'coda_kv_layers',())),
                physical_GPU=cfg['gpus'][0],uuid=str(torch.cuda.get_device_properties(0).uuid),
                unique_model_parameters=sum(p.numel() for p in model.parameters()),weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters())))
            if cfg.get('linear_geometry_probe'):
                from .projection_probe import run_probe
                run_probe(engine,model,barrier,folder,cfg)
            for index,job in enumerate(group['jobs']):
                job_seed=cfg['seed']+cfg.get('job_seed_stride',0)*index
                data=json.loads((ROOT/'data'/job['dataset']).read_text())
                batch=job['batch'];tokens=[r['input_ids'][cfg['model']] for r in data[:batch]]
                assert len(tokens)==batch and all(len(v)+job['cap']<=4096 for v in tokens),'Controlled context contract'
                required=sum(cache.required_blocks(len(v)+job['cap']-1) for v in tokens)
                tag=f'g{group_index}_j{index}'
                if cfg.get('native_reference_only'):
                    warm=telemetry.Recorder(enabled=False);telemetry.ACTIVE=None
                    cohort(engine,tokens[:2],2,cfg['seed']-10000,tag+'_warm',barrier,warm,use_barrier=False)
                    result=cohort(engine,tokens,job['cap'],job_seed,tag+'_reference',barrier,warm,use_barrier=False)
                    append(folder/'references.jsonl',dict(job=job,**result));continue
                if required>cache.num_blocks or batch>cfg.get('slots',4):
                    append(folder/'cohorts.jsonl',dict(tag=tag,job=job,status='BLOCKED_KV_CAPACITY',required_blocks=required,available_blocks=cache.num_blocks));continue
                warm=telemetry.Recorder(enabled=False);telemetry.ACTIVE=None
                cohort(engine,tokens,min(2,job['cap']),cfg['seed']-10000,tag+'_warm',barrier,warm)
                reference=None
                if cfg.get('verify_native',True):
                    telemetry.ACTIVE=None
                    reference=cohort(engine,tokens,job['cap'],job_seed,tag+'_reference',barrier,warm,use_barrier=False)
                    append(folder/'references.jsonl',dict(job=job,chunk=group['chunk'],KV_blocks=group['kv_blocks'],**reference))
                for repeat in range(job.get('repeats',1)):
                    recorder=telemetry.Recorder(enabled=job.get('instrument',True));telemetry.ACTIVE=recorder if job.get('instrument',True) else None
                    # Repeated service samples use the same seeded inputs as the native reference.
                    os.environ['ABC_MEASURED_ACTIVE']='1'
                    if job.get('profile'):
                        from vllm_rlt.profiling import ProfileConfig
                        os.environ['BC_ACTIVE_PROFILE']='1'
                        engine.start_profile(ProfileConfig(output_dir=str(folder/'profile'/tag),record_shapes=True))
                    result=cohort(engine,tokens,job['cap'],job_seed,tag+f'_r{repeat}',barrier,recorder)
                    if job.get('profile'):
                        engine.stop_profile();engine.wait_for_profile_artifacts(180)
                        os.environ.pop('BC_ACTIVE_PROFILE',None)
                    os.environ['ABC_MEASURED_ACTIVE']='0'
                    if reference is not None:
                        a={k.rsplit('_q',1)[1]:v for k,v in reference['outputs'].items()}
                        b={k.rsplit('_q',1)[1]:v for k,v in result['outputs'].items()}
                        if a!=b:
                            write(folder/f'parity_failure_{tag}_r{repeat}.json',dict(reference=a,candidate=b,input_ids=tokens))
                            raise RuntimeError('Native versus phase-barrier output parity failed')
                        result['native_output_parity']=True
                    if cfg.get('canonical_reference'):
                        references=[json.loads(s) for s in (ROOT/'raw'/cfg['canonical_reference']/'references.jsonl').read_text().splitlines()]
                        references=[r for r in references if r['job']['dataset']==job['dataset'] and r['job']['cap']==job['cap'] and r['job']['batch']>=batch]
                        assert references,'Missing canonical setting reference'
                        expected={k.rsplit('_q',1)[1]:v for k,v in references[0]['outputs'].items()}
                        actual={k.rsplit('_q',1)[1]:v for k,v in result['outputs'].items()}
                        if any(v!=expected[i] for i,v in actual.items()):
                            write(folder/f'canonical_failure_{tag}_r{repeat}.json',dict(reference=expected,candidate=actual,input_ids=tokens))
                            raise RuntimeError('Expanded engine setting violates canonical output parity')
                        result['canonical_output_parity']=True
                    result.update(job=job,repeat=repeat,model=cfg['model'],chunk=group['chunk'],KV_blocks=group['kv_blocks'],status='COMPLETE',
                        seed=job_seed,evidence='MEASURED_HEAVY_CORRECTNESS' if cfg.get('state_probe') else 'MEASURED_HEAVY_PROFILE' if job.get('profile') else 'MEASURED_EVENT_INSTRUMENTED' if job.get('instrument',True) else 'MEASURED_NO_CUDA_EVENT_INSTRUMENTATION')
                    append(folder/'cohorts.jsonl',result);append(folder/'frontiers.jsonl',dict(tag=result['tag'],events=recorder.frontiers))
                    print('COHORT_COMPLETE',cfg['name'],tag,repeat,flush=True)
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if engine is not None:engine.close()
        monitor.stop()

if __name__=='__main__':main()
