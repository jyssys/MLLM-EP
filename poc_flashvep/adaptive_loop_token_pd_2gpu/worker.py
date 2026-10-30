"""One isolated server start. Scheduled arrivals include harness delivery delay."""
import json
import os
import sys
import time
from pathlib import Path
from .common import ROOT,MODEL_PINS,config,redirect,setup,write,append,digest

def install():
    redirect();cfg=config()
    hashes=json.loads(os.environ.get('LTP_SOURCE_HASHES','{}'))
    for source,expected in hashes.items():
        if digest(source)!=expected:raise RuntimeError('Source changed during worker startup: '+source)
    if hashes:
        dest=ROOT/'raw'/cfg['name']/f'loaded_source_{os.getpid()}.json'
        if not dest.exists():write(dest,dict(pid=os.getpid(),hashes=hashes,time=time.time(),visible=os.environ.get('CUDA_VISIBLE_DEVICES')))
    from ..adaptive_recurrence_cut import worker as old
    old.ROOT=ROOT
    old.install()
    from .connector import PartitionConnector,install_token_barrier
    import vllm_rlt.pd.engine as pd
    if cfg['arm']!='PD_G1':pd.PDStageConnector=PartitionConnector
    from vllm_rlt.core.scheduler import Scheduler
    if not getattr(Scheduler,'_ltp_barrier',False):
        Scheduler._ltp_barrier=True;install_token_barrier()
    if cfg.get('debug_progress'):
        from vllm_rlt.engine.stage_engine import StageEngine
        execute=StageEngine._step
        def traced(self):
            try:return execute(self)
            except RuntimeError as error:
                if 'scheduler made no progress' in str(error):
                    write(ROOT/'raw'/cfg['name']/f'debug_frontier_{os.getpid()}.json',dict(
                        role=self.stage_config.role,blocked=list(self._connector_blocked),
                        queues={str(k):list(v) for k,v in self.scheduler.queues.items()},
                        requests={rid:dict(phase=str(r.phase),preluded=r.num_preluded_tokens,
                            prefilled=r.num_prefilled_tokens,boundaries=r.prefill_depth_ranges,
                            export_cut=getattr(r,'arc_export_cut',None),prefix=getattr(r,'ltp_prefix',None),
                            compute_released=r.compute_released) for rid,r in self.scheduler.requests.items()}))
                raise
        StageEngine._step=traced

if os.environ.get('LTP_WORKER'):install()

def serve(engine,inputs,args,run,warmup=False,contract=None):
    from vllm_rlt import SamplingParams
    cfg=contract or config();n=min(len(inputs),cfg.get('warmup_count',2)) if warmup else len(inputs)
    inputs=inputs[:n]
    offsets=([0.]*n if warmup else cfg['arrival_offsets'])
    caps=([cfg.get('warmup_cap',2)]*n if warmup else cfg['request_caps'])
    reference={}
    if not warmup and cfg.get('phase')=='primary' and cfg['arm']!='PD_G1':
        path=ROOT/'raw'/f'primary_{cfg["model"]}_{cfg["workload"].lower()}_pd_g1_start0'/'requests.jsonl'
        reference={r['sample_id']:r for r in (json.loads(s) for s in path.read_text().splitlines()) if not r['warmup']}
    start=time.monotonic();active={};first={};last={};admitted={};events={};out={};index=0;observation=start
    while index<n or active:
        while index<n and time.monotonic()>=start+offsets[index]:
            rid=f'r{run}_q{index}'
            if hasattr(engine,'ltp_choose'):engine.ltp_choose(rid,inputs[index])
            params=SamplingParams(max_tokens=caps[index],temperature=0.,seed=cfg['seed']+index,
                min_loops=cfg['depth'],max_loops=cfg['depth'],exit_threshold=1.,ignore_eos=True)
            engine.add_request(rid,inputs[index],params)
            active[rid]=index;admitted[rid]=time.monotonic();events[rid]=[];index+=1
        for output in engine.step():
            now=time.monotonic();rid=output.request_id;i=active[rid];count=len(output.token_ids)
            previous=last.get(rid,(now,0))
            if count>previous[1]:
                first.setdefault(rid,now)
                events[rid].append(dict(time=now,previous_time=previous[0],new_tokens=count-previous[1],first=previous[1]==0))
                last[rid]=(now,count)
            if not output.finished:continue
            if count!=caps[i] or any(d!=cfg['depth'] for d in output.exit_depths):
                raise RuntimeError('Controlled output length or fixed recurrent-depth mismatch')
            active.pop(rid)
            row=dict(model=cfg['model'],arm=cfg['arm'],name=cfg['name'],workload=cfg.get('workload','gate'),cohort=cfg.get('cohort'),
                run=run,warmup=warmup,sample_id=i,prompt_id=cfg['prompt_ids'][i],seed=cfg['seed']+i,
                input_ids=inputs[i],output_ids=list(output.token_ids),exit_depths=list(output.exit_depths),
                finish_reason=str(output.finish_reason),input_tokens=len(inputs[i]),output_tokens=count,
                arrival=start+offsets[i],admission=admitted[rid],first_token=first.get(rid,now),finish=now,
                ttft_ms=1000*(first.get(rid,now)-start-offsets[i]),
                tpot_ms=1000*(now-first.get(rid,now))/max(1,count-1),
                e2e_ms=1000*(now-start-offsets[i]),token_events=events[rid],
                gpus=cfg['gpus'],chunk_tokens=cfg['chunk'],recurrent_depth=cfg['depth'],
                co_batching=False,generation='CONTROLLED_IGNORE_EOS',evidence='MEASURED_HEAVY_CORRECTNESS' if cfg.get('state_probe') else 'MEASURED_HEAVY_PROFILE' if cfg.get('profile') else 'MEASURED',
                timing_scope='Scheduled open-loop arrival to final output. No HTTP transport.',
                transfer_protocol='native_chunk_streaming' if cfg['arm']=='PD_G1' else 'snapshot' if cfg['arm'] not in ('DP','DP_Chunked') else 'none')
            append(ROOT/'raw'/cfg['name']/'requests.jsonl',row);out[i]=row
            print('DONE',cfg['name'],i,count,round(row['e2e_ms'],2),flush=True)
            if reference:
                expected=reference.get(i)
                exact=bool(expected and all(expected[k]==row[k] for k in ('input_ids','output_ids','exit_depths','finish_reason')))
                if not exact:
                    divergence=next((j for j,(x,y) in enumerate(zip(expected['output_ids'],row['output_ids'])) if x!=y),None) if expected else None
                    write(ROOT/'raw'/cfg['name']/'live_output_mismatch.json',dict(sample_id=i,prompt_id=row['prompt_id'],
                        first_divergent_token=divergence,reference_ids=expected.get('output_ids') if expected else None,actual_ids=row['output_ids']))
                    raise RuntimeError('Unexplained controlled output mismatch. Further serving expansion is forbidden.')
        if not warmup and time.monotonic()-observation>=1.:
            observation=time.monotonic()
            phases={}
            if hasattr(engine,'transfers'):
                for pending in engine.transfers.values():phases[pending.phase]=phases.get(pending.phase,0)+1
            append(ROOT/'raw'/cfg['name']/'queue_observations.jsonl',dict(time=observation,
                since_start=observation-start,cohort=cfg.get('cohort'),workload=cfg.get('workload'),
                delivered=index,completed=len(out),outstanding=len(active),pending_phases=phases,
                worker_credits={name:dict(slots=p.slots,reserved_blocks=p.reserved_blocks) for name,p in getattr(engine,'peers',{}).items()}))
        if time.monotonic()-start>cfg.get('timeout',3600):raise TimeoutError('Open-loop cohort watchdog')
        if not active and index<n:time.sleep(min(.001,max(0.,start+offsets[index]-time.monotonic())))
    if not warmup:
        name=f'window_{cfg["cohort"]}.json' if cfg.get('cohort') is not None else 'window.json'
        write(ROOT/'raw'/cfg['name']/name,dict(start=start,end=time.monotonic(),last_arrival=start+offsets[-1]))
    return out

def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());install()
    from ..looped_pd_abc import worker as base
    base.ROOT=ROOT;base.setup=setup;base.serve=serve
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    base.prompts=lambda model,count,length:[r['input_ids'][model] for r in data[:count]]
    configure=base.configuration
    def configuration(model,width,chunk,graphs=False,asynchronous=False):
        from dataclasses import replace
        result=configure(model,width,chunk,graphs,asynchronous)
        result['scheduler_config']=replace(result['scheduler_config'],max_num_batched_tokens=cfg.get('token_budget',chunk))
        # Explicit common pool capacity. Huginn admits fewer long requests than Ouro.
        if cfg.get('kv_blocks'):result['cache_config']=replace(result['cache_config'],num_blocks=cfg['kv_blocks'])
        return result
    base.configuration=configuration
    if cfg.get('tune_jobs'):
        def tuning(engine,inputs,args,run,warmup=False):
            if warmup:return serve(engine,inputs,args,run,True)
            final={}
            for i,job in enumerate(cfg['tune_jobs']):
                contract=dict(cfg,**job)
                rows=json.loads((ROOT/'data'/job['dataset']).read_text())[:job['count']]
                local=[r['input_ids'][cfg['model']] for r in rows]
                engine.ltp_forced_plan=(job['cut'],job.get('prefix_fraction',1.))
                engine.ltp_update_contract(contract)
                stats=lambda:{name:peer.client.stats() for name,peer in engine.peers.items()}
                before=stats();final=serve(engine,local,args,i,contract=contract)
                while engine.has_unfinished_requests():engine.step()
                write(ROOT/'raw'/cfg['name']/f'cohort_{i}_stats.json',dict(before=before,after=stats(),contract=contract))
            return final
        base.serve=tuning
    if cfg['arm'] not in ('DP','DP_Chunked'):
        import vllm_rlt.pd.engine as pd
        initialize=pd.PDEngine.__init__
        def init(self,*args,**kwargs):
            original_config=kwargs['pd_config']
            from dataclasses import replace
            kwargs['pd_config']=replace(original_config,request_timeout=cfg.get('timeout',3600))
            initialize(self,*args,**kwargs)
            from .policy import install as policy
            from ..adaptive_recurrence_cut.worker import install_local_endpoint
            policy(self,cfg);install_local_endpoint(self)
        pd.PDEngine.__init__=init
    write(ROOT/'raw'/cfg['name']/'arc_contract.json',cfg)
    mode='dp' if cfg['arm'] in ('DP','DP_Chunked') else 'pd'
    flags=['worker','--model',cfg['model'],'--mode',mode,'--gpus',','.join(map(str,cfg['gpus'])),
        '--name',cfg['name'],'--requests',str(cfg['count']),'--length','0','--cap',str(max(cfg['request_caps'])),
        '--chunk',str(cfg['chunk']),'--concurrency',str(cfg['capacity']),'--timeout',str(cfg.get('timeout',3600)),
        '--warmup']
    if cfg.get('wavefront',True):flags+=['--wavefront']
    if cfg.get('graphs',True):flags+=['--graphs']
    if cfg.get('profile'):flags+=['--profile']
    if cfg['model']=='ouro' and cfg.get('asynchronous',True):flags+=['--asynchronous']
    os.environ['ABC_ALLOW_BOUNDARY_PD']='1';os.environ['ABC_ALLOW_STAGED_BOUNDARY']='1'
    sys.argv=flags;base.main()

if __name__=='__main__':main()
