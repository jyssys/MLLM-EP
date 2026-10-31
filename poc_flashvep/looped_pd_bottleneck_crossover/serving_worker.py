"""Qualified PD execution with reasoned admission and submission timestamps."""
import sys
import collections
from .common import *

def install():
    redirect()
    from ..adaptive_loop_token_pd_2gpu import worker as prior
    prior.install()
    from vllm_rlt.worker.executor import Executor
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import RequestPhase
    from vllm_rlt.pd.engine import PDEngine
    if getattr(Executor,'_bc_host_accounting',False):return
    Executor._bc_host_accounting=True
    first=set();frontiers=[];counts=collections.Counter();CPU=collections.defaultdict(float);snapshots=[]
    last_sample=[0.];schedule=Scheduler.schedule;submit=Executor.submit;close=StageEngine.close
    def schedule_host(self,*a,**kw):
        t=time.perf_counter();cpu=time.process_time();batch=schedule(self,*a,**kw);CPU['scheduler_s']+=time.perf_counter()-t
        CPU['process_CPU_during_schedule_s']+=time.process_time()-cpu
        counts['schedule_calls']+=1;now=time.monotonic()
        if now-last_sample[0]>.5:
            last_sample[0]=now
            phases=collections.Counter(str(r.phase) for r in self.requests.values())
            snapshots.append(dict(time=now,phases=dict(phases),queued={str(k):len(v) for k,v in self.queues.items()}))
        return batch
    def submit_host(self,batch):
        now=time.monotonic();counts['submissions']+=1;counts['rows']+=batch.num_tokens
        phase=batch.phase.value if batch.phase is not None else 'mixed'
        counts[f'phase_{phase}']+=1
        counts[f'rows_phase_{phase}']+=batch.num_tokens
        for item in batch.requests:
            r=item.request;p=batch.phase_for(item);rid=r.request_id
            if p in (RequestPhase.PREFILL,RequestPhase.PREFILLING_PRELUDE,RequestPhase.PREFILLING_RECURRENT):
                if (rid,'any') not in first:
                    first.add((rid,'any'));frontiers.append(dict(event='P_FIRST_SUBMISSION',time=now,request_id=rid,
                        prompt_tokens=len(r.prompt_token_ids),seed=r.sampling_params.seed))
                if p==RequestPhase.PREFILLING_RECURRENT and (rid,'core') not in first:
                    first.add((rid,'core'));frontiers.append(dict(event='P_FIRST_RECURRENT_SUBMISSION',time=now,request_id=rid))
                if p==RequestPhase.PREFILLING_PRELUDE and item.token_start+item.token_count==len(r.prompt_token_ids):
                    frontiers.append(dict(event='P_PRELUDE_FINAL_SUBMISSION',time=now,request_id=rid))
                if getattr(item,'completes_prompt',False):
                    frontiers.append(dict(event='P_FINAL_SUBMISSION',time=now,request_id=rid,
                        depth=getattr(item,'depth',None)))
        t=time.perf_counter();cpu=time.process_time();result=submit(self,batch);CPU['submit_s']+=time.perf_counter()-t
        CPU['process_CPU_during_submit_s']+=time.process_time()-cpu
        return result
    def close_host(self):
        self.model_runner.synchronize()
        folder=ROOT/'raw'/config()['name']
        write(folder/f'host_execution_{os.getpid()}.json',dict(role=self.stage_config.role,CPU=dict(CPU),counts=dict(counts),
            frontiers=frontiers,queue_samples=snapshots,
            scope='Host wall spans and process CPU accumulated during those spans. Process CPU includes helper threads. Values are not additive E2E components or exclusive operator CPU time.'))
        return close(self)
    Executor.submit=submit_host;Scheduler.schedule=schedule_host;StageEngine.close=close_host
    admit=PDEngine._admit;last_admission=[0.]
    def admit_host(self):
        now=time.monotonic()
        if now-last_admission[0]>.25:
            last_admission[0]=now;waiting=[]
            for w in self.transfers.values():
                if w.phase!='waiting':continue
                reasons=[]
                for role in ('prefill','decode'):
                    length=len(w.request.prompt_token_ids)+(w.request.sampling_params.max_tokens-1 if role=='decode' else 0)
                    peers=[p for p in self.peers.values() if p.role==role]
                    if not any(p.slots<p.client.info['transfer_slots'] for p in peers):reasons.append(role.upper()+'_NO_SLOT')
                    if not any(p.reserved_blocks+self._blocks(p,length)<=p.client.info['num_blocks']-(p.client.info['watermark_blocks'] if p.slots else 0) for p in peers):
                        reasons.append(role.upper()+'_NO_KV_CREDIT')
                waiting.append(dict(request_id=w.request.request_id,age_s=now-w.created,reasons=reasons))
            append(ROOT/'raw'/config()['name']/'admission_reasons.jsonl',dict(time=now,waiting=waiting,
                peers={n:dict(role=p.role,slots=p.slots,reserved_blocks=p.reserved_blocks,
                    total_blocks=p.client.info['num_blocks'],watermark_blocks=p.client.info['watermark_blocks']) for n,p in self.peers.items()}))
        return admit(self)
    PDEngine._admit=admit_host
    start_profile=PDEngine.start_profile
    def bounded_profile(self,profile_config,*args,**kwargs):
        from dataclasses import replace
        profile_config=replace(profile_config,record_shapes=config().get('profile_record_shapes',profile_config.record_shapes))
        schedule=config().get('profile_schedule')
        if schedule:
            profile_config=replace(profile_config,wait=schedule.get('wait',200),warmup=10,active=schedule.get('active',200),repeat=1)
            kwargs['scheduled']=True
        return start_profile(self,profile_config,*args,**kwargs)
    PDEngine.start_profile=bounded_profile
    if config().get('prepare_D_only'):
        from vllm_rlt.pd.connector import PDStageConnector
        can_schedule=PDStageConnector.can_schedule
        def ready_cohort(self,request):
            if self.engine.stage_config.role!='decode' or getattr(self,'_bc_D_released',False):return can_schedule(self,request)
            marker=ROOT/'raw'/config()['name']/'prepared_D_marker.json'
            if not marker.exists():return can_schedule(self,request)
            expected=json.loads(marker.read_text())['count']
            incoming=[t for t in self.transfers.values() if not t.outgoing and not t.cancelled]
            if len(incoming)!=expected or not all(t.ready for t in incoming):return False
            self.engine.model_runner.synchronize()
            assert all(t.request.num_prefilled_tokens==len(t.request.prompt_token_ids) for t in incoming)
            self._bc_D_released=True
            write(marker.with_name(f'prepared_D_ready_{os.getpid()}.json'),dict(time=time.monotonic(),count=expected,
                valid_KV_and_hidden=True,transfer_ids=[t.tid for t in incoming],
                scope='All incoming NIXL transfers completed and native imports accepted. Device synchronized before timed Decode.'))
            return can_schedule(self,request)
        PDStageConnector.can_schedule=ready_cohort

if os.environ.get('BC_WORKER')=='1':install()

def main():
    install()
    cfg=config();source=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    assert len(source)>=cfg['count'] and len(cfg['request_caps'])==cfg['count'],'Missing offered requests'
    assert all(len(source[i]['input_ids'][cfg['model']])+cfg['request_caps'][i]<=4096 for i in range(cfg['count'])),'Context limit exceeded'
    from ..adaptive_loop_token_pd_2gpu import worker as prior
    if cfg.get('calibration_serial'):
        original_serve=prior.serve
        def serial_calibration(engine,inputs,args,run,warmup=False,contract=None):
            if warmup:return original_serve(engine,inputs,args,run,True,contract)
            values={};start=time.monotonic()
            for i,tokens in enumerate(inputs):
                local=dict(cfg,count=1,seed=cfg['seed']+i,arrival_offsets=[0.],request_caps=[cfg['request_caps'][i]],
                    prompt_ids=[cfg['prompt_ids'][i]],cohort=i)
                output=original_serve(engine,[tokens],args,i,False,local)
                values[i]=output[0]
                while engine.has_unfinished_requests():engine.step()
            write(ROOT/'raw'/cfg['name']/'window.json',dict(start=start,end=time.monotonic(),last_arrival=values[len(inputs)-1]['arrival'],
                scope='Closed-loop serial calibration. Not an open-loop capacity run.'))
            return values
        prior.serve=serial_calibration
    if config().get('prepare_D_only'):
        original_serve=prior.serve
        def prepared_decode(engine,inputs,args,run,warmup=False,contract=None):
            if warmup:return original_serve(engine,inputs,args,run,True,contract)
            cfg=contract or config();original_send=engine._send;held=[];release=[]
            prepare_start=time.monotonic()
            write(ROOT/'raw'/cfg['name']/'prepared_D_marker.json',dict(count=cfg['count'],time=prepare_start))
            def hold_commit(destination,kind,**fields):
                if kind!='commit':return original_send(destination,kind,**fields)
                held.append((destination,dict(fields)))
                if len(held)==cfg['count']:
                    release.append(time.monotonic())
                    for target,payload in held:original_send(target,'commit',**payload)
            engine._send=hold_commit
            try:rows=original_serve(engine,inputs,args,run,False,contract)
            finally:engine._send=original_send
            assert len(held)==cfg['count'] and len(release)==1,'Prepared D cohort did not retain all source lifetimes'
            ready=[json.loads(p.read_text()) for p in (ROOT/'raw'/cfg['name']).glob('prepared_D_ready_*.json')]
            assert len(ready)==1 and ready[0]['valid_KV_and_hidden'],'Timed Decode started without a fully valid prepared cohort'
            actual_release=ready[0]['time']
            finish=max(r['finish'] for r in rows.values())
            ref_run=cfg.get('prepared_reference_run') or ('phase_a_ouro_batch8_start0' if cfg['model']=='ouro' else 'phase_a_huginn_grid_start0')
            refs=[json.loads(s) for s in (ROOT/'raw'/ref_run/'references.jsonl').read_text().splitlines()]
            candidates=[r for r in refs if r['job']['dataset']==cfg['dataset'] and r['job']['cap']==cfg['request_caps'][0] and r['job']['batch']>=cfg['count']]
            assert candidates,'No matched native reference for prepared PD Decode'
            reference=min(candidates,key=lambda r:r['job']['batch'])
            expected={int(k.rsplit('_q',1)[1]):v for k,v in reference['outputs'].items()}
            assert all(r['output_ids']==expected[i]['output_ids'] and r['exit_depths']==expected[i]['depths'] for i,r in rows.items()),'Prepared native PD D-only output parity failed'
            write(ROOT/'raw'/cfg['name']/'D_only_window.json',dict(prepare_start=prepare_start,commit_release=release[0],release=actual_release,finish=finish,
                preparation_s=actual_release-prepare_start,prepared_D_service_s=finish-actual_release,batch=len(rows),
                mu_D_finite=len(rows)/(finish-actual_release),
                native_output_parity=True,reference_run=ref_run,
                scope='Native PD coordinator, first CODA and full Decode. All P computation and NIXL import excluded. Valid KV/hidden preserved. Native ACK manages source lifetime.',
                evidence='MEASURED_PREPARED_NATIVE_PD_D_ONLY'))
            return rows
        prior.serve=prepared_decode
    prior.main()

if __name__=='__main__':main()
