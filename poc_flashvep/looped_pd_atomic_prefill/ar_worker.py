"""Native vLLM/NIXL Full-Prefill PD and independent colocated replicas."""
import argparse
import collections
import socket
import subprocess
import tempfile
from multiprocessing.connection import Listener,Client
from .common import *


def prompt_rows(result,requests,before):
    rows=[]
    for rid,count in result.num_scheduled_tokens.items():
        request=requests.get(rid)
        if request is None:continue
        length=len(request.prompt_token_ids);start=before.get(rid,0)
        if start<length:rows.append((rid,start,min(count,length-start),length))
    return rows


def llm_options(cfg,role):
    slots=cfg['P_slots'] if role=='prefill' else cfg['D_slots']
    options=dict(model=str(MODEL_PATHS['qwen8']),dtype='bfloat16',max_model_len=cfg['length']+cfg['cap'],
        max_num_seqs=slots,max_num_batched_tokens=cfg['token_budget'],enable_prefix_caching=False,
        enable_chunked_prefill=True,async_scheduling=False,gpu_memory_utilization=cfg.get('gpu_memory_utilization',.80),
        seed=cfg['seed'],compilation_config={'cudagraph_capture_sizes':[n for n in (1,2,4,8,16,32) if n<=slots]})
    if role=='prefill' and cfg['chunk']<cfg['length']:
        # vLLM 0.26 rejects nondefault Concurrent Partial Prefill options.
        # Keep its supported scheduler and use only the per-request threshold.
        options.update(long_prefill_token_threshold=cfg['chunk'])
    if role in ('prefill','decode'):
        options['kv_transfer_config']=dict(kv_connector='NixlConnector',kv_role='kv_producer' if role=='prefill' else 'kv_consumer')
    return options


def stage(cfg,role,address):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_NIXL_SIDE_CHANNEL_HOST']='127.0.0.1'
    os.environ['VLLM_NIXL_SIDE_CHANNEL_PORT']=os.environ['APD_SIDE_PORT']
    import torch
    from vllm import LLM,SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    torch.set_num_threads(1);channel=Client(address,family='AF_UNIX')
    spans=[];samples=[];step_spans=[];native_stats=[];done_events=[];allocation_refusals=[];clock=[0.]
    allocate=KVCacheManager.allocate_slots
    def observed_allocate(self,request,*a,**kw):
        value=allocate(self,request,*a,**kw)
        if value is None:allocation_refusals.append(dict(time=time.monotonic(),request_id=request.request_id,
            free_blocks=self.block_pool.get_num_free_blocks(),reason='KV_ALLOCATION_RETURNED_NONE'))
        return value
    KVCacheManager.allocate_slots=observed_allocate
    schedule=Scheduler.schedule
    def observed_schedule(self,*a,**kw):
        before={rid:r.num_computed_tokens for rid,r in self.requests.items()} if role=='prefill' else {}
        result=schedule(self,*a,**kw)
        if role=='prefill':
            for rid,start,actual,length in prompt_rows(result,self.requests,before):
                if cfg['atomic_prompt']:assert start==0 and actual==length,('ATOMIC_AR_PROMPT_SPLIT',rid,start,actual,length)
                spans.append(dict(time=time.monotonic(),request_id=rid,start=start,count=actual,prompt_tokens=length,
                    batch_tokens=sum(result.num_scheduled_tokens.values()),scheduled_requests=len(result.num_scheduled_tokens)))
        now=time.monotonic()
        if now-clock[0]>=.25:
            clock[0]=now;samples.append(dict(time=now,waiting=len(self.waiting),running=len(self.running),KV_usage=self.kv_cache_manager.usage,
                KV_free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks()))
        return result
    Scheduler.schedule=observed_schedule
    get_stats=NixlBaseConnectorWorker.get_kv_connector_stats
    def collect(self):
        value=get_stats(self)
        if value is not None:native_stats.append(dict(time=time.monotonic(),data={k:list(v) for k,v in value.data.items()}))
        return value
    NixlBaseConnectorWorker.get_kv_connector_stats=collect
    pop_done=NixlBaseConnectorWorker._pop_done_transfers
    def observe_done(self,transfers):
        result=pop_done(self,transfers)
        if result:done_events.append(dict(time=time.monotonic(),request_ids=sorted(result)))
        return result
    NixlBaseConnectorWorker._pop_done_transfers=observe_done
    llm=LLM(**llm_options(cfg,role));engine=llm.llm_engine
    runner=engine.model_executor.driver_worker.worker.model_runner
    channel.send(dict(kind='ready',role=role,pid=os.getpid(),uuid=str(torch.cuda.get_device_properties(0).uuid),
        unique_model_parameters=sum(p.numel() for p in runner.model.parameters()),weight_bytes=sum(p.numel()*p.element_size() for p in runner.model.parameters()),
        KV_blocks=runner.kv_cache_config.num_blocks,KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),
        KV_dtype=str(runner.kv_cache_dtype),engine='Native vLLM 0.26/NIXL',model_revision=MODEL_PINS['qwen8'][1],options=llm_options(cfg,role),
        VLLM_BATCH_INVARIANT=cfg.get('ar_batch_invariant',False)))
    running=True
    try:
        while running:
            while channel.poll():
                command=channel.recv()
                if command['kind']=='stop':running=False;break
                extra={}
                if role in ('prefill','decode'):
                    extra['extra_args']=dict(kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False) if role=='prefill' else command['kv_params'])
                params=SamplingParams(max_tokens=1 if role=='prefill' else command['cap'],temperature=0,ignore_eos=True,seed=command['seed'],**extra)
                engine.add_request(command['request_id'],{'prompt_token_ids':command['input_ids']},params)
                channel.send(dict(kind='accepted',request_id=command['request_id'],time=time.monotonic()))
            if not running:break
            start=time.monotonic();outputs=engine.step();end=time.monotonic()
            if outputs:step_spans.append(dict(start=start,end=end,outputs=len(outputs)))
            for result in outputs:
                value=result.outputs[0]
                channel.send(dict(kind='output',role=role,request_id=result.request_id,time=time.monotonic(),
                    count=len(value.token_ids),finished=result.finished,output_ids=list(value.token_ids) if result.finished else None,
                    finish_reason=value.finish_reason,kv_params=result.kv_transfer_params if result.finished else None,
                    cached_prompt_tokens=getattr(result,'num_cached_tokens',None) if result.finished else None))
            if not engine.has_unfinished_requests():time.sleep(.0005)
    finally:
        from vllm.distributed.kv_transfer import get_kv_transfer_group,has_kv_transfer_group
        if has_kv_transfer_group():get_kv_transfer_group().get_kv_connector_stats()
        write(ROOT/'raw'/cfg['name']/f'native_{role}_{os.getpid()}.json',dict(role=role,prompt_spans=spans,
            queue_samples=samples,host_step_spans=step_spans,NIXL_stats_intervals=native_stats,NIXL_done_events=done_events,
            KV_allocation_refusals=allocation_refusals,
            scope='Process lifetime includes warm-up. Engine host steps are not GPU kernel durations.'))
        shutdown=getattr(engine.engine_core,'shutdown',None)
        if shutdown:shutdown()
        channel.close()


def coordinator(cfg):
    if cfg.get('nsys'):
        import torch
    from ..looped_pd_abc.nvml_watch import GPUWatch
    folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start()
    roles=('replica0','replica1') if cfg['arm']=='DP2' else ('prefill','decode')
    channels={};children=[];handles=[];directory=Path(tempfile.mkdtemp(prefix='atomic_pd_native_'))
    try:
        for role,gpu in zip(roles,cfg['gpus']):
            address=str(directory/role);listener=Listener(address,family='AF_UNIX')
            env=environment([gpu])
            with socket.socket() as probe:probe.bind(('127.0.0.1',0));env['APD_SIDE_PORT']=str(probe.getsockname()[1])
            handle=(folder/f'{role}.log').open('w');handles.append(handle)
            child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.ar_worker',os.environ['ARC_CONFIG'],
                '--role',role,'--address',address],cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
            children.append(child);listener._listener._socket.settimeout(300)
            channel=listener.accept();listener.close();channels[role]=channel
            assert channel.poll(600),'Native Qwen engine startup timeout'
            ready=channel.recv();assert ready['kind']=='ready'
            assert ready['uuid'].removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
            write(folder/f'{role}_ready.json',ready)
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
        assert len(data)==cfg['count'] and all(len(r['input_ids']['qwen8'])+cap<=cfg['context_limit'] for r,cap in zip(data,cfg['request_caps']))
        def add(role,rid,row,cap,seed,kv=None):
            channels[role].send(dict(kind='add',request_id=rid,input_ids=row['input_ids']['qwen8'],cap=cap,seed=seed,kv_params=kv))
        for i,row in enumerate(data[:cfg.get('warmup_count',2)]):
            rid=f'warm_{i}';seed=cfg['seed']-10000+i;done=False;deadline=time.monotonic()+300
            add(roles[i%2] if cfg['arm']=='DP2' else 'prefill',rid,row,2,seed)
            while not done:
                for role,channel in channels.items():
                    while channel.poll():
                        message=channel.recv()
                        if message['kind']=='accepted' or not message['finished']:continue
                        if role=='prefill':
                            assert message['kv_params'] and message['kv_params'].get('remote_block_ids')
                            add('decode',rid,row,2,seed,message['kv_params'])
                        else:assert message['count']==2;done=True
                assert time.monotonic()<deadline,'Native Qwen warm-up watchdog'
                time.sleep(.0002)
        first={};events={};completed={};arrivals={};states={};admissions={};i=0
        serial=cfg.get('calibration_serial',False);start=time.monotonic();observation=start
        if cfg.get('nsys'):
            import torch
            torch.cuda.nvtx.range_push('APD_MEASURED_WINDOW')
        try:
            while i<len(data) or len(completed)<len(data):
                now=time.monotonic()
                while i<len(data) and now>=start+cfg['arrival_offsets'][i]:
                    if serial and i>len(completed):break
                    rid=f'q{i}';arrival=now if serial else start+cfg['arrival_offsets'][i]
                    arrivals[rid]=arrival;states[rid]='prefill' if cfg['arm']!='DP2' else roles[i%2]
                    send_start=time.monotonic();add(states[rid],rid,data[i],cfg['request_caps'][i],cfg['seed']+i)
                    append(folder/'offered_requests.jsonl',dict(sample_id=i,prompt_id=data[i]['prompt_id'],arrival=arrival,
                        actual_send_start=send_start,actual_send_end=time.monotonic(),cap=cfg['request_caps'][i],seed=cfg['seed']+i));i+=1
                for role,channel in channels.items():
                    while channel.poll():
                        message=channel.recv();rid=message['request_id'];index=int(rid[1:])
                        if message['kind']=='accepted':admissions.setdefault(rid,message['time']);continue
                        if role=='prefill':
                            if not message['finished']:continue
                            kv=message['kv_params'];assert kv and kv.get('remote_block_ids'),'No real remote NIXL descriptors'
                            append(folder/'native_handoffs.jsonl',dict(request_id=rid,time=message['time'],params=kv))
                            states[rid]='decode';add('decode',rid,data[index],cfg['request_caps'][index],cfg['seed']+index,kv)
                        else:
                            delivered=time.monotonic();first.setdefault(rid,delivered)
                            events.setdefault(rid,[]).append(dict(time=delivered,count=message['count'],worker_time=message['time']))
                            if not message['finished']:continue
                            count=message['count'];assert count==cfg['request_caps'][index]
                            record=dict(model='qwen8',arm=cfg['arm'],sample_id=index,prompt_id=data[index]['prompt_id'],seed=cfg['seed']+index,
                                input_ids=data[index]['input_ids']['qwen8'],output_ids=message['output_ids'],input_tokens=len(data[index]['input_ids']['qwen8']),output_tokens=count,
                                arrival=arrivals[rid],admission=admissions[rid],first_token=first[rid],finish=delivered,
                                ttft_ms=1000*(first[rid]-arrivals[rid]),tpot_ms=1000*(delivered-first[rid])/max(1,count-1),e2e_ms=1000*(delivered-arrivals[rid]),
                                token_events=events[rid],finish_reason=message['finish_reason'],warmup=False,
                                D_cached_prompt_tokens=message.get('cached_prompt_tokens'),evidence='MEASURED_NATIVE_VLLM_NIXL_PD' if cfg['arm']!='DP2' else 'MEASURED_NATIVE_VLLM_COLOCATED_DP2',
                                model_revision=MODEL_PINS['qwen8'][1],generation='CONTROLLED_IGNORE_EOS',
                                timing_scope='Scheduled open-loop arrival to coordinator delivery. No HTTP.',
                                first_sample_scope='The native P sample is discarded. D resamples from the original prompt with imported KV.')
                            append(folder/'requests.jsonl',record);completed[rid]=record;states[rid]='finished'
                            print('DONE',cfg['name'],index,count,round(record['e2e_ms'],2),flush=True)
                now=time.monotonic()
                if now-observation>=.5:
                    observation=now;append(folder/'queue_observations.jsonl',dict(time=now,delivered=i,completed=len(completed),
                        outstanding=i-len(completed),pending_phases=dict(collections.Counter(states.values()))))
                assert not any(c.poll() is not None for c in children),'Native child exited during Serving'
                assert now-start<cfg['timeout'],'Native Qwen Serving watchdog'
                time.sleep(.0002)
        finally:
            if cfg.get('nsys'):
                import torch
                torch.cuda.nvtx.range_pop()
        write(folder/'window.json',dict(start=start,end=time.monotonic(),last_arrival=max(arrivals.values()),
            arrival_pattern='SERIAL_CALIBRATION' if serial else 'OPEN_LOOP_POISSON'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        for channel in channels.values():
            try:channel.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError):pass
        for child in children:
            try:child.wait(45)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for handle in handles:handle.close()
        monitor.stop()


def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--role');p.add_argument('--address');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    os.environ['VLLM_BATCH_INVARIANT']='1' if cfg.get('ar_batch_invariant') else '0'
    for source,expected in json.loads(os.environ.get('ATOMIC_SOURCE_HASHES','{}')).items():
        assert digest(source)==expected,'Runtime source changed before worker startup: '+source
    if a.role:stage(cfg,a.role,a.address)
    else:coordinator(cfg)


if __name__=='__main__':main()
