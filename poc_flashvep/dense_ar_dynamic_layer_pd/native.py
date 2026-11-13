"""Reuse the qualified vLLM/NIXL stage with an isolated Dense AR coordinator."""
import argparse,collections,json,os,socket,subprocess,tempfile,time
from multiprocessing.connection import Client,Listener
from .common import *


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')


def environment(gpus):
    root=os.environ.get('DENSE_FROZEN_ROOT',str(REPO))
    return dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PYTHONPATH=root,
        PATH=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH',''),
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1')


class BatchChannel:
    """Drain a complete control batch before the existing Native engine step."""
    def __init__(self,channel):self.channel=channel;self.pending=collections.deque()
    def poll(self,*args):return bool(self.pending) or self.channel.poll(*args)
    def recv(self):
        if self.pending:return self.pending.popleft()
        value=self.channel.recv()
        if value['kind']!='add_batch':return value
        assert value['commands'] and all(c['kind']=='add' for c in value['commands'])
        self.pending.extend(value['commands']);return self.pending.popleft()
    def send(self,*args):return self.channel.send(*args)
    def close(self):return self.channel.close()


def stage(cfg,role,address):
    from .ack_retention import install as retain_ACK
    retain_ACK(cfg,role)
    if cfg.get('dynamic_layer_extension') or cfg.get('worker_state_feedback'):
        from . import worker as qualified
    else:
        from ..looped_pd_atomic_prefill import ar_worker as qualified
    qualified.ROOT=ROOT;qualified.MODEL_PATHS=dict(qwen8=MODEL)
    if cfg.get('batch_commands'):
        native_client=qualified.Client
        qualified.Client=lambda *a,**kw:BatchChannel(native_client(*a,**kw))
    notice_channel={}
    if cfg.get('reuse_prefill_token'):
        original_client=qualified.Client
        def connected(*a,**kw):
            channel=original_client(*a,**kw);notice_channel['value']=channel;return channel
        qualified.Client=connected
    if cfg.get('fixed_KV_bytes') or cfg.get('isolate_layer_compile_cache') or cfg.get('ar_custom_ops') or cfg.get('ar_compile_mode') is not None:
        original_options=qualified.llm_options
        def matched_options(config,stage_role):
            value=original_options(config,stage_role)
            if config.get('ar_custom_ops'):
                value['compilation_config']['custom_ops']=config['ar_custom_ops']
            if config.get('ar_compile_mode') is not None:
                value['compilation_config']['mode']=config['ar_compile_mode']
            if config.get('decode_only_graph_roles') and stage_role not in config['decode_only_graph_roles']:
                value['compilation_config']['cudagraph_mode']='NONE'
            if config.get('fixed_KV_bytes'):value['kv_cache_memory_bytes']=config['fixed_KV_bytes']
            if stage_role=='decode' and config.get('D_token_budget'):
                value['max_num_batched_tokens']=config['D_token_budget']
            if config.get('isolate_layer_compile_cache'):
                # Mode and Cut are closure constants, not native VllmConfig
                # fields. Give each static function variant an explicit cache.
                tag=f"{config.get('layer_mode','NATIVE')}_{config.get('layer_cut',36)}_{stage_role}"
                tag+='_r32' if config.get('retain_fp32_boundary_residual') else '_r16'
                tag+='_'+config.get('ar_execution_family','DEFAULT')
                value['compilation_config']['cache_dir']=str(ROOT/'tools/layer_compile_cache'/tag)
            return value
        qualified.llm_options=matched_options
    if cfg.get('physical_activity'):
        from .activity import install as install_activity
        install_activity(cfg,role)
    observe=cfg.get('runtime_observation') and role in cfg.get('runtime_observation_roles',('prefill','decode'))
    if observe or cfg.get('layer_mode') or cfg.get('hidden_transport')=='NIXL_GPU' or cfg.get('cohort_barrier') or cfg.get('execution_events') or cfg.get('normal_lifetime_audit') or cfg.get('reuse_prefill_token'):
        from .extension import install
        install(dict(cfg,runtime_observation=bool(observe)),role,
            notify=(lambda message:notice_channel['value'].send(message)) if cfg.get('reuse_prefill_token') else None)
    qualified.stage(cfg,role,address)


def coordinator(cfg):
    from .watch import GPUWatch
    folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start()
    roles=('replica0','replica1') if cfg['arm']=='DP2' else ('prefill','decode')
    channels={};children=[];handles=[];sender=None;directory=Path(tempfile.mkdtemp(prefix='dense_layer_pd_'))
    try:
        for role,gpu in zip(roles,cfg['gpus']):
            address=str(directory/role);listener=Listener(address,family='AF_UNIX')
            env=environment([gpu])
            with socket.socket() as probe:probe.bind(('127.0.0.1',0));env['APD_SIDE_PORT']=str(probe.getsockname()[1])
            handle=(folder/f'{role}.log').open('w');handles.append(handle)
            command=[str(PYTHON),'-m','poc_flashvep.dense_ar_dynamic_layer_pd.native',os.environ['DENSE_CONFIG'],
                '--role',role,'--address',address]
            child=subprocess.Popen(command,cwd=os.environ.get('DENSE_FROZEN_ROOT',str(REPO)),env=env,
                stdout=handle,stderr=subprocess.STDOUT);children.append(child)
            listener._listener._socket.settimeout(cfg.get('startup_timeout',360))
            channel=listener.accept();listener.close();channels[role]=channel
            assert channel.poll(cfg.get('startup_timeout',360)),'Native startup timeout'
            ready=channel.recv();assert ready['kind']=='ready'
            from ..looped_pd_abc.common import UUIDS
            assert ready['uuid'].removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
            assert ready['effective_VLLM_BATCH_INVARIANT'] is True
            write(folder/f'{role}_ready.json',ready)
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
        assert len(data)==cfg['count'] and all(len(r['input_ids']['qwen8'])+cap<=40960 for r,cap in zip(data,cfg['request_caps']))
        def add(role,rid,row,cap,seed,kv=None):
            value=dict(kind='add',request_id=rid,input_ids=row['input_ids']['qwen8'],cap=cap,seed=seed,kv_params=kv)
            if sender is not None:sender.submit(role,value)
            else:channels[role].send(value)
        def add_group(role,values):
            if cfg.get('batch_commands'):
                value=dict(kind='add_batch',commands=[dict(kind='add',request_id=rid,
                    input_ids=row['input_ids']['qwen8'],cap=cap,seed=seed,kv_params=kv)
                    for rid,row,cap,seed,kv in values])
                if sender is not None:sender.submit(role,value)
                else:channels[role].send(value)
            else:
                for rid,row,cap,seed,kv in values:add(role,rid,row,cap,seed,kv)
        for i,row in enumerate(data[:cfg.get('warmup_count',2)]):
            rid=f'warm_{i}';seed=cfg['seed']-10000+i;done=False;deadline=time.monotonic()+180
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
                assert time.monotonic()<deadline,'Native warm-up timeout'
                assert all(c.poll() is None for c in children),'Native warm-up child failed'
                time.sleep(.0002)
        warm_start=time.monotonic()
        for batch_size in cfg.get('warmup_batch_sizes',[]):
            if cfg['arm']=='DP2':
                for lane,role in enumerate(roles):
                    values=[(f'geometry_{lane}_{batch_size}_{j}',data[j],cfg.get('warmup_output_cap',16),cfg['seed']-20000+j,None)
                        for j in range(batch_size)]
                    add_group(role,values);done=set();deadline=time.monotonic()+180
                    while len(done)<batch_size:
                        for source,channel in channels.items():
                            while channel.poll():
                                message=channel.recv()
                                if message['kind']=='output' and message['finished']:done.add(message['request_id'])
                        assert all(c.poll() is None for c in children),'Colocated warm-up child failed'
                        assert time.monotonic()<deadline,'Colocated warm-up timeout'
                        time.sleep(.0002)
                continue
            assert batch_size<=len(data) and batch_size*cfg['length']<=cfg['token_budget']
            incoming={};done=set();deadline=time.monotonic()+180
            rows={f'geometry_{batch_size}_{j}':data[j] for j in range(batch_size)}
            add_group('prefill',[(rid,row,cfg.get('warmup_output_cap',16),cfg['seed']-20000+j,None)
                for j,(rid,row) in enumerate(rows.items())])
            submitted=False
            while len(done)<batch_size:
                for role,channel in channels.items():
                    while channel.poll():
                        message=channel.recv()
                        if message['kind']=='accepted' or not message['finished']:continue
                        if role=='prefill':incoming[message['request_id']]=message['kv_params']
                        else:done.add(message['request_id'])
                if len(incoming)==batch_size and not submitted:
                    destination=[]
                    for j,(rid,row) in enumerate(rows.items()):
                        kv=dict(incoming[rid],dense_cohort_id=f'warm_geometry_{batch_size}',dense_cohort_size=batch_size)
                        destination.append((rid,row,cfg.get('warmup_output_cap',16),cfg['seed']-20000+j,kv))
                    add_group('decode',destination)
                    submitted=True
                assert all(c.poll() is None for c in children),'Geometry warm-up child failed'
                assert time.monotonic()<deadline,'Geometry warm-up timeout'
                time.sleep(.0002)
        write(folder/'geometry_warmup.json',dict(batch_sizes=cfg.get('warmup_batch_sizes',[]),
            output_cap=cfg.get('warmup_output_cap',16),seconds=time.monotonic()-warm_start,
            scope='Actual source Prefill, Native handoff, and destination Decode geometry. Excluded from Serving statistics.'))
        for channel in channels.values():channel.send(dict(kind='measurement_begin'))
        for channel in channels.values():
            assert channel.poll(30) and channel.recv()['kind']=='measurement_ready'
        if cfg.get('async_ingress'):
            from .async_ingress import AsyncIngress
            sender=AsyncIngress(channels,lambda role,row:append(folder/(
                'D_actual_submissions.jsonl' if role=='decode' else 'actual_submissions.jsonl'),row))
        first={};events={};completed={};arrivals={};states={};admissions={};i=0;source_tokens={}
        if cfg.get('reuse_prefill_token'):assert cfg['arm']!='DP2' and cfg.get('layer_mode')!='MIGRATE' and cfg['cap']>=2
        def destination_value(rid,index,kv):
            row=data[index];cap=cfg['request_caps'][index]
            if cfg.get('reuse_prefill_token'):
                token=source_tokens[rid]
                from .token_protocol import continuation_input
                row,cap,kv=continuation_input(row,token,cap,cfg['length'],kv)
                append(folder/'native_token_reuse.jsonl',dict(request_id=rid,P_first_token=token,
                    original_input_tokens=cfg['length'],D_input_tokens=cfg['length']+1,D_output_cap=cap,
                    scope='The source token is an actual greedy model result. The original prompt and cap in client metrics remain unchanged.'))
            return (rid,row,cap,cfg['seed']+index,kv)
        pending_D=collections.deque();active_D=set();pending_since=None;cohort_number=0
        local_pending={r:collections.deque() for r in roles};local_active={r:set() for r in roles}
        local_since={r:None for r in roles}
        serial=cfg.get('calibration_serial',False);start=time.monotonic();observation=start
        write(folder/'measurement_begin.json',dict(time=start))
        while i<len(data) or len(completed)<len(data):
            now=time.monotonic()
            while i<len(data) and now>=start+cfg['arrival_offsets'][i]:
                if serial and i>len(completed):break
                rid=f'q{i}';arrival=now if serial else start+cfg['arrival_offsets'][i]
                arrivals[rid]=arrival;states[rid]='prefill' if cfg['arm']!='DP2' else roles[i%2]
                if cfg['arm']=='DP2' and cfg.get('placement')=='LEAST_OUTSTANDING':
                    if cfg.get('colocated_cohorts'):
                        states[rid]=min(roles,key=lambda r:(len(local_pending[r])+len(local_active[r]),r!=roles[i%2]))
                    else:
                        states[rid]=min(roles,key=lambda r:(sum(v==r for k,v in states.items() if k!=rid),r!=roles[i%2]))
                placement_role=states[rid]
                if cfg.get('colocated_cohorts'):
                    assert cfg['arm']=='DP2'
                    role=states[rid]
                    if not local_pending[role]:local_since[role]=time.monotonic()
                    local_pending[role].append(i);states[rid]='LOCAL_HOST_PENDING';send_start=None
                else:
                    send_start=time.monotonic();add(states[rid],rid,data[i],cfg['request_caps'][i],cfg['seed']+i)
                append(folder/'offered_requests.jsonl',dict(sample_id=i,prompt_id=data[i]['prompt_id'],arrival=arrival,
                    coordinator_observed_arrival_time=time.monotonic(),
                    placement_role=placement_role,
                    actual_send_start=send_start if sender is None else None,
                    actual_send_end=time.monotonic() if send_start is not None and sender is None else None,
                    ingress_enqueue_start=send_start if sender is not None else None,
                    send_timing_scope='SOCKET_WRITE_IN_ACTUAL_SUBMISSIONS' if sender is not None else 'SYNCHRONOUS_SOCKET_WRITE',
                    cap=cfg['request_caps'][i],seed=cfg['seed']+i));i+=1
            for role,channel in channels.items():
                while channel.poll():
                    message=channel.recv();rid=message['request_id'];index=int(rid[1:])
                    if message['kind']=='prefill_token_ready':
                        assert cfg.get('reuse_prefill_token') and rid in source_tokens
                        if cfg.get('source_token_release_policy')=='SOURCE_READY':
                            assert rid in first
                            continue
                        assert rid not in first
                        delivered=time.monotonic();first[rid]=delivered
                        events[rid]=[dict(time=delivered,count=1,worker_time=message['time'],source='P_TOKEN_RELEASED_AFTER_NATIVE_D_IMPORT')]
                        continue
                    if message['kind']=='accepted':admissions.setdefault(rid,message['time']);continue
                    if role=='prefill':
                        if not message['finished']:continue
                        if cfg.get('reuse_prefill_token'):
                            assert len(message['output_ids'])==1;source_tokens[rid]=message['output_ids'][0]
                            if cfg.get('source_token_release_policy')=='SOURCE_READY':
                                delivered=time.monotonic();assert rid not in first
                                first[rid]=delivered;events[rid]=[dict(time=delivered,count=1,
                                    worker_time=message['time'],source='ACTUAL_P_TOKEN_RELEASED_AT_SOURCE_READY')]
                        kv=message['kv_params'];assert kv and kv.get('remote_block_ids')
                        append(folder/'native_handoffs.jsonl',dict(request_id=rid,time=message['time'],params=kv))
                        if cfg.get('serialize_D_imports'):
                            if not pending_D:pending_since=time.monotonic()
                            states[rid]='D_HOST_PENDING';pending_D.append((rid,index,kv))
                        else:
                            states[rid]='decode';value=destination_value(rid,index,kv);add('decode',*value)
                    else:
                        delivered=time.monotonic();first.setdefault(rid,delivered)
                        offset=1 if cfg.get('reuse_prefill_token') else 0
                        if offset:assert events[rid][0]['count']==1 and rid in source_tokens
                        events.setdefault(rid,[]).append(dict(time=delivered,count=message['count']+offset,worker_time=message['time']))
                        if not message['finished']:continue
                        count=message['count']+offset;assert count==cfg['request_caps'][index]
                        record=dict(model='qwen8',arm=cfg['arm'],sample_id=index,prompt_id=data[index]['prompt_id'],seed=cfg['seed']+index,
                            input_ids=data[index]['input_ids']['qwen8'],output_ids=([source_tokens[rid]] if offset else [])+message['output_ids'],
                            input_tokens=len(data[index]['input_ids']['qwen8']),output_tokens=count,
                            arrival=arrivals[rid],admission=admissions[rid],first_token=first[rid],finish=delivered,
                            ttft_ms=1000*(first[rid]-arrivals[rid]),tpot_ms=1000*(delivered-first[rid])/max(1,count-1),
                            e2e_ms=1000*(delivered-arrivals[rid]),token_events=events[rid],finish_reason=message['finish_reason'],
                            warmup=False,D_cached_prompt_tokens=message.get('cached_prompt_tokens'),model_revision=MODEL_REVISION,
                            evidence='MEASURED_NATIVE_VLLM_NIXL_PD' if cfg['arm']!='DP2' else 'MEASURED_NATIVE_VLLM_COLOCATED_DP2',
                            generation='CONTROLLED_IGNORE_EOS',timing_scope='Scheduled open-loop arrival to coordinator delivery. No HTTP.',
                            first_sample_scope=('The P token is released at Source readiness. Import and next Decode execution remain in the first ITL.' if offset and cfg.get('source_token_release_policy')=='SOURCE_READY' else
                                'The P token is held until Native import and next Decode submission. D consumes it and generates the remaining tokens.' if offset else
                                'P placeholder sample is discarded. D completes the suffix and generates the first user token.' if cfg.get('layer_mode')=='MIGRATE' else
                                'Native P sample is discarded. D resamples the original prompt with imported KV.' if cfg['arm']!='DP2' else
                                'The local full-model Prefill sample is the first user token. No P-to-D handoff.'))
                        append(folder/'requests.jsonl',record);completed[rid]=record;states[rid]='finished';active_D.discard(rid)
                        local_active[role].discard(rid)
                        print('DONE',cfg['name'],index,count,round(record['e2e_ms'],2),flush=True)
            if pending_D and not active_D:
                collect_ms=cfg.get('D_cohort_collect_ms',0)
                if time.monotonic()-pending_since>=collect_ms/1000:
                    count=min(len(pending_D),cfg.get('D_admission_batch',1));cohort_number+=1
                    destination=[]
                    for _ in range(count):
                        rid,index,kv=pending_D.popleft();states[rid]='decode';active_D.add(rid)
                        append(folder/'D_host_admission.jsonl',dict(time=time.monotonic(),request_id=rid,
                            reason='NO_EXISTING_DECODE_COHORT_ADMISSION',pending=len(pending_D),
                            cohort=cohort_number,cohort_size=count,collection_ms=collect_ms))
                        if cfg.get('cohort_barrier'):
                            kv=dict(kv,dense_cohort_id=f'measured_{cohort_number}',dense_cohort_size=count)
                        destination.append(destination_value(rid,index,kv))
                    add_group('decode',destination)
                    pending_since=time.monotonic() if pending_D else None
            if cfg.get('colocated_cohorts'):
                for role in roles:
                    if not local_pending[role] or local_active[role]:continue
                    if time.monotonic()-local_since[role]<cfg.get('D_cohort_collect_ms',2)/1000:continue
                    count=min(len(local_pending[role]),cfg.get('D_admission_batch',4));values=[]
                    for _ in range(count):
                        index=local_pending[role].popleft();rid=f'q{index}';states[rid]=role;local_active[role].add(rid)
                        if sender is None:append(folder/'actual_submissions.jsonl',dict(sample_id=index,actual_send_start=time.monotonic(),role=role))
                        append(folder/'colocated_host_admission.jsonl',dict(request_id=rid,time=time.monotonic(),role=role,
                            cohort_size=count,reason='NO_EXISTING_LOCAL_REQUEST',pending=len(local_pending[role])))
                        values.append((rid,data[index],cfg['request_caps'][index],cfg['seed']+index,None))
                    add_group(role,values)
                    local_since[role]=arrivals[f'q{local_pending[role][0]}'] if local_pending[role] else None
            now=time.monotonic()
            if now-observation>=.5:
                observation=now;append(folder/'queue_observations.jsonl',dict(time=now,delivered=i,completed=len(completed),
                    outstanding=i-len(completed),pending_phases=dict(collections.Counter(states.values()))))
            assert all(c.poll() is None for c in children),'Native child exited during Serving'
            if sender is not None:sender.check()
            assert now-start<cfg['timeout'],'Native Serving watchdog'
            time.sleep(.0002)
        write(folder/'window.json',dict(start=start,end=time.monotonic(),last_arrival=max(arrivals.values()),
            arrival_pattern='SERIAL_CALIBRATION' if serial else 'OPEN_LOOP_POISSON'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        drained=sender.close() if sender is not None else True
        for channel in channels.values() if drained else ():
            try:channel.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError):pass
        exit_rows=[]
        for child in children:
            forced=False
            try:child.wait(45)
            except subprocess.TimeoutExpired:
                forced=True;child.terminate()
                try:child.wait(10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            exit_rows.append(dict(pid=child.pid,returncode=child.returncode,forced_cleanup=forced))
        write(folder/'worker_exits.json',dict(records=exit_rows,passed=bool(exit_rows) and all(r['returncode']==0 and not r['forced_cleanup'] for r in exit_rows)))
        for handle in handles:handle.close()
        monitor.stop()


def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--role');p.add_argument('--address');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ['VLLM_BATCH_INVARIANT']='1'
    if cfg.get('kv_layout'):os.environ['VLLM_KV_CACHE_LAYOUT']=cfg['kv_layout']
    for filename,expected in json.loads(os.environ.get('DENSE_SOURCE_HASHES','{}')).items():
        assert digest(filename)==expected,'Source changed before Native startup: '+filename
    if a.role:stage(cfg,a.role,a.address)
    else:coordinator(cfg)


if __name__=='__main__':main()

