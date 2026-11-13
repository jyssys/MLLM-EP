"""Group exact prefixes before submission and negotiate destination credits."""
import argparse,collections,json,os,socket,subprocess,tempfile,time
from multiprocessing.connection import Listener
from dataclasses import asdict
from .common import *
from .native import stage,environment,cpu_preflight
from .policy import LayerPolicy,Observation,Choice

def coordinator(cfg):
    from .watch import GPUWatch
    folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    assert cfg.get('dynamic_layer_extension') and cfg['arm']=='PD_G1'
    assert (ROOT/'configs/frozen_SLO_regimes.json').exists() or cfg.get('correctness_only')
    costs=json.loads((ROOT/'configs'/cfg['cost_model']).read_text()) if cfg.get('cost_model') else {}
    policy=LayerPolicy(cfg,costs)
    cuts={};generations={};P_pending=collections.deque();P_active=set();P_since=None
    pending_negotiation=None;group_number=0;feedback={};delivered_counts={};last_delivered={}
    decode_step_ms=float(costs.get('decode_step_p95_ms',35.))
    def control(role,value):
        if sender is not None:sender.submit(role,value)
        else:channels[role].send(value)
    def submit_prefix(indices,choice,generation=None):
        nonlocal P_since
        values=[]
        for index in indices:
            rid=f'q{index}';cuts[rid]=36-choice.layers
            if generation is not None:generations[rid]=generation
            states[rid]='prefill';P_active.add(rid)
            values.append((rid,data[index],cfg['request_caps'][index],cfg['seed']+index,None))
        append(folder/'actual_cut_commits.jsonl',dict(time=time.monotonic(),requests=[f'q{x}' for x in indices],
            cut=36-choice.layers,offloaded_layers=choice.layers,generation=generation,
            decision=asdict(choice),stage='BEFORE_P_PREFIX_SUBMISSION',
            policy=cfg['policy'],forced_correctness_only=cfg['policy']=='FORCED_QUALIFICATION'))
        add_group('prefill',values)
        P_since=time.monotonic() if P_pending else None
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start()
    roles=('prefill','decode')
    channels={};children=[];handles=[];sender=None;deferred_records=[];directory=Path(tempfile.mkdtemp(prefix='dense_layer_pd_'))
    try:
        for role,gpu in zip(roles,cfg['gpus']):
            address=str(directory/role);listener=Listener(address,family='AF_UNIX')
            env=environment([gpu])
            with socket.socket() as probe:probe.bind(('127.0.0.1',0));env['APD_SIDE_PORT']=str(probe.getsockname()[1])
            handle=(folder/f'{role}.log').open('w');handles.append(handle)
            command=[str(PYTHON),'-m','poc_flashvep.dense_ar_elastic_layer_pd.native',os.environ['DENSE_CONFIG'],
                '--role',role,'--address',address]
            if cfg.get('nsys_bounded_capture'):
                from .nsys_window import command as profile_command
                command=profile_command(cfg,role,command)
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
            value=dict(kind='add',request_id=rid,input_ids=row['input_ids']['qwen8'],cap=cap,seed=seed,kv_params=kv,layer_cut=cuts.get(rid,36))
            if sender is not None:sender.submit(role,value)
            else:channels[role].send(value)
        def add_group(role,values):
            if cfg.get('batch_commands'):
                value=dict(kind='add_batch',commands=[dict(kind='add',request_id=rid,
                    input_ids=row['input_ids']['qwen8'],cap=cap,seed=seed,kv_params=kv,layer_cut=cuts.get(rid,36))
                    for rid,row,cap,seed,kv in values])
                if sender is not None:sender.submit(role,value)
                else:channels[role].send(value)
            else:
                for rid,row,cap,seed,kv in values:add(role,rid,row,cap,seed,kv)
        for i,row in enumerate(data[:cfg.get('warmup_count',2)]):
            rid=f'warm_{i}';seed=cfg['seed']-10000+i;done=False;deadline=time.monotonic()+180
            warm_layers=cfg.get('warmup_layer_counts',[0,2,4,8,12])
            cuts[rid]=36-warm_layers[i%len(warm_layers)]
            add('prefill',rid,row,2,seed)
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
        assert cfg.get('reuse_prefill_token') and cfg['cap']>=2
        def destination_value(rid,index,kv):
            row=data[index];cap=cfg['request_caps'][index]
            if cuts[rid]==36:
                token=source_tokens[rid]
                from .token_protocol import continuation_input
                row,cap,kv=continuation_input(row,token,cap,cfg['length'],kv)
                append(folder/'native_token_reuse.jsonl',dict(request_id=rid,P_first_token=token,
                    original_input_tokens=cfg['length'],D_input_tokens=cfg['length']+1,D_output_cap=cap,
                    scope='The source token is an actual greedy model result. The original prompt and cap in client metrics remain unchanged.'))
            kv=dict(kv,dense_layer_cut=cuts[rid])
            if rid in generations:kv['dynamic_reservation_id']=generations[rid]
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
                if not P_pending:P_since=time.monotonic()
                P_pending.append(i);states[rid]='P_HOST_COMPUTE_PENDING';send_start=None
                append(folder/'offered_requests.jsonl',dict(sample_id=i,prompt_id=data[i]['prompt_id'],arrival=arrival,
                    coordinator_observed_arrival_time=time.monotonic(),
                    actual_send_start=send_start if sender is None else None,
                    actual_send_end=time.monotonic() if send_start is not None and sender is None else None,
                    ingress_enqueue_start=send_start if sender is not None else None,
                    send_timing_scope='SOCKET_WRITE_IN_ACTUAL_SUBMISSIONS' if sender is not None else 'SYNCHRONOUS_SOCKET_WRITE',
                    cap=cfg['request_caps'][i],seed=cfg['seed']+i));i+=1
            for role,channel in channels.items():
                while channel.poll():
                    receive_start=time.monotonic() if cfg.get('completion_handler_timing') else None
                    message=channel.recv()
                    receive_end=time.monotonic() if receive_start is not None else None
                    if message['kind']=='runtime_state':
                        feedback[role]=message
                        append(folder/'runtime_states.jsonl',message)
                        continue
                    if message['kind']=='reservation_result':
                        assert pending_negotiation and message['generation']==pending_negotiation['generation']
                        job=pending_negotiation
                        append(folder/'reservation_observations.jsonl',dict(message,source_receive=time.monotonic(),
                            negotiation_ms=1000*(time.monotonic()-job['submitted'])))
                        if job.get('cancel_sent'):continue
                        pending_negotiation=None
                        if message['accepted']:
                            control('decode',dict(kind='commit',request_id=job['generation'],generation=job['generation']))
                            submit_prefix(job['indices'],job['choice'],job['generation'])
                        else:
                            job['choice'].layers=0;job['choice'].reason='DESTINATION_CREDIT_REJECTED'
                            submit_prefix(job['indices'],job['choice'])
                        continue
                    if message['kind']=='reservation_cancelled':
                        assert pending_negotiation and pending_negotiation.get('cancel_sent')
                        assert message['generation']==pending_negotiation['generation']
                        job=pending_negotiation;pending_negotiation=None
                        job['choice'].layers=0;job['choice'].reason='NEGOTIATION_TIMEOUT_NATIVE_BEFORE_PREFIX'
                        submit_prefix(job['indices'],job['choice'])
                        continue
                    if message['kind']=='commit_result':
                        append(folder/'reservation_commit_ACK.jsonl',dict(message,source_receive=time.monotonic()))
                        continue
                    rid=message['request_id'];index=int(rid[1:])
                    if message['kind']=='prefill_token_ready':
                        assert cuts[rid]==36 and rid in source_tokens
                        if cfg.get('source_token_release_policy')=='SOURCE_READY':
                            assert rid in first
                            continue
                        assert rid not in first
                        delivered=time.monotonic();first[rid]=delivered
                        events[rid]=[dict(time=delivered,count=1,worker_time=message['time'],source='P_TOKEN_RELEASED_AFTER_NATIVE_D_IMPORT')]
                        delivered_counts[rid]=1;last_delivered[rid]=delivered
                        continue
                    if message['kind']=='accepted':admissions.setdefault(rid,message['time']);continue
                    if role=='prefill':
                        if not message['finished']:continue
                        P_active.discard(rid)
                        if cuts[rid]==36:
                            assert len(message['output_ids'])==1;source_tokens[rid]=message['output_ids'][0]
                            if cfg.get('source_token_release_policy')=='SOURCE_READY':
                                delivered=time.monotonic();assert rid not in first
                                first[rid]=delivered;events[rid]=[dict(time=delivered,count=1,
                                    worker_time=message['time'],source='ACTUAL_P_TOKEN_RELEASED_AT_SOURCE_READY')]
                        kv=message['kv_params'];assert kv and kv.get('remote_block_ids')
                        assert int(kv['dense_layer_cut'])==cuts[rid]
                        append(folder/'native_handoffs.jsonl',dict(request_id=rid,time=message['time'],params=kv))
                        if cfg.get('serialize_D_imports'):
                            if not pending_D:pending_since=time.monotonic()
                            states[rid]='D_HOST_PENDING';pending_D.append((rid,index,kv))
                        else:
                            states[rid]='decode';value=destination_value(rid,index,kv);add('decode',*value)
                    else:
                        delivered=time.monotonic();first.setdefault(rid,delivered)
                        offset=1 if cuts[rid]==36 else 0
                        if offset:assert events[rid][0]['count']==1 and rid in source_tokens
                        events.setdefault(rid,[]).append(dict(time=delivered,count=message['count']+offset,worker_time=message['time']))
                        delivered_counts[rid]=message['count']+offset;last_delivered[rid]=delivered
                        if not message['finished']:continue
                        count=message['count']+offset;assert count==cfg['request_caps'][index]
                        record=dict(model='qwen8',arm=cfg['arm'],sample_id=index,prompt_id=data[index]['prompt_id'],seed=cfg['seed']+index,
                            input_ids=data[index]['input_ids']['qwen8'],output_ids=([source_tokens[rid]] if offset else [])+message['output_ids'],
                            input_tokens=len(data[index]['input_ids']['qwen8']),output_tokens=count,
                            arrival=arrivals[rid],admission=admissions[rid],first_token=first[rid],finish=delivered,
                            ttft_ms=1000*(first[rid]-arrivals[rid]),tpot_ms=1000*(delivered-first[rid])/max(1,count-1),
                            e2e_ms=1000*(delivered-arrivals[rid]),token_events=events[rid],finish_reason=message['finish_reason'],
                            warmup=False,actual_layer_cut=cuts[rid],completed_D_prefill_layers=36-cuts[rid],dynamic_reservation=generations.get(rid),D_cached_prompt_tokens=message.get('cached_prompt_tokens'),model_revision=MODEL_REVISION,
                            evidence='MEASURED_NATIVE_VLLM_NIXL_PD' if cfg['arm']!='DP2' else 'MEASURED_NATIVE_VLLM_COLOCATED_DP2',
                            generation='CONTROLLED_IGNORE_EOS',timing_scope='Scheduled open-loop arrival to coordinator delivery. No HTTP.',
                            first_sample_scope=('The P token is released at Source readiness. Import and next Decode execution remain in the first ITL.' if offset and cfg.get('source_token_release_policy')=='SOURCE_READY' else
                                'The P token is held until Native import and next Decode submission. D consumes it and generates the remaining tokens.' if offset else
                                'P placeholder sample is discarded. D completes the suffix and generates the first user token.' if cfg.get('layer_mode')=='MIGRATE' else
                                'Native P sample is discarded. D resamples the original prompt with imported KV.' if cfg['arm']!='DP2' else
                                'The local full-model Prefill sample is the first user token. No P-to-D handoff.'))
                        record_start=time.monotonic() if receive_start is not None else None
                        if cfg.get('defer_completion_log'):deferred_records.append(record)
                        else:append(folder/'requests.jsonl',record)
                        completed[rid]=record;states[rid]='finished';active_D.discard(rid)
                        local_active[role].discard(rid)
                        print('DONE',cfg['name'],index,count,round(record['e2e_ms'],2),flush=True)
                        if receive_start is not None:
                            append(folder/'completion_handler_timings.jsonl',dict(request_id=rid,role=role,
                                receive_start=receive_start,receive_end=receive_end,record_start=record_start,
                                record_end=time.monotonic(),worker_time=message['time'],delivered=delivered,
                                scope='Host observation only. Request log serialization occurs after the recorded delivery time.'))
            ready_suffix=any(cuts[rid]<36 for rid,_,_ in pending_D)
            active_suffix=any(cuts[rid]<36 and delivered_counts.get(rid,0)==0 for rid in active_D)
            native_pass=cfg.get('native_continuous_D_ingress') and not ready_suffix and not active_suffix and len(active_D)<cfg['D_slots']
            if pending_D and (not active_D or native_pass):
                collect_ms=cfg.get('D_cohort_collect_ms',0)
                if time.monotonic()-pending_since>=collect_ms/1000:
                    if cfg.get('native_continuous_D_ingress') and not active_D and ready_suffix:
                        # Preserve progress for committed, ready suffix work.
                        # Never interrupt an existing Decode cohort.
                        suffix_cut=next(cuts[rid] for rid,_,_ in pending_D if cuts[rid]<36)
                        waiting=list(pending_D)
                        pending_D=collections.deque([r for r in waiting if cuts[r[0]]==suffix_cut]+
                            [r for r in waiting if cuts[r[0]]!=suffix_cut])
                    first_cut=cuts[pending_D[0][0]]
                    compatible=0
                    for waiting_rid,_,_ in pending_D:
                        if cuts[waiting_rid]!=first_cut:break
                        compatible+=1
                    maximum=cfg.get('D_admission_batch',1)
                    maximum=min(maximum,cfg['D_slots']-len(active_D))
                    if first_cut<36:maximum=min(maximum,cfg.get('D_token_budget',cfg['token_budget'])//cfg['length'])
                    count=min(compatible,maximum);cohort_number+=1
                    destination=[];joining=bool(active_D)
                    for _ in range(count):
                        rid,index,kv=pending_D.popleft();states[rid]='decode';active_D.add(rid)
                        append(folder/'D_host_admission.jsonl',dict(time=time.monotonic(),request_id=rid,
                            reason='NATIVE_CACHED_QUERY_JOIN_ACTIVE_DECODE' if joining else 'NO_EXISTING_DECODE_COHORT_ADMISSION',pending=len(pending_D),
                            cohort=cohort_number,cohort_size=count,collection_ms=collect_ms))
                        if cfg.get('cohort_barrier'):
                            kv=dict(kv,dense_cohort_id=f'measured_{cohort_number}',dense_cohort_size=count)
                        destination.append(destination_value(rid,index,kv))
                    add_group('decode',destination)
                    pending_since=time.monotonic() if pending_D else None
            if P_pending and not P_active and pending_negotiation is None:
                if time.monotonic()-P_since>=cfg.get('P_collect_ms',2)/1000 or len(P_pending)>=cfg['P_slots']:
                    count=min(len(P_pending),cfg.get('P_group_size',cfg['P_slots']))
                    fixed_layers=None
                    if cfg['policy']=='STATIC' and (cfg.get('static_fraction_axis')=='REQUEST_ID_BLOCKS' or 'static_layers_by_request' in cfg):
                        fixed_layers=policy.static_request_layers(P_pending[0])
                        count=next((j for j in range(1,count) if policy.static_request_layers(P_pending[j])!=fixed_layers),count)
                    indices=[P_pending.popleft() for _ in range(count)]
                    current=time.monotonic();state=feedback.get('decode')
                    if state is None:
                        age=1000000.;KV_free=0
                    else:
                        age=1000*(current-state['time']);KV_free=state['KV_free_blocks']-state.get('reserved_future_blocks',0)
                    remaining=max((cfg['request_caps'][int(rid[1:])]-delivered_counts.get(rid,0) for rid in active_D),default=0)
                    active_step_ms=policy.decode_bound(len(active_D))
                    available=current+remaining*active_step_ms/1000
                    if costs and any(delivered_counts.get(rid,0)==0 and cuts[rid]<36 for rid in active_D):
                        active_cut=next(cuts[rid] for rid in active_D)
                        available+=policy._cost('D_suffix_ms',36-active_cut,len(active_D))/1000
                    active_available=available
                    pending_plan=[];waiting=list(pending_D);position=0
                    while position<len(waiting):
                        cut=cuts[waiting[position][0]];limit=cfg.get('D_admission_batch',4)
                        if cut<36:limit=min(limit,cfg.get('D_token_budget',cfg['token_budget'])//cfg['length'])
                        size=0
                        while position+size<len(waiting) and size<limit and cuts[waiting[position+size][0]]==cut:size+=1
                        layer_count=36-cut
                        preparation=(policy._cost('D_suffix_ms',layer_count,size)+policy._cost('handoff_ms',layer_count,size)+
                            policy._cost('D_commit_ms',layer_count,size)) if costs else 0.
                        pending_step_ms=policy.decode_bound(size)
                        drain=preparation+(cfg['cap']-1)*pending_step_ms
                        available+=drain/1000;position+=size
                        pending_plan.append(dict(cut=cut,batch=size,predicted_preparation_ms=preparation,
                            predicted_Decode_step_ms=pending_step_ms,predicted_drain_ms=drain))
                    deadlines=[last_delivered[rid]+cfg.get('decode_deadline_ms',40)/1000 for rid in active_D if rid in last_delivered]
                    decision_state=Observation(current,count,len(P_pending),available,len(pending_D),max(0,KV_free),
                        state.get('decode_ready',0) if state else 0,min(deadlines) if deadlines else None,age,
                        tuple(cuts[rid] for rid,_,_ in pending_D),len(active_D))
                    choice=policy.choose(decision_state);group_number+=1
                    if fixed_layers is not None:choice=Choice(fixed_layers,'STATIC_REQUEST_ID_BLOCK_FRACTION',[])
                    append(folder/'online_decisions.jsonl',dict(time=current,observation=asdict(decision_state),choice=asdict(choice),
                        generation=f'res_{group_number}',policy=cfg['policy'],
                        active_D_requests=sorted(active_D),pending_D_requests=[rid for rid,_,_ in pending_D],
                        active_delivered_counts={rid:delivered_counts.get(rid,0) for rid in active_D},
                        predicted_active_D_finish=active_available,observed_decode_step_bound_ms=active_step_ms,
                        pending_cohort_prediction=pending_plan,
                        evidence='CORRECTNESS_ONLY' if cfg['policy']=='FORCED_QUALIFICATION' else 'MEASURED_ONLINE_DECISION',
                        scope='Observed Host state and calibrated predictions. Predicted availability is not physical idle.'))
                    if choice.layers:
                        generation=f'res_{group_number}'
                        pending_negotiation=dict(generation=generation,indices=indices,choice=choice,submitted=time.monotonic())
                        control('decode',dict(kind='reserve',request_id=generation,generation=generation,
                            requests=[f'q{x}' for x in indices],cut=36-choice.layers,
                            blocks_each=(cfg['length']+cfg['cap']+15)//16))
                    else:submit_prefix(indices,choice)
            if pending_negotiation is not None:
                if not pending_negotiation.get('cancel_sent') and time.monotonic()-pending_negotiation['submitted']>=cfg.get('negotiation_timeout_s',2):
                    pending_negotiation['cancel_sent']=True
                    generation=pending_negotiation['generation']
                    control('decode',dict(kind='cancel_reservation',request_id=generation,generation=generation))
                    append(folder/'negotiation_timeouts.jsonl',dict(time=time.monotonic(),generation=generation,
                        scope='No Prefix was submitted. Wait for cancellation ACK before Native fallback.'))
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
        for record in deferred_records:append(folder/'requests.jsonl',record)


def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--role');p.add_argument('--address');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ['VLLM_BATCH_INVARIANT']='1'
    if cfg.get('kv_layout'):os.environ['VLLM_KV_CACHE_LAYOUT']=cfg['kv_layout']
    for filename,expected in json.loads(os.environ.get('DENSE_SOURCE_HASHES','{}')).items():
        assert digest(filename)==expected,'Source changed before Native startup: '+filename
    if a.role:stage(cfg,a.role,a.address)
    else:cpu_preflight(cfg);coordinator(cfg)


if __name__=='__main__':main()


