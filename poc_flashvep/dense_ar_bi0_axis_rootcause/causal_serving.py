"""Compare standard DP2 with Full36 Native PD, without custom cohorts."""
import collections
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener
from .common import ROOT,PYTHON,MODEL,write
from ..dense_ar_bi0_shortanswer_pd.ingress import Ingress

BUFFERED={}


def append(path,row):
    if path.name in ('credit_events.jsonl','P_wait_segments.jsonl','request_submissions.jsonl'):
        BUFFERED.setdefault(path,[]).append(row)
        return
    with path.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def output_contract(cfg,role,rid,cap,ignore):
    lengths=cfg.get('replay_output_lengths')
    if lengths is not None and rid.startswith('q'):
        assert cfg.get('diagnostic_controlled_output_replay')
        assert rid[1:].isdigit()
        if role!='prefill':cap=int(lengths[int(rid[1:])])
        ignore=True
    return cap,ignore


def main():
    cfg_path=Path(sys.argv[1]);cfg=json.loads(cfg_path.read_text());folder=ROOT/'raw'/cfg['name']
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    assert len(data)==cfg['count'];native=cfg['method'] in ('Native','Layer','Token') and not cfg.get('reference_single');reuse=cfg.get('protocol','OFFICIAL_PROXY') in ('FIRST_TOKEN_REUSE','FIRST_TOKEN_REUSE_BOUNDED')
    roles=('single',) if cfg.get('reference_single') else (('prefill','decode') if native else ('replica0','replica1'))
    channels={};children=[];handles=[];sender=None;directory=Path(tempfile.mkdtemp(prefix='native_real200_'));control_ports={}
    eos=json.loads((MODEL/'generation_config.json').read_text())['eos_token_id']
    controlled=bool(cfg.get('controlled_pipeline'))
    def add(role,rid,row,cap,kv=None,first=None,ignore=False,cleanup=False,action=None):
        cap,ignore=output_contract(cfg,role,rid,cap,ignore)
        ids=row['input_ids']['qwen8']+([first] if first is not None else [])
        extra={}
        if action is not None:
            extra=dict(axis_action=action['kind'],axis_cut=action['cut'])
            if kv is not None:kv=dict(kv,BI0_axis_action=action['kind'],BI0_axis_cut=action['cut'])
        if cfg.get('credit_event_instrumentation',True):
            append(folder/'request_submissions.jsonl',dict(request_id=rid,role=role,time=time.monotonic(),
                scope='Coordinator enqueue to the ingress sender. Not worker acceptance or GPU start.'))
        sender.send(role,dict(kind='cleanup_remote' if cleanup else 'add',request_id=rid,
            input_ids=ids,original_length=row['input_tokens'],cap=cap,kv_params=kv,ignore_eos=ignore,**extra))
    def certify_idle():
        assert controlled and native
        sender.send('decode',dict(kind='certify_idle'))
        assert channels['decode'].poll(5),'D idle certificate timeout'
        value=channels['decode'].recv();assert value['kind']=='idle_certificate',value
        append(folder/'idle_certificates.jsonl',value)
        return value['passed']
    try:
        write(folder/'status.json',dict(status='STARTING',time=time.time(),pid=os.getpid()))
        for role,gpu in zip(roles,cfg['gpus']):
            address=str(directory/role);listener=Listener(address,family='AF_UNIX')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),VLLM_NIXL_SIDE_CHANNEL_HOST='127.0.0.1')
            from ..dense_ar_bi0_shortanswer_pd.control_ports import choose
            port=choose(control_ports.values());control_ports[role]=port
            env['VLLM_NIXL_SIDE_CHANNEL_PORT']=str(port)
            write(folder/'control_ports.json',dict(ports=control_ports,
                scope='Availability checked outside the OS ephemeral range. This reduces startup collision risk, not serving time.'))
            handle=(folder/f'{role}.log').open('w');handles.append(handle)
            command=[str(PYTHON),'-u','-m',cfg.get('worker_module','poc_flashvep.dense_ar_bi0_axis_rootcause.standard_worker'),str(cfg_path),'--role',role,'--address',address]
            child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT);children.append(child)
            listener._listener._socket.settimeout(600);channel=listener.accept();listener.close();channels[role]=channel
            assert channel.poll(600),'Standard engine startup timeout'
            ready=channel.recv();assert ready['kind']=='ready'
            assert ready['scheduler_B']==cfg['token_budget'] and ready['scheduler_S']==cfg['max_num_seqs']
            assert ready['chunked_prefill'];write(folder/f'actual_runtime_{role}.json',ready)
        sender=Ingress(channels)
        # Real maximum/minimum-length controls, excluded from the measured trace.
        warm=[max(data,key=lambda r:r['input_tokens']),min(data,key=lambda r:r['input_tokens'])]
        for index,row in enumerate(warm):
            rid=f'warm{index}';done=False;start=time.monotonic()
            add('prefill' if native else roles[index%len(roles)],rid,row,16,ignore=True)
            while not done:
                for role,channel in channels.items():
                    while channel.poll():
                        message=channel.recv()
                        if message['kind']!='output' or not message['finished']:continue
                        if role=='prefill':
                            first=message['output_ids'][0]
                            add('decode',rid,row,15 if reuse else 16,message['kv_params'],first if reuse else None,ignore=True)
                        else:done=True
                assert time.monotonic()-start<300
                assert all(p.poll() is None for p in children)
                sender.check();time.sleep(.0002)
        for role in roles:sender.send(role,dict(kind='boundary'))
        for channel in channels.values():assert channel.poll(30) and channel.recv()['kind']=='boundary_ready'
        started=time.monotonic();events={};ids={};arrivals={};states={};completed={};admitted={};source_first={};i=0;last_sample=started
        P_pending=collections.deque();D_pending=collections.deque();P_active=None;D_active=None;source_owned=set();actions={}
        if controlled:
            assert native and not reuse
            write(folder/'controlled_protocol.json',dict(P_actual_request_batch=1,D_suffix_batch=1,D_decode_batch=1,
                Source_credit_limit=cfg.get('source_credit_limit',4),reservation='No new D admission before device-wide idle certificate.',
                scope='Mechanism control, not the normal continuous Native/Agg baseline. Source admission does not wait for a D reservation ACK.'))
        append(folder/'measurement_begin.jsonl',dict(time=started,protocol=cfg.get('protocol'),method=cfg['method']))
        wait_previous=None;wait_left=started;wait_signature=None
        def wait_state():
            return (tuple(P_pending),P_active,len(source_owned),D_active,len(D_pending))
        def signature():
            return (P_pending[0] if P_pending else None,P_pending[-1] if P_pending else None,
                    len(P_pending),P_active,len(source_owned),D_active,len(D_pending))
        while i<len(data) or len(completed)<len(data):
            now=time.monotonic()
            while i<len(data) and now>=started+cfg['arrival_offsets'][i]:
                if cfg.get('serial') and i>len(completed):break
                rid=f'q{i}';arrival=now if cfg.get('serial') else started+cfg['arrival_offsets'][i]
                arrivals[rid]=arrival;states[rid]='P_PENDING' if native else roles[i%len(roles)]
                if controlled:P_pending.append(i)
                else:add('prefill' if native else roles[i%len(roles)],rid,data[i],cfg['output_cap'],ignore=cfg.get('ignore_eos',False))
                append(folder/'offered_requests.jsonl',dict(request_id=rid,prompt_id=data[i]['prompt_id'],arrival=arrival,enqueue=now,task=data[i].get('task')));i+=1
            for role,channel in channels.items():
                while channel.poll():
                    message=channel.recv();rid=message.get('request_id','');event_role=role
                    if rid.startswith('cleanup'):continue
                    if message['kind']=='accepted':
                        admitted.setdefault(rid,message['time'])
                        append(folder/'admissions.jsonl',dict(request_id=rid,role=role,
                            worker_accepted=message['time'],coordinator_received=time.monotonic()))
                        continue
                    assert rid.startswith('q') and rid[1:].isdigit(),message
                    index=int(rid[1:]);row=data[index]
                    if role=='prefill':
                        assert message['finished'] and message['count']==1
                        first=message['output_ids'][0];source_first[rid]=first
                        append(folder/'handoffs.jsonl',dict(request_id=rid,source_ready=message['time'],received=time.monotonic(),params=message['kv_params']))
                        if controlled:
                            assert P_active==rid;P_active=None
                            D_pending.append((index,message['kv_params']));states[rid]='D_ADMISSION_PENDING'
                            continue
                        if reuse:
                            delivered=time.monotonic();ids[rid]=[first]
                            events[rid]=[dict(time=delivered,worker_time=message['time'],count=1,source='P_FIRST_SAMPLE')]
                        if reuse and ((first in eos and not cfg.get('ignore_eos')) or cfg['output_cap']==1):
                            add('decode','cleanup'+rid,row,1,message['kv_params'],cleanup=True)
                            message=dict(kind='output',count=0,finished=True,new_ids=[],output_ids=[],finish_reason='stop' if first in eos else 'length',stop_reason=first)
                            event_role='decode'
                        else:
                            states[rid]='D_IMPORT_OR_ADMISSION'
                            add('decode',rid,row,cfg['output_cap']-1 if reuse else cfg['output_cap'],message['kv_params'],first if reuse else None,ignore=cfg.get('ignore_eos',False))
                            continue
                    delivered=time.monotonic();tokens=ids.setdefault(rid,[])
                    tokens.extend(message['new_ids'])
                    if message['new_ids']:
                        events.setdefault(rid,[]).append(dict(time=delivered,worker_time=message.get('time',delivered),count=len(tokens),source=event_role))
                        if controlled and rid in source_owned:
                            append(folder/'credit_events.jsonl',dict(event='RELEASE_FIRST_CLIENT_TOKEN',request_id=rid,time=delivered,owned_before=len(source_owned)))
                            source_owned.discard(rid)
                    states[rid]='D_DECODE_ACTIVE' if native else event_role
                    if not message['finished']:continue
                    assert tokens and len(tokens)<=cfg['output_cap']
                    token_events=events[rid];count=len(tokens);first_time=token_events[0]['time']
                    record=dict(sample_id=index,prompt_id=row['prompt_id'],task=row.get('task'),
                        input_tokens=row['input_tokens'],input_ids_SHA256=row.get('input_ids_SHA256'),output_ids=tokens,
                        output_tokens=count,arrival=arrivals[rid],first_token=first_time,finish=delivered,
                        TTFT_ms=1000*(first_time-arrivals[rid]),TPOT_ms=1000*(delivered-first_time)/(count-1) if count>1 else None,
                        E2E_ms=1000*(delivered-arrivals[rid]),token_events=token_events,
                        finish_reason=message['finish_reason'],stop_reason=message.get('stop_reason'),
                        cap_hit=message['finish_reason']=='length',method=cfg['method'],
                        output_contract='DIAGNOSTIC_FIXED_LENGTH_REPLAY' if cfg.get('diagnostic_controlled_output_replay') else 'NATURAL_EOS',
                        timing_scope='Scheduled open-loop arrival to coordinator token delivery. No HTTP. First interval includes handoff for first-token-reuse Native.')
                    append(folder/'requests.jsonl',record);completed[rid]=record;states[rid]='FINISHED'
                    if controlled:
                        assert D_active==rid;D_active=None
                    print('DONE',cfg['name'],index,row.get('task'),count,round(record['TTFT_ms'],2),flush=True)
            if controlled:
                if D_active is None and D_pending and certify_idle():
                    index,kv=D_pending.popleft();rid=f'q{index}'
                    add('decode',rid,data[index],cfg['output_cap'],kv,ignore=cfg.get('ignore_eos',False),action=actions[rid])
                    D_active=rid;states[rid]='D_IMPORT_OR_PREFILL'
                if P_active is None and P_pending and len(source_owned)<cfg.get('source_credit_limit',4) and (not cfg.get('strict_serial_pipeline') or (D_active is None and not D_pending and not source_owned)):
                    index=P_pending.popleft();rid=f'q{index}';kind=cfg.get('axis_kind','Native');cut=cfg.get('axis_cut',0)
                    mode=cfg.get('axis_mode','STATIC')
                    reasons=[]
                    if mode=='WINDOW':
                        from ..dense_ar_bi0_shortanswer_pd.quiescent_policy import rejection_reasons
                        reasons=rejection_reasons(D_active,len(D_pending),len(source_owned),len(P_pending),cfg.get('minimum_waiting_backlog',1))
                        if not reasons and not certify_idle():reasons.append('DEVICE_OR_NIXL_NOT_QUIESCENT')
                        if reasons:kind='Native';cut=0
                    action=dict(kind=kind,cut=cut);actions[rid]=action
                    append(folder/'axis_decisions.jsonl',dict(request_id=rid,time=time.monotonic(),action=action,
                        P_waiting=len(P_pending),D_active=D_active,D_pending=len(D_pending),
                        source_outstanding=len(source_owned),online_observable=True,rejection_reasons=reasons,
                        scope='Window admission uses current state only. Single P admission prevents future arrivals from producing another D-ready request before this P completion.'))
                    append(folder/'credit_events.jsonl',dict(event='ACQUIRE_P_DISPATCH',request_id=rid,time=time.monotonic(),owned_before=len(source_owned)))
                    source_owned.add(rid);P_active=rid
                    add('prefill',rid,data[index],cfg['output_cap'],ignore=cfg.get('ignore_eos',False),action=action)
                    states[rid]='P_ACTIVE'
            now=time.monotonic()
            if controlled and cfg.get('credit_event_instrumentation',True):
                current=signature()
                if current!=wait_signature:
                    if wait_previous is not None:
                        pending,pactive,owned,dactive,dpending=wait_previous
                        append(folder/'P_wait_segments.jsonl',dict(start=wait_left,end=now,pending=list(pending),
                            P_active=pactive,source_owned=owned,source_credit_limit=cfg.get('source_credit_limit',4),
                            D_active=dactive,D_pending=dpending,
                            reason='P_FORWARD_OUTSTANDING' if pactive is not None else 'SOURCE_CREDIT_FULL' if pending and owned>=cfg.get('source_credit_limit',4) else 'SERIAL_ADMISSION_BARRIER' if pending and cfg.get('strict_serial_pipeline') and (dactive is not None or dpending) else 'HOST_ELIGIBLE_GAP' if pending else 'NO_PENDING',
                            scope='Coordinator state interval. P active includes IPC, preparation and GPU work. Not GPU-exclusive time.'))
                    wait_previous=wait_state();wait_signature=current;wait_left=now
            if now-last_sample>=.05:
                append(folder/'coordinator_queues.jsonl',dict(time=now,arrived=i,completed=len(completed),phases=dict(collections.Counter(states.values())),
                    P_pipeline_waiting=len(P_pending) if controlled else None,D_pipeline_waiting=len(D_pending) if controlled else None,
                    P_pipeline_active=int(P_active is not None) if controlled else None,D_pipeline_active=int(D_active is not None) if controlled else None));last_sample=now
            sender.check();assert all(p.poll() is None for p in children)
            assert now-started<cfg.get('serving_timeout',7200)
            time.sleep(.0002)
        write(folder/'window.json',dict(start=started,end=time.monotonic(),last_arrival=max(arrivals.values()),
            finite_cohort=True,sustainable_capacity=False))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time(),completed=len(completed)))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if sender is not None:
            try:
                for role in roles:sender.send(role,dict(kind='stop'))
                sender.close()
            except BaseException as error:
                write(folder/'ingress_shutdown_error.json',dict(error=repr(error),time=time.time()))
        exits=[]
        for child in children:
            forced=False
            try:child.wait(45)
            except subprocess.TimeoutExpired:
                forced=True;child.terminate()
                try:child.wait(10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            exits.append(dict(pid=child.pid,returncode=child.returncode,forced=forced))
        for handle in handles:handle.close()
        write(folder/'worker_exits.json',dict(records=exits,passed=bool(exits) and all(r['returncode']==0 and not r['forced'] for r in exits)))
        for path,rows in BUFFERED.items():
            with path.open('w') as stream:
                for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        BUFFERED.clear()


if __name__=='__main__':main()
