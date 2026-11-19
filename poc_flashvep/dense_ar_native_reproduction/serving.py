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
from .ingress import Ingress


def append(path,row):
    with path.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    cfg_path=Path(sys.argv[1]);cfg=json.loads(cfg_path.read_text());folder=ROOT/'raw'/cfg['name']
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    assert len(data)==cfg['count'];native=cfg['method']=='Native' and not cfg.get('reference_single');reuse=cfg.get('protocol','OFFICIAL_PROXY') in ('FIRST_TOKEN_REUSE','FIRST_TOKEN_REUSE_BOUNDED')
    roles=('single',) if cfg.get('reference_single') else (('prefill','decode') if native else ('replica0','replica1'))
    channels={};children=[];handles=[];sender=None;directory=Path(tempfile.mkdtemp(prefix='native_real200_'))
    eos=json.loads((MODEL/'generation_config.json').read_text())['eos_token_id']
    def add(role,rid,row,cap,kv=None,first=None,ignore=False,cleanup=False):
        ids=row['input_ids']['qwen8']+([first] if first is not None else [])
        sender.send(role,dict(kind='cleanup_remote' if cleanup else 'add',request_id=rid,
            input_ids=ids,original_length=row['input_tokens'],cap=cap,kv_params=kv,ignore_eos=ignore))
    try:
        write(folder/'status.json',dict(status='STARTING',time=time.time(),pid=os.getpid()))
        for role,gpu in zip(roles,cfg['gpus']):
            address=str(directory/role);listener=Listener(address,family='AF_UNIX')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),VLLM_NIXL_SIDE_CHANNEL_HOST='127.0.0.1')
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',0));env['VLLM_NIXL_SIDE_CHANNEL_PORT']=str(probe.getsockname()[1])
            handle=(folder/f'{role}.log').open('w');handles.append(handle)
            command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_native_reproduction.standard_worker',str(cfg_path),'--role',role,'--address',address]
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
        append(folder/'measurement_begin.jsonl',dict(time=started,protocol=cfg.get('protocol'),method=cfg['method']))
        while i<len(data) or len(completed)<len(data):
            now=time.monotonic()
            while i<len(data) and now>=started+cfg['arrival_offsets'][i]:
                if cfg.get('serial') and i>len(completed):break
                rid=f'q{i}';arrival=now if cfg.get('serial') else started+cfg['arrival_offsets'][i]
                arrivals[rid]=arrival;states[rid]='P_PENDING' if native else roles[i%len(roles)]
                add('prefill' if native else roles[i%len(roles)],rid,data[i],cfg['output_cap'],ignore=cfg.get('ignore_eos',False))
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
                        timing_scope='Scheduled open-loop arrival to coordinator token delivery. No HTTP. First interval includes handoff for first-token-reuse Native.')
                    append(folder/'requests.jsonl',record);completed[rid]=record;states[rid]='FINISHED'
                    print('DONE',cfg['name'],index,row.get('task'),count,round(record['TTFT_ms'],2),flush=True)
            now=time.monotonic()
            if now-last_sample>=.05:
                append(folder/'coordinator_queues.jsonl',dict(time=now,arrived=i,completed=len(completed),phases=dict(collections.Counter(states.values()))));last_sample=now
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


if __name__=='__main__':main()
