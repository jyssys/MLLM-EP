"""Use native vLLM NIXL P/D engines. The coordinator does not implement KV transfer."""
import argparse
import subprocess
import tempfile
import socket
from multiprocessing.connection import Listener,Client
from .common import *

def stage(cfg,role,address):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_NIXL_SIDE_CHANNEL_HOST']='127.0.0.1'
    os.environ['VLLM_NIXL_SIDE_CHANNEL_PORT']=os.environ['BC_NIXL_SIDE_PORT']
    import torch
    from vllm import LLM,SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    native_stats=[];native_done=[];get_stats=NixlBaseConnectorWorker.get_kv_connector_stats
    def collect_stats(self):
        value=get_stats(self)
        if value is not None:native_stats.append(dict(time=time.monotonic(),data={k:list(v) for k,v in value.data.items()}))
        return value
    NixlBaseConnectorWorker.get_kv_connector_stats=collect_stats
    pop_done=NixlBaseConnectorWorker._pop_done_transfers
    def observe_done(self,transfers):
        done=pop_done(self,transfers)
        if done:
            native_done.append(dict(time=time.monotonic(),request_ids=sorted(done),
                pending_stats={k:list(v) for k,v in self.xfer_stats.data.items()},
                evidence='STOCK_NIXL_COMPLETED_TRANSFER_OBSERVATION_NOT_A_SIMULATION'))
        return done
    NixlBaseConnectorWorker._pop_done_transfers=observe_done
    torch.set_num_threads(1)
    channel=Client(address,family='AF_UNIX')
    checkpoint=json.loads((ROOT/'configs/dense_ar_checkpoint.json').read_text())['path']
    llm=LLM(model=checkpoint,dtype='bfloat16',max_model_len=4096,max_num_seqs=32,
        max_num_batched_tokens=32768,enable_prefix_caching=False,enable_chunked_prefill=True,
        async_scheduling=False,gpu_memory_utilization=.65,seed=cfg['seed'],
        kv_transfer_config=dict(kv_connector='NixlConnector',kv_role='kv_producer' if role=='prefill' else 'kv_consumer'),
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32]})
    engine=llm.llm_engine
    runner=engine.model_executor.driver_worker.worker.model_runner
    channel.send(dict(kind='ready',role=role,pid=os.getpid(),uuid=str(torch.cuda.get_device_properties(0).uuid),
        engine='vLLM native NIXL',source='vLLM 0.26.0',original_prompt_preserved=True,
        KV_dtype=str(runner.kv_cache_dtype),KV_blocks=runner.kv_cache_config.num_blocks,
        KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),
        unique_model_parameters=sum(p.numel() for p in runner.model.parameters())))
    running=True;steps=[];snapshots=[];last_sample=0.
    try:
        while running:
            while channel.poll():
                cmd=channel.recv()
                if cmd['kind']=='stop':running=False;break
                params=SamplingParams(max_tokens=1 if role=='prefill' else cmd['cap'],temperature=0,
                    ignore_eos=True,seed=cmd['seed'],extra_args=dict(kv_transfer_params=
                    dict(do_remote_decode=True,do_remote_prefill=False) if role=='prefill' else cmd['kv_params']))
                engine.add_request(cmd['request_id'],{'prompt_token_ids':cmd['input_ids']},params)
                channel.send(dict(kind='accepted',role=role,request_id=cmd['request_id'],time=time.monotonic()))
            if not running:break
            t=time.monotonic()
            outputs=engine.step()
            end=time.monotonic()
            if outputs:steps.append(dict(start=t,end=end,outputs=len(outputs),scope='Host engine step, not kernel busy'))
            for out in outputs:
                value=out.outputs[0]
                channel.send(dict(kind='output',role=role,request_id=out.request_id,time=time.monotonic(),
                    count=len(value.token_ids),finished=out.finished,
                    output_ids=list(value.token_ids) if out.finished else None,
                    finish_reason=value.finish_reason,
                    reported_cached_prompt_tokens=getattr(out,'num_cached_tokens',None) if out.finished else None,
                    kv_params=out.kv_transfer_params if out.finished else None))
            if end-last_sample>.5:
                last_sample=end
                core=getattr(engine.engine_core,'engine_core',None);sched=getattr(core,'scheduler',None)
                if sched is not None:
                    snapshots.append(dict(time=end,running=len(sched.running),waiting=len(sched.waiting),
                        KV_usage=sched.kv_cache_manager.usage))
            if not engine.has_unfinished_requests():time.sleep(.0005)
    finally:
        # The offline LLM API can suppress periodic metric collection.
        # Drain the stock connector after request timing, without changing transfer behavior.
        from vllm.distributed.kv_transfer import get_kv_transfer_group,has_kv_transfer_group
        final_stats=None
        if has_kv_transfer_group():
            value=get_kv_transfer_group().get_kv_connector_stats()
            if value is not None:final_stats={k:list(v) for k,v in value.data.items()}
        write(ROOT/'raw'/cfg['name']/f'ar_native_{role}.json',dict(steps=steps,queue_samples=snapshots,
            role=role,NIXL_stats_intervals=native_stats,NIXL_final_stats=final_stats,NIXL_done_events=native_done,
            scope='Native engine host spans and returned NIXL telemetry. Transfer durations are not exclusive E2E components.'))
        shutdown=getattr(engine.engine_core,'shutdown',None)
        if shutdown:shutdown()
        channel.close()

def coordinator(cfg):
    from ..looped_pd_abc.nvml_watch import GPUWatch
    folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start();children=[];connections={};handles=[]
    directory=Path(tempfile.mkdtemp(prefix='bc_ar_pd_',dir='/tmp'))
    try:
        for role,gpu in zip(('prefill','decode'),cfg['gpus']):
            address=str(directory/role);listener=Listener(address,family='AF_UNIX')
            env=environment([gpu]);env['BC_WORKER']='0'
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',0));env['BC_NIXL_SIDE_PORT']=str(probe.getsockname()[1])
            handle=(folder/f'{role}.log').open('w');handles.append(handle)
            child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_bottleneck_crossover.ar_pd',
                os.environ['ARC_CONFIG'],'--role',role,'--address',address],env=env,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            children.append(child);listener._listener._socket.settimeout(120)
            channel=listener.accept();listener.close();connections[role]=channel
            if not channel.poll(300):raise TimeoutError('Native vLLM NIXL engine startup')
            ready=channel.recv();assert ready['kind']=='ready',ready
            assert ready['uuid'].removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
            write(folder/f'{role}_ready.json',ready)
        rows=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
        assert len(rows)==cfg['count'] and all(len(r['input_ids']['qwen'])+cap<=4096 for r,cap in zip(rows,cfg['request_caps']))
        if cfg.get('warmup',True):
            warm=[]
            for index,row in enumerate(rows[:2]):
                rid=f'warm_{index}';done=False;deadline=time.monotonic()+120
                connections['prefill'].send(dict(kind='add',request_id=rid,input_ids=row['input_ids']['qwen'],cap=2,seed=cfg['seed']-10000+index))
                while not done:
                    for role,channel in connections.items():
                        while channel.poll():
                            message=channel.recv();assert message['request_id']==rid
                            if message['kind']=='accepted' or not message['finished']:continue
                            if role=='prefill':
                                assert message['kv_params'] and message['kv_params'].get('remote_block_ids')
                                connections['decode'].send(dict(kind='add',request_id=rid,input_ids=row['input_ids']['qwen'],
                                    cap=2,seed=cfg['seed']-10000+index,kv_params=message['kv_params']))
                            else:
                                assert message['count']==2;done=True
                                warm.append(dict(input_ids=row['input_ids']['qwen'],output_ids=message['output_ids'],warmup=True))
                    if time.monotonic()>deadline:raise TimeoutError('Native NIXL warm-up watchdog')
                    time.sleep(.0002)
            write(folder/'warmup_outputs.json',warm)
        first={};events={};completed={};phases={};p_done={};offsets=cfg['arrival_offsets'];arrivals={}
        start=time.monotonic();i=0;last_observation=0.
        os.environ['ABC_MEASURED_ACTIVE']='1'
        while i<len(rows) or len(completed)<len(rows):
            now=time.monotonic()
            while i<len(rows) and now>=start+offsets[i]:
                if cfg.get('calibration_serial') and i>len(completed):break
                rid=f'q{i}';phases[rid]='prefill'
                arrivals[rid]=now if cfg.get('calibration_serial') else start+offsets[i]
                connections['prefill'].send(dict(kind='add',request_id=rid,input_ids=rows[i]['input_ids']['qwen'],cap=cfg['request_caps'][i],seed=cfg['seed']+i))
                i+=1
            for role,channel in connections.items():
                while channel.poll():
                    message=channel.recv();rid=message['request_id'];index=int(rid[1:])
                    if message['kind']=='accepted':continue
                    if role=='prefill':
                        if not message['finished']:continue
                        params=message['kv_params'];assert params and params.get('remote_block_ids'),'No actual NIXL remote KV handoff'
                        append(folder/'native_handoffs.jsonl',dict(request_id=rid,time=message['time'],params=params))
                        p_done[rid]=message['time'];phases[rid]='decode'
                        # Native demo semantics: repeat the original prompt on D with the exported KV descriptor.
                        # D performs its normal boundary recompute and all requested sampling.
                        connections['decode'].send(dict(kind='add',request_id=rid,input_ids=rows[index]['input_ids']['qwen'],
                            cap=cfg['request_caps'][index],seed=cfg['seed']+index,kv_params=params))
                    else:
                        delivered=time.monotonic()
                        first.setdefault(rid,delivered);events.setdefault(rid,[]).append(dict(time=delivered,count=message['count'],worker_time=message['time']))
                        if not message['finished']:continue
                        finish=delivered;count=message['count'];assert count==cfg['request_caps'][index]
                        record=dict(model='qwen',arm='Native_vLLM_NIXL_PD',sample_id=index,prompt_id=rows[index]['prompt_id'],
                            input_ids=rows[index]['input_ids']['qwen'],output_ids=message['output_ids'],
                            arrival=arrivals[rid],first_token=first[rid],finish=finish,
                            ttft_ms=1000*(first[rid]-arrivals[rid]),tpot_ms=1000*(finish-first[rid])/max(1,count-1),
                            e2e_ms=1000*(finish-arrivals[rid]),output_tokens=count,token_events=events[rid],
                            finish_reason=message['finish_reason'],warmup=False,
                            D_reported_cached_prompt_tokens=message.get('reported_cached_prompt_tokens'),
                            evidence='MEASURED_NATIVE_VLLM_NIXL_PD',engine_difference='Native vLLM AR. Not RLT.',
                            transport='Native NIXL pull',P_computes_first_sample_but_D_resamples=True)
                        record['timing_scope']='Serial admission to coordinator delivery.' if cfg.get('calibration_serial') else 'Scheduled open-loop arrival to coordinator delivery.'
                        record['timing_scope']+=' Includes IPC and polling. No HTTP.'
                        record['scheduler_scope']='Native vLLM permits chunked prefill/decode batching. This is an engine-confounded AR diagnostic, not a matched RLT partition baseline.'
                        completed[rid]=record;phases[rid]='finished';append(folder/'requests.jsonl',record)
            if now-last_observation>1:
                last_observation=now;append(folder/'queue_observations.jsonl',dict(time=now,since_start=now-start,delivered=i,
                    completed=len(completed),outstanding=i-len(completed),pending_phases=dict(__import__('collections').Counter(phases.values()))))
            if any(c.poll() is not None for c in children):raise RuntimeError('Native AR worker exited')
            if now-start>cfg.get('timeout',1800):raise TimeoutError('Native AR PD serving watchdog')
            time.sleep(.0002)
        write(folder/'window.json',dict(start=start,end=time.monotonic(),last_arrival=max(arrivals.values()),
            arrival_pattern='CLOSED_LOOP_SERIAL_CALIBRATION' if cfg.get('calibration_serial') else 'OPEN_LOOP_POISSON'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        os.environ['ABC_MEASURED_ACTIVE']='0'
        for channel in connections.values():
            try:channel.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError):pass
        for child in children:
            try:child.wait(30)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for h in handles:h.close()
        monitor.stop()

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--role');p.add_argument('--address');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    if a.role:stage(cfg,a.role,a.address)
    else:coordinator(cfg)

if __name__=='__main__':main()
