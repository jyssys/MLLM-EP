"""Run official vLLM servers under a finite, owned GPU lease."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import urllib.request

from .common import ROOT,REPO,PYTHON,MODEL_ID,resources,write,append,digest,initialize,preserve,experiment_deadline


def command(role,port,ec_path,side_port,observe=False,encoder_native_ops=False,eager_reference=False,compiled_encoder=False,encoder_token_budget=114688,encoder_whole_request=False,encoder_sync_scheduling=False,mm_cache_gb=1,renderer_workers=1,ec_transport='cpu',encoder_api='http://127.0.0.1:18840',token_split=False,kv_cache_gib=42,gpu_memory_utilization=.9,prefix_probe=False,fixed_reduction=False,deterministic_reductions=False):
    assert mm_cache_gb>=0 and renderer_workers>=1
    assert renderer_workers==1 or mm_cache_gb==0,'Official renderer threads require a disabled processor cache'
    assert ec_transport in ('cpu','cuda_ipc')
    assert ec_transport=='cpu' or observe,'CUDA IPC requires the fixed owned-worker RPC endpoints'
    assert not token_split or observe and role in ('prefill','decode')
    assert not prefix_probe or observe and role in ('pd','decode') and not token_split
    assert not fixed_reduction or observe and role in ('prefill','decode','pd','agg','reference') and not eager_reference
    assert not deterministic_reductions or observe and role in ('prefill','decode','pd','agg','reference') and not eager_reference
    assert 0<kv_cache_gib<=42 and 0<gpu_memory_utilization<=.95
    model=json.loads((ROOT/'state/model_download_complete.json').read_text())
    args=[str(PYTHON),'-m','vllm.entrypoints.openai.api_server','--model',model['directory'],
        '--served-model-name',MODEL_ID,'--dtype','bfloat16','--port',str(port),'--host','127.0.0.1',
        '--max-model-len','32768','--max-num-batched-tokens','8192','--max-num-seqs','1024',
        '--kv-cache-memory-bytes',str(int(kv_cache_gib*1024**3)),
        '--gpu-memory-utilization',str(gpu_memory_utilization),'--block-size','128','--no-enable-prefix-caching',
        '--enable-chunked-prefill','--mm-processor-cache-gb',str(mm_cache_gb),
        '--renderer-num-workers',str(renderer_workers),'--skip-mm-profiling',
        '--limit-mm-per-prompt',json.dumps(dict(image=256,video=0)),
        '--allowed-local-media-path',str(ROOT/'media'),'--enable-request-id-headers',
        '--generation-config','vllm','--seed','2026112401']
    if role=='encoder':
        assert encoder_token_budget>0
        index=args.index('--max-num-batched-tokens')+1;args[index]=str(encoder_token_budget)
        args+=['--mm-encoder-only']
        if encoder_whole_request:
            args[args.index('--enable-chunked-prefill')]='--no-enable-chunked-prefill'
        if encoder_sync_scheduling:args+=['--no-async-scheduling']
        if not compiled_encoder:args+=['--enforce-eager']
        if encoder_native_ops:args+=['--compilation-config',json.dumps(dict(custom_ops=['none']))]
    if eager_reference:
        assert role=='reference' or token_split and role in ('prefill','decode'), (
            'The eager teacher is an explicit correctness diagnostic, not a serving baseline')
        args+=['--enforce-eager','--compilation-config',json.dumps(dict(custom_ops=['none']))]
    if fixed_reduction or deterministic_reductions:
        settings={}
        if fixed_reduction:settings['dynamic_scale_rblock']=False
        if deterministic_reductions:
            settings.update(deterministic=True,combo_kernels=True,benchmark_combo_kernel=False)
        args+=['--compilation-config',json.dumps(dict(inductor_compile_config=settings))]
    if role in ('encoder','prefill','pd','decode','reference_export'):
        # Use the installed official debug connector. Its CPU staging is not a NIXL EC transfer.
        cfg=dict(ec_connector='ECExampleConnector',ec_role='ec_producer' if role in ('encoder','reference_export') else 'ec_consumer',
            ec_connector_extra_config=dict(shared_storage_path=str(ec_path)))
        if cfg['ec_role']=='ec_consumer':
            cfg.update(ec_connector='StrictExampleConnector',
                ec_connector_module_path='poc_flashvep.mllm_epd_prefill_heavy_token_split.strict_ec')
        if ec_transport=='cuda_ipc':
            cfg.update(ec_connector='CUDAIPCEncoderConnector',
                ec_connector_module_path='poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec')
            cfg['ec_connector_extra_config']['producer_api']=encoder_api
        args+=['--ec-transfer-config',json.dumps(cfg)]
    if role in ('prefill','decode','reference_prefill','reference_decode'):
        cfg=dict(kv_connector='NixlConnector',kv_role='kv_producer' if role in ('prefill','reference_prefill') else 'kv_consumer')
        if token_split:
            cfg.update(kv_connector='ExactTokenNixlConnector',
                kv_connector_module_path='poc_flashvep.mllm_epd_prefill_heavy_token_split.token_nixl')
        args+=['--kv-transfer-config',json.dumps(cfg)]
    if observe:
        args[2]='poc_flashvep.mllm_epd_prefill_heavy_token_split.server_entry'
        args+=['--worker-extension-cls',
            'poc_flashvep.mllm_epd_prefill_heavy_token_split.worker_observer.CampaignWorkerExtension',
            '--middleware','poc_flashvep.mllm_epd_prefill_heavy_token_split.host_timeline.APITimelineMiddleware']
    return args


def validate_co_resident_facts(lease,gpu,facts,uid):
    assert lease['gpu']==gpu and lease['role']=='encoder' and lease['ready'] and not lease['finished']
    assert lease['sid']==lease['pid'] and facts,'The original encoder owner must remain present'
    assert all(row['gpu']==gpu and row['uid']==uid and row['sid']==lease['sid'] for row in facts), 'Foreign or additional GPU owner left untouched'


def inspect_co_resident(owner,gpu,name,ec_path,encoder_api):
    lease_path=ROOT/'state'/('server_'+name+'.json')
    lease=json.loads(lease_path.read_text())
    facts=[]
    for row in owner.processes():
        if row['gpu']!=gpu:continue
        proc=Path('/proc',str(row['pid']))
        facts.append(dict(row,uid=proc.stat().st_uid,sid=os.getsid(row['pid'])))
    validate_co_resident_facts(lease,gpu,facts,os.getuid())
    proc=Path('/proc',str(lease['pid']))
    assert proc.stat().st_uid==os.getuid() and os.getsid(lease['pid'])==lease['sid']
    assert lease['command'][2].encode() in (proc/'cmdline').read_bytes()
    assert lease['EC_path']==str(ec_path) and encoder_api==f'http://127.0.0.1:{lease["port"]}'
    return dict(name=name,lease_SHA256=digest(lease_path),pid=lease['pid'],sid=lease['sid'],observed=facts)


def serve(name,gpu,role,port,ec_path,side_port,observe=False,encoder_native_ops=False,legacy_runner=False,eager_reference=False,compiled_encoder=False,encoder_token_budget=114688,encoder_whole_request=False,encoder_sync_scheduling=False,api_switch_interval=None,mm_cache_gb=1,renderer_workers=1,ec_transport='cpu',encoder_api='http://127.0.0.1:18840',token_split=False,kv_cache_gib=42,gpu_memory_utilization=.9,co_resident_with=None,prefix_probe=False,fixed_reduction=False,deterministic_reductions=False):
    initialize();owner=resources()
    stop=ROOT/'state'/('stop_server_'+name)
    record=ROOT/'state'/('server_'+name+'.json')
    assert not stop.exists(),'Use a new run name after a prior stop'
    if record.exists():
        old=json.loads(record.read_text());assert old.get('finished'),('Existing lease',name)
    ec_path.mkdir(parents=True,exist_ok=True)
    child=None;request_stop=False;started=time.time();last_sample=0
    def on_signal(*_):
        nonlocal request_stop
        request_stop=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    assert not co_resident_with or role=='pd' and 0<kv_cache_gib<=16 and gpu_memory_utilization<=.5
    cmd=command(role,port,ec_path,side_port,observe=observe,encoder_native_ops=encoder_native_ops,eager_reference=eager_reference,compiled_encoder=compiled_encoder,encoder_token_budget=encoder_token_budget,encoder_whole_request=encoder_whole_request,encoder_sync_scheduling=encoder_sync_scheduling,mm_cache_gb=mm_cache_gb,renderer_workers=renderer_workers,ec_transport=ec_transport,encoder_api=encoder_api,token_split=token_split,kv_cache_gib=kv_cache_gib,gpu_memory_utilization=gpu_memory_utilization,prefix_probe=prefix_probe,fixed_reduction=fixed_reduction,deterministic_reductions=deterministic_reductions)
    if observe:
        from .worker_observer import prepare_artifact
        prepare_artifact()
    from .source_snapshot import snapshot,materialize
    archived_source=snapshot('server_'+name)
    frozen_cwd=materialize(archived_source)
    env=dict(os.environ,PATH=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH',''),
        CUDA_VISIBLE_DEVICES=str(gpu),VLLM_NIXL_SIDE_CHANNEL_PORT=str(side_port),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',VLLM_BATCH_INVARIANT='0',
        VLLM_CACHE_ROOT=str(ROOT/'runtime_cache'/role),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        UCX_NET_DEVICES='all',VLLM_ALLOW_LONG_MAX_MODEL_LEN='1',MLLM_EPD_ROLE=role)
    env.update(MLLM_EPD_EC_PATH=str(ec_path),MLLM_EPD_METADATA_PATH=str(ec_path/'processed_prompts'),
        MLLM_EPD_EC_TRANSPORT=ec_transport)
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['MLLM_EPD_TOKEN_SPLIT']='1' if token_split else '0'
    env['MLLM_EPD_PREFIX_PROBE']='1' if prefix_probe else '0'
    env['MLLM_EPD_ALLOW_DECODE_PROBE']='1' if prefix_probe and role=='decode' else '0'
    if legacy_runner:env['VLLM_USE_V2_MODEL_RUNNER']='0'
    else:env.pop('VLLM_USE_V2_MODEL_RUNNER',None)
    env.pop('MLLM_EPD_API_SWITCH_INTERVAL',None)
    if api_switch_interval is not None:
        assert observe and .0001<=api_switch_interval<=.01
        env['MLLM_EPD_API_SWITCH_INTERVAL']=str(api_switch_interval)
    ninja=shutil.which('ninja',path=env['PATH'])
    assert ninja,'The actual server PATH must expose the installed ninja binary'
    assert subprocess.run(['ninja','--version'],env=env,capture_output=True).returncode==0
    data=dict(name=name,gpu=gpu,role=role,port=port,EC_path=str(ec_path),command=cmd,
        guardian=os.getpid(),guardian_sid=os.getsid(0),started=started,ready=False,finished=False,legacy_runner=legacy_runner,
        source_SHA256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        source_archive=archived_source['archive'],source_archive_SHA256=archived_source['archive_SHA256'],
        executed_source_cwd=str(frozen_cwd),source_execution='Verified extracted boot snapshot',
        GPU_info=subprocess.check_output(['nvidia-smi','-i',str(gpu),
            '--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True),
        compiler_preflight=dict(ninja=ninja,passed=True),
        API_switch_interval_requested=api_switch_interval,
        frontend_configuration=dict(mm_processor_cache_gb=mm_cache_gb,renderer_num_workers=renderer_workers),
        EC_transport=ec_transport,token_split_adapter_enabled=token_split,
        isolated_prefix_probe_enabled=prefix_probe,
        compiled_reduction_policy=('deterministic=True; dynamic_scale_rblock='+str(not fixed_reduction)
            if deterministic_reductions else 'dynamic_scale_rblock=False' if fixed_reduction else 'Original default'),
        deterministic_reductions=deterministic_reductions,
        deterministic_combo_benchmark=False if deterministic_reductions else 'Original default',
        KV_cache_GiB=kv_cache_gib,gpu_memory_utilization=gpu_memory_utilization,
        co_resident_with=co_resident_with,
        environment={k:env[k] for k in ('PATH','CUDA_VISIBLE_DEVICES','VLLM_NIXL_SIDE_CHANNEL_PORT','OMP_NUM_THREADS',
            'MKL_NUM_THREADS','VLLM_BATCH_INVARIANT','VLLM_CACHE_ROOT')},
        observer='Inactive fixed RPC endpoints plus NVML. CUPTI/state hooks require an explicit begin/end and CPC gate.'
            if observe else 'NVML only. Not a kernel-busy or GPU-exclusive measurement.')
    try:
        if co_resident_with:
            data['co_resident_owner']=inspect_co_resident(owner,gpu,co_resident_with,ec_path,encoder_api)
        else:owner.stop([gpu])
        with (ROOT/'logs'/('server_'+name+'.log')).open('ab') as log:
            child=subprocess.Popen(cmd,cwd=frozen_cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        data['pid']=child.pid;data['sid']=os.getsid(child.pid);write(record,data)
        append(ROOT/'resource_ledger.jsonl',dict(action='START_OWNED_SERVER',time=time.time(),**data))
        while child.poll() is None and not request_stop and not stop.exists() and time.time()<experiment_deadline():
            if not data['ready']:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as reply:
                        if reply.status==200:
                            data['ready']=True;data['ready_time']=time.time();write(record,data)
                except Exception:pass
                if time.time()-started>1500:raise TimeoutError('Server initialization exceeded 25 minutes')
            if time.time()-last_sample>=2:
                last_sample=time.time()
                sample=subprocess.run(['nvidia-smi','-i',str(gpu),
                    '--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True)
                append(ROOT/'raw'/('NVML_'+name+'.jsonl'),dict(time=time.time(),value=sample.stdout.strip()))
            time.sleep(.5)
        data['returncode']=child.poll();data['stop_requested']=request_stop or stop.exists()
    except BaseException as error:
        data['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            proc=Path('/proc')/str(child.pid)
            assert proc.stat().st_uid==os.getuid() and os.getsid(child.pid)==data['sid']==child.pid
            assert cmd[2].encode() in (proc/'cmdline').read_bytes()
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
        data['finished']=True;data['finished_time']=time.time();write(record,data)
        until=time.time()+45
        while not co_resident_with and any(r['gpu']==gpu for r in owner.processes()) and time.time()<until:time.sleep(.5)
        if not any(r['gpu']==gpu for r in owner.processes()):owner.restore([gpu])
        else:append(ROOT/'resource_ledger.jsonl',dict(action='RESTORE_DEFERRED_GPU_OCCUPIED',gpu=gpu,time=time.time()))
        preserve()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','serve','stop','status'])
    parser.add_argument('--name',required=True);parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--role',choices=['reference','reference_export','reference_prefill','reference_decode','agg','encoder','prefill','decode','pd'],default='reference')
    parser.add_argument('--port',type=int,default=18303);parser.add_argument('--side-port',type=int,default=19303)
    parser.add_argument('--ec-path',type=Path,default=Path('/dev/shm/mllm_epd_20261124_calibration'))
    parser.add_argument('--observe',action='store_true')
    parser.add_argument('--encoder-native-ops',action='store_true')
    parser.add_argument('--legacy-runner',action='store_true')
    parser.add_argument('--eager-reference',action='store_true')
    parser.add_argument('--compiled-encoder',action='store_true')
    parser.add_argument('--encoder-token-budget',type=int,default=114688)
    parser.add_argument('--encoder-whole-request',action='store_true')
    parser.add_argument('--encoder-sync-scheduling',action='store_true')
    parser.add_argument('--api-switch-interval',type=float)
    parser.add_argument('--mm-cache-gb',type=float,default=1)
    parser.add_argument('--renderer-workers',type=int,default=1)
    parser.add_argument('--ec-transport',choices=['cpu','cuda_ipc'],default='cpu')
    parser.add_argument('--encoder-api',default='http://127.0.0.1:18840')
    parser.add_argument('--token-split',action='store_true')
    parser.add_argument('--prefix-probe',action='store_true',help='Discarded prefix work in a pd helper or explicitly gated decode probe, not task completion')
    parser.add_argument('--fixed-reduction',action='store_true',help='One-factor compiled numerical control: dynamic_scale_rblock=False')
    parser.add_argument('--deterministic-reductions',action='store_true',help='Explicit common numerical control; may reduce performance. Not BI1.')
    parser.add_argument('--kv-cache-gib',type=float,default=42)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.9)
    parser.add_argument('--co-resident-with',help='Owned encoder lease for the diagnostic E/LLM co-residency test only')
    args=parser.parse_args()
    assert args.mm_cache_gb>=0 and args.renderer_workers>=1
    assert args.renderer_workers==1 or args.mm_cache_gb==0,'Official renderer threads require a disabled processor cache'
    assert args.ec_transport=='cpu' or args.observe
    if args.action=='start':
        with (ROOT/'logs'/('guardian_'+args.name+'.log')).open('ab') as log:
            command=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','serve',
                '--name',args.name,'--gpu',str(args.gpu),'--role',args.role,'--port',str(args.port),
                '--side-port',str(args.side_port),'--ec-path',str(args.ec_path)]
            if args.observe:command+=['--observe']
            if args.encoder_native_ops:command+=['--encoder-native-ops']
            if args.legacy_runner:command+=['--legacy-runner']
            if args.eager_reference:command+=['--eager-reference']
            if args.compiled_encoder:command+=['--compiled-encoder']
            if args.token_split:command+=['--token-split']
            if args.prefix_probe:command+=['--prefix-probe']
            if args.fixed_reduction:command+=['--fixed-reduction']
            if args.deterministic_reductions:command+=['--deterministic-reductions']
            command+=['--kv-cache-gib',str(args.kv_cache_gib),'--gpu-memory-utilization',str(args.gpu_memory_utilization)]
            if args.co_resident_with:command+=['--co-resident-with',args.co_resident_with]
            if args.encoder_whole_request:command+=['--encoder-whole-request']
            if args.encoder_sync_scheduling:command+=['--encoder-sync-scheduling']
            if args.api_switch_interval is not None:command+=['--api-switch-interval',str(args.api_switch_interval)]
            command+=['--mm-cache-gb',str(args.mm_cache_gb),'--renderer-workers',str(args.renderer_workers)]
            command+=['--ec-transport',args.ec_transport,'--encoder-api',args.encoder_api]
            command+=['--encoder-token-budget',str(args.encoder_token_budget)]
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(guardian=child.pid,name=args.name)))
    elif args.action=='serve':serve(args.name,args.gpu,args.role,args.port,args.ec_path,args.side_port,args.observe,args.encoder_native_ops,args.legacy_runner,args.eager_reference,args.compiled_encoder,args.encoder_token_budget,args.encoder_whole_request,args.encoder_sync_scheduling,args.api_switch_interval,args.mm_cache_gb,args.renderer_workers,args.ec_transport,args.encoder_api,args.token_split,args.kv_cache_gib,args.gpu_memory_utilization,args.co_resident_with,args.prefix_probe,args.fixed_reduction,args.deterministic_reductions)
    elif args.action=='stop':
        write(ROOT/'state'/('stop_server_'+args.name),dict(time=time.time()))
    else:print((ROOT/'state'/('server_'+args.name+'.json')).read_text())


if __name__=='__main__':main()
