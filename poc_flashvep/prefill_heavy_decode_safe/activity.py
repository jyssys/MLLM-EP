"""Enable once before model construction. Export only after serving drains."""
import ctypes,json,os,subprocess,time
from .common import *


def build(source=None):
    initialize();source=Path(source) if source is not None else SOURCE/'preallocated_activity.cc'
    target=ROOT/'tools'/('activity_'+digest(source)[:16]+'.so')
    command=['g++','-O2','-std=c++17','-fPIC','-shared','-pthread',str(source),
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(target)]
    if not target.exists():subprocess.run(command,check=True)
    info=dict(command=command,source_SHA256=digest(source),
        library=str(target),library_SHA256=digest(target))
    write(ROOT/'tools/activity_build.json',info)
    return info


def library():
    path=ROOT/'tools/activity_build.json'
    if os.environ.get('ARC_CONFIG'):
        name=json.loads(Path(os.environ['ARC_CONFIG']).read_text())['name']
        frozen=ROOT/'raw'/name/'activity_runtime_build.json'
        if frozen.exists():path=frozen
    info=json.loads(path.read_text());assert digest(info['library'])==info['library_SHA256']
    h=ctypes.CDLL(info['library']);h.activity_begin.argtypes=[ctypes.c_size_t,ctypes.c_int]
    h.activity_prepare.argtypes=[ctypes.c_size_t];h.activity_enable.argtypes=[ctypes.c_int]
    h.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
    h.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    h.activity_options.argtypes=[ctypes.c_int,ctypes.c_size_t,ctypes.c_size_t]
    h.activity_kernel_mode.argtypes=[ctypes.c_int]
    h.activity_graph_mode.argtypes=[ctypes.c_int]
    for key in ('dropped','invalid','count','all_kernels','all_graphs','requested_buffers','completed_buffers','reserved_host_bytes'):
        getattr(h,'activity_'+key).restype=ctypes.c_uint64
    return h,info


def stamp(h):
    ns=ctypes.c_uint64();a=time.monotonic();assert h.activity_timestamp(ctypes.byref(ns))==0;b=time.monotonic()
    return dict(cupti_ns=ns.value,host_lower=a,host_upper=b,host_midpoint=(a+b)/2)


def install(cfg):
    if not cfg.get('discovery_activity'):return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    early,early_build=library()
    assert early.activity_kernel_mode(int(cfg.get('activity_serial_kernels',False)))==0
    options=None
    if cfg.get('activity_device_buffers') or cfg.get('activity_internal_host_pinned'):
        before_context=torch.cuda.is_initialized()
        assert not before_context,'Buffer placement must precede CUDA context creation'
        device_buffers=not cfg.get('activity_internal_host_pinned',False)
        status=early.activity_options(int(device_buffers),cfg.get('activity_internal_buffer_bytes',8*1024**2),
            cfg.get('activity_internal_preallocate',16))
        assert status==0,('CUPTI buffer configuration failed',status)
        options=dict(device_buffers=device_buffers,internal_host_pinned=not device_buffers,
            bytes_per_buffer=cfg.get('activity_internal_buffer_bytes',8*1024**2),
            preallocate=cfg.get('activity_internal_preallocate',16),before_torch_context=not before_context,status=status)
    from vllm_rlt.worker.model_runner import ModelRunner
    old_init,old_step,old_close=StageEngine.__init__,StageEngine._step,StageEngine.close
    old_execute=ModelRunner._execute_batch;old_submit=ModelRunner.submit
    folder=ROOT/'raw'/cfg['name'];state={}
    def prepare_role(role):
        if state:
            assert state['role']==role
            return
        before_initialized=torch.cuda.is_initialized()
        h,info=library();begin=time.monotonic()
        graph_mode=bool(cfg.get('activity_D_graph_enclosures') and role=='decode')
        assert h.activity_graph_mode(int(graph_mode))==0
        delayed=cfg.get('activity_delayed_start',False) or cfg.get('activity_quiescent_decode',False)
        if delayed:assert h.activity_prepare(cfg.get('activity_host_pool_bytes',4*1024**3))==0
        else:assert h.activity_begin(cfg.get('activity_host_pool_bytes',4*1024**3),int(cfg.get('activity_memcpy',False)))==0
        version=ctypes.c_uint();assert h.cuptiGetVersion(ctypes.byref(version))==0
        state.update(h=h,build=info,prepare_ms=1000*(time.monotonic()-begin),start=None,stop=None,
            role=role,prepared_before_torch_context=not before_initialized,next_poll=0.,origin=None,internal_buffer_options=options,
            delayed_actual_activation=delayed,enable_host_ms=None,disable_host_ms=None,
            graph_enclosures=graph_mode,quiet_epochs=[],quiet_collecting=False,
            CUPTI_API_version=version.value,
            kernel_activity_kind='SERIAL_KERNEL_TRACE' if cfg.get('activity_serial_kernels') else 'CONCURRENT_KERNEL_TRACE')
    if cfg.get('activity_before_model'):
        from vllm_rlt.engine.worker import StageWorker
        previous_worker=StageWorker.__init__
        def before_model(worker,stage_config,*args,**kwargs):
            if not (cfg.get('activity_D_disabled') and stage_config.role=='decode'):
                prepare_role(stage_config.role)
            return previous_worker(worker,stage_config,*args,**kwargs)
        StageWorker.__init__=before_model
    def close_epoch():
        if not state.get('quiet_collecting'):return
        stop=stamp(state['h']);begin=time.monotonic()
        assert state['h'].activity_disable()==0
        state['quiet_epochs'].append(dict(start=state.pop('quiet_start'),stop=stop,
            enable_host_ms=state.get('quiet_enable_host_ms'),
            disable_host_ms=1000*(time.monotonic()-begin)))
        state['quiet_collecting']=False
    def initialized(engine,*args,**kwargs):
        if cfg.get('activity_D_disabled') and kwargs['stage_config'].role=='decode':
            state.update(disabled=True,role='decode')
            return old_init(engine,*args,**kwargs)
        prepare_role(kwargs['stage_config'].role)
        result=old_init(engine,*args,**kwargs);state['role']=engine.stage_config.role
        state['model_cuda_device_index']=int(engine.model_runner.device.index)
        state['quiescent_only']=bool(cfg.get('activity_quiescent_decode') and state['role']=='decode')
        engine.model_runner._discovery_activity_engine=engine
        write(folder/f'activity_early_{state["role"]}.json',dict(time=time.monotonic(),prepare_ms=state['prepare_ms'],
            prepared_before_torch_context=state['prepared_before_torch_context'],build=state['build'],
            CUPTI_API_version=state['CUPTI_API_version'],LD_PRELOAD=os.environ.get('LD_PRELOAD'),
            CUPTI_mappings=[line.strip() for line in open('/proc/self/maps') if 'libcupti' in line],
            delayed_actual_activation=state['delayed_actual_activation'],
            scope='Buffers precede model construction. Delayed mode enables and disables collection during the declared window. Other modes remain active until close. Export follows all outputs.'))
        return result
    def stepped(engine,*args,**kwargs):
        if state.get('disabled'):return old_step(engine,*args,**kwargs)
        now=time.monotonic()
        if now>=state['next_poll']:
            state['next_poll']=now+.05
            marker=folder/'measurement_begin.json'
            if state['origin'] is None and marker.exists():
                state['origin']=json.loads(marker.read_text())['time']
                if cfg.get('activity_measurement_start_device_fence'):
                    before=stamp(state['h']);torch.cuda.synchronize(engine.model_runner.device);after=stamp(state['h'])
                    state['measurement_start_device_fence']=dict(begin=before,end=after,
                        scope='All prior device work completes before this fence returns. A launch racing with the fence remains unresolved. This is not an idle measurement.')
            if state['origin'] is not None:
                target=state['origin']+cfg.get('activity_window_fraction',.35)*cfg['arrival_offsets'][-1]
                if state['start'] is None and now>=target:
                    if state['delayed_actual_activation'] and not state['quiescent_only']:
                        begin=time.monotonic()
                        assert state['h'].activity_enable(int(cfg.get('activity_memcpy',False)))==0
                        state['enable_host_ms']=1000*(time.monotonic()-begin)
                    state['start']=stamp(state['h'])
                if state['start'] is not None and state['stop'] is None and now>=target+cfg.get('activity_window_seconds',5.):
                    state['stop']=stamp(state['h'])
                    if state['quiescent_only']:close_epoch()
                    elif state['delayed_actual_activation'] and not cfg.get('activity_disable_after_outputs'):
                        begin=time.monotonic();assert state['h'].activity_disable()==0
                        state['disable_host_ms']=1000*(time.monotonic()-begin)
        if state.get('quiescent_only') and state['start'] is not None and state['stop'] is None:
            from vllm_rlt.request import RequestPhase
            decode_phases=(RequestPhase.CODA,RequestPhase.PRELUDE,RequestPhase.RECURRENT)
            active=any(r.phase in decode_phases for r in engine.scheduler.requests.values())
            if active:close_epoch()
            elif not state['quiet_collecting']:
                begin=time.monotonic();assert state['h'].activity_enable(int(cfg.get('activity_memcpy',False)))==0
                state['quiet_enable_host_ms']=1000*(time.monotonic()-begin)
                state['quiet_start']=stamp(state['h']);state['quiet_collecting']=True
        return old_step(engine,*args,**kwargs)
    def executing(runner,*args,**kwargs):
        if state.get('quiescent_only'):close_epoch()
        return old_execute(runner,*args,**kwargs)
    def submitting(runner,*args,**kwargs):
        if state.get('quiescent_only') and cfg.get('activity_quiet_stop_before_submit'):close_epoch()
        return old_submit(runner,*args,**kwargs)
    def closed(engine):
        if state.get('disabled'):return old_close(engine)
        if state.get('start'):
            engine.model_runner.synchronize()
            device_sync_ms=None
            if cfg.get('activity_full_device_sync_after_outputs'):
                begin=time.monotonic();torch.cuda.synchronize(engine.model_runner.device)
                device_sync_ms=1000*(time.monotonic()-begin)
            if state['stop'] is None:state['stop']=stamp(state['h'])
            witness=None;export_stop=state['stop']['cupti_ns']
            if state.get('quiescent_only'):
                close_epoch();assert state['h'].activity_enable(0)==0
                begin_stamp=stamp(state['h'])
                tensor=torch.ones(1024,device=engine.model_runner.device)
                tensor.add_(1);torch.cuda.synchronize(engine.model_runner.device)
                end_stamp=stamp(state['h']);assert state['h'].activity_disable()==0
                witness=dict(start=begin_stamp,stop=end_stamp,
                    scope='Synthetic collector witness after all Serving outputs. Exclude from capacity and Serving statistics.')
                export_stop=end_stamp['cupti_ns']
            path=folder/f'physical_activity_{state["role"]}.csv';begin=time.monotonic()
            status=state['h'].activity_end(str(path).encode(),state['start']['cupti_ns'],export_stop)
            value={k:v for k,v in state.items() if k not in ('h',)}
            value.update(status=status,export_after_outputs_ms=1000*(time.monotonic()-begin),
                full_device_sync_after_outputs_ms=device_sync_ms,
                post_serving_collector_witness=witness,
                counters={key:getattr(state['h'],'activity_'+key)() for key in
                    ('dropped','invalid','count','all_kernels','all_graphs','requested_buffers','completed_buffers','reserved_host_bytes')},
                scope='Actual kernel timestamps, filtered after serving. The tracing mode remains explicit. All-stream union requires a matched interference gate. Serial tracing cannot establish natural concurrent execution. Not a Decode-safe capacity claim.')
            valid=value['counters']['invalid']==0
            if cfg.get('activity_causal_record_scope'):
                from .record_scope import audit
                value['window_record_scope']=audit(path,value);valid=value['window_record_scope']['passed']
            write(path.with_suffix('.json'),value)
            assert status==0 and value['counters']['dropped']==0 and valid
        return old_close(engine)
    StageEngine.__init__,StageEngine._step,StageEngine.close=initialized,stepped,closed
    ModelRunner._execute_batch=executing
    ModelRunner.submit=submitting


if __name__=='__main__':build()
