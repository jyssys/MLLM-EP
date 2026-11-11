"""Enable once before model construction. Export only after serving drains."""
import ctypes,json,os,subprocess,time
from .common import *


def build():
    initialize();source=SOURCE/'preallocated_activity.cc';target=ROOT/'tools'/('activity_'+digest(source)[:16]+'.so')
    command=['g++','-O2','-std=c++17','-fPIC','-shared','-pthread',str(source),
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(target)]
    if not target.exists():subprocess.run(command,check=True)
    write(ROOT/'tools/activity_build.json',dict(command=command,source_SHA256=digest(source),
        library=str(target),library_SHA256=digest(target)))


def library():
    info=json.loads((ROOT/'tools/activity_build.json').read_text());assert digest(info['library'])==info['library_SHA256']
    h=ctypes.CDLL(info['library']);h.activity_begin.argtypes=[ctypes.c_size_t,ctypes.c_int]
    h.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
    h.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    h.activity_options.argtypes=[ctypes.c_int,ctypes.c_size_t,ctypes.c_size_t]
    for key in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes'):
        getattr(h,'activity_'+key).restype=ctypes.c_uint64
    return h,info


def stamp(h):
    ns=ctypes.c_uint64();a=time.monotonic();assert h.activity_timestamp(ctypes.byref(ns))==0;b=time.monotonic()
    return dict(cupti_ns=ns.value,host_lower=a,host_upper=b,host_midpoint=(a+b)/2)


def install(cfg):
    if not cfg.get('discovery_activity'):return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    options=None
    if cfg.get('activity_device_buffers'):
        early,info=library();before_context=torch.cuda.is_initialized()
        assert not before_context,'Buffer placement must precede CUDA context creation'
        status=early.activity_options(1,cfg.get('activity_internal_buffer_bytes',8*1024**2),
            cfg.get('activity_internal_preallocate',16))
        assert status==0,('CUPTI buffer configuration failed',status)
        options=dict(device_buffers=True,bytes_per_buffer=cfg.get('activity_internal_buffer_bytes',8*1024**2),
            preallocate=cfg.get('activity_internal_preallocate',16),before_torch_context=not before_context,status=status)
    old_init,old_step,old_close=StageEngine.__init__,StageEngine._step,StageEngine.close
    folder=ROOT/'raw'/cfg['name'];state={}
    def initialized(engine,*args,**kwargs):
        before_initialized=torch.cuda.is_initialized()
        h,info=library();begin=time.monotonic()
        assert h.activity_begin(cfg.get('activity_host_pool_bytes',4*1024**3),int(cfg.get('activity_memcpy',False)))==0
        state.update(h=h,build=info,prepare_ms=1000*(time.monotonic()-begin),start=None,stop=None,
            prepared_before_torch_context=not before_initialized,next_poll=0.,origin=None,internal_buffer_options=options)
        version=ctypes.c_uint();assert h.cuptiGetVersion(ctypes.byref(version))==0
        state['CUPTI_API_version']=version.value
        result=old_init(engine,*args,**kwargs);state['role']=engine.stage_config.role
        write(folder/f'activity_early_{state["role"]}.json',dict(time=time.monotonic(),prepare_ms=state['prepare_ms'],
            prepared_before_torch_context=not before_initialized,build=info,
            CUPTI_API_version=version.value,LD_PRELOAD=os.environ.get('LD_PRELOAD'),
            CUPTI_mappings=[line.strip() for line in open('/proc/self/maps') if 'libcupti' in line],
            scope='Collection starts before this StageEngine constructs its model and graphs. No mid-serving enable/disable or export.'))
        return result
    def stepped(engine,*args,**kwargs):
        now=time.monotonic()
        if now>=state['next_poll']:
            state['next_poll']=now+.05
            marker=folder/'measurement_begin.json'
            if state['origin'] is None and marker.exists():state['origin']=json.loads(marker.read_text())['time']
            if state['origin'] is not None:
                target=state['origin']+cfg.get('activity_window_fraction',.35)*cfg['arrival_offsets'][-1]
                if state['start'] is None and now>=target:state['start']=stamp(state['h'])
                if state['start'] is not None and state['stop'] is None and now>=target+cfg.get('activity_window_seconds',5.):state['stop']=stamp(state['h'])
        return old_step(engine,*args,**kwargs)
    def closed(engine):
        if state.get('start'):
            engine.model_runner.synchronize()
            if state['stop'] is None:state['stop']=stamp(state['h'])
            path=folder/f'physical_activity_{state["role"]}.csv';begin=time.monotonic()
            status=state['h'].activity_end(str(path).encode(),state['start']['cupti_ns'],state['stop']['cupti_ns'])
            value={k:v for k,v in state.items() if k not in ('h',)}
            value.update(status=status,export_after_outputs_ms=1000*(time.monotonic()-begin),
                counters={key:getattr(state['h'],'activity_'+key)() for key in
                    ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes')},
                scope='Actual concurrent-kernel timestamps, filtered after serving. All-stream union requires analysis and a matched interference gate. Not a Decode-safe capacity claim.')
            write(path.with_suffix('.json'),value)
            assert status==0 and value['counters']['dropped']==value['counters']['invalid']==0
        return old_close(engine)
    StageEngine.__init__,StageEngine._step,StageEngine.close=initialized,stepped,closed


if __name__=='__main__':build()
