"""Reuse the fixed-buffer activity collector without API callback tracing."""
import atexit,ctypes,json,os,subprocess,time
from pathlib import Path
from .common import ROOT,REPO,digest,write


def build():
    source=REPO/'poc_flashvep/prefill_heavy_decode_safe/preallocated_activity.cc'
    tools=ROOT/'tools';tools.mkdir(exist_ok=True)
    library=tools/('activity_'+digest(source)[:16]+'.so')
    command=['g++','-O2','-std=c++17','-fPIC','-shared','-pthread',str(source),
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(library)]
    if not library.exists():subprocess.run(command,check=True)
    value=dict(source=str(source),source_SHA256=digest(source),library=str(library),library_SHA256=digest(library),command=command)
    write(tools/'activity_build.json',value);return value


def stamp(lib):
    ns=ctypes.c_uint64();a=time.monotonic();assert lib.activity_timestamp(ctypes.byref(ns))==0;b=time.monotonic()
    return dict(cupti_ns=ns.value,host_lower=a,host_upper=b,host_midpoint=(a+b)/2)


def install(cfg,role):
    if role not in cfg.get('activity_roles',('prefill','decode')):return
    import torch
    from vllm.v1.engine.llm_engine import LLMEngine
    value=json.loads((ROOT/'tools/activity_build.json').read_text())
    assert digest(value['library'])==value['library_SHA256']
    assert digest(value['source'])==value['source_SHA256']
    lib=ctypes.CDLL(value['library'])
    lib.activity_begin.argtypes=[ctypes.c_size_t,ctypes.c_int]
    lib.activity_prepare.argtypes=[ctypes.c_size_t]
    lib.activity_enable.argtypes=[ctypes.c_int]
    lib.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
    lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    for key in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes'):
        getattr(lib,'activity_'+key).restype=ctypes.c_uint64
    assert not torch.cuda.is_initialized(),'Activate before CUDA context and model construction'
    quiet=role=='decode' and cfg.get('activity_quiet_destination',False)
    if quiet:assert lib.activity_prepare(cfg.get('activity_host_pool_bytes',1024**3))==0
    else:assert lib.activity_begin(cfg.get('activity_host_pool_bytes',1024**3),int(cfg.get('activity_memcpy',False)))==0
    folder=ROOT/'raw'/cfg['name'];started=stamp(lib);state={'closed':False}
    state.update(collecting=False,epochs=[])
    write(folder/f'activity_{role}_begin.json',dict(anchor=started,build=value,pid=os.getpid(),
        quiet_destination=quiet,
        scope='Buffers precede model construction. Full mode remains active. Quiet-D mode records only explicitly fenced empty-engine epochs and stops before the next model preparation. Export follows all outputs.'))
    if quiet:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        original_step=LLMEngine.step;original_inputs=GPUModelRunner.prepare_inputs
        def close_epoch():
            if not state['collecting']:return
            end=stamp(lib);assert lib.activity_disable()==0
            state['epochs'].append(dict(start=state['epoch_start'],end=end))
            state['collecting']=False
        def prepared(self,*a,**kw):
            close_epoch();return original_inputs(self,*a,**kw)
        def stepped(self,*a,**kw):
            if (folder/'measurement_begin.json').exists() and not self.has_unfinished_requests() and not state['collecting']:
                torch.cuda.synchronize()
                assert lib.activity_enable(int(cfg.get('activity_memcpy',False)))==0
                state['epoch_start']=stamp(lib);state['collecting']=True
            return original_step(self,*a,**kw)
        GPUModelRunner.prepare_inputs=prepared;LLMEngine.step=stepped
    def close():
        if state['closed']:return
        state['closed']=True
        torch.cuda.synchronize()
        witness=None
        if quiet:
            close_epoch();assert lib.activity_enable(0)==0;first=stamp(lib)
            data=torch.ones(1024,device='cuda');data.add_(1);torch.cuda.synchronize()
            last=stamp(lib);assert lib.activity_disable()==0
            witness=dict(start=first,end=last,scope='Collector witness after all Serving outputs. Exclude it from capacity.')
        ended=stamp(lib)
        code=lib.activity_end(str(folder/f'activity_{role}.csv').encode(),started['cupti_ns'],ended['cupti_ns'])
        counters={key:getattr(lib,'activity_'+key)() for key in
            ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes')}
        write(folder/f'activity_{role}_end.json',dict(anchor=ended,result=code,counters=counters,
            quiet_destination=quiet,capture_epochs=state['epochs'],post_serving_witness=witness,
            scope='Actual concurrent CUDA kernel timestamps. Union accounting and matched interference qualification are separate analysis gates.'))
        assert code==0 and counters['dropped']==0
    original=LLMEngine.__init__
    def initialized(self,*a,**kw):
        original(self,*a,**kw)
        shutdown=self.engine_core.shutdown
        def closed(*args,**kwargs):
            close();return shutdown(*args,**kwargs)
        self.engine_core.shutdown=closed
    LLMEngine.__init__=initialized
    atexit.register(close)


if __name__=='__main__':print(build())
