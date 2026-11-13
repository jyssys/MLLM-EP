"""Capture a bounded activity window without synchronizing the serving thread."""
import atexit,ctypes,threading
from .common import *


def install(cfg,role):
    if role not in cfg.get('activity_roles',('prefill','decode')):return
    import torch
    from vllm.v1.engine.llm_engine import LLMEngine
    from ..dense_ar_layerwise_pd.activity import stamp
    build=json.loads((ROOT/'tools/activity_build.json').read_text())
    assert digest(build['source'])==build['source_SHA256'] and digest(build['library'])==build['library_SHA256']
    assert not torch.cuda.is_initialized(),'Prepare activity before model construction'
    lib=ctypes.CDLL(build['library'])
    lib.activity_prepare.argtypes=[ctypes.c_size_t];lib.activity_enable.argtypes=[ctypes.c_int]
    lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    lib.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
    keys=('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes')
    for key in keys:getattr(lib,'activity_'+key).restype=ctypes.c_uint64
    assert lib.activity_prepare(cfg.get('activity_host_pool_bytes',256*1024**2))==0
    folder=ROOT/'raw'/cfg['name'];stop=threading.Event();warm=threading.Event();state=dict(closed=False)
    def background():
        active=False
        try:
            while not stop.is_set() and not warm.wait(.01):pass
            if stop.is_set():return
            assert lib.activity_enable(int(cfg.get('activity_memcpy',True)))==0;active=True
            stop.wait(.1);assert lib.activity_disable()==0;active=False
            while not stop.is_set() and not (folder/'measurement_begin.json').exists():stop.wait(.01)
            if stop.is_set():return
            measured=json.loads((folder/'measurement_begin.json').read_text())['time']
            stop.wait(max(0.,measured+cfg.get('activity_background_offset_s',5.)-time.monotonic()))
            if stop.is_set():return
            assert lib.activity_enable(int(cfg.get('activity_memcpy',True)))==0;active=True
            state['start']=stamp(lib)
            stop.wait(cfg.get('activity_background_seconds',.6));state['end']=stamp(lib)
            assert lib.activity_disable()==0;active=False
        except BaseException as error:state['error']=repr(error)
        finally:
            if active:lib.activity_disable()
    thread=threading.Thread(target=background,daemon=True,name='bounded-CUPTI-diagnostic');thread.start()
    step=LLMEngine.step
    def stepped(self,*a,**kw):
        if not warm.is_set() and self.has_unfinished_requests():warm.set()
        return step(self,*a,**kw)
    LLMEngine.step=stepped
    def close():
        if state['closed']:return
        state['closed']=True;stop.set();thread.join(3)
        assert not thread.is_alive() and 'error' not in state,state
        assert 'start' in state and 'end' in state,'No background capture window'
        # This synchronization follows all delivered outputs. The serving
        # thread performs no collector fence while measuring latency.
        torch.cuda.synchronize()
        result=lib.activity_end(str(folder/f'activity_{role}.csv').encode(),state['start']['cupti_ns'],state['end']['cupti_ns'])
        counters={key:getattr(lib,'activity_'+key)() for key in keys}
        write(folder/f'activity_{role}_background.json',dict(start=state['start'],end=state['end'],
            result=result,counters=counters,build=build,pid=os.getpid(),edge_trim_s=.1,
            scope='Actual bounded CUDA activity. No serving-thread collector fence. Require coverage, common-window, A/A, and capture-interference gates before any clean claim.'))
        assert result==0 and counters['dropped']==0
    initialize=LLMEngine.__init__
    def initialized(self,*a,**kw):
        initialize(self,*a,**kw);shutdown=self.engine_core.shutdown
        def closed(*args,**kwargs):close();return shutdown(*args,**kwargs)
        self.engine_core.shutdown=closed
    LLMEngine.__init__=initialized;atexit.register(close)
