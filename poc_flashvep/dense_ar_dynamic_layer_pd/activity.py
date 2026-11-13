"""Record a bounded activity window after the real Serving warm-up."""
import atexit,ctypes,json,os,time
from .common import ROOT,digest,write


def build():
    from ..dense_ar_layerwise_pd import activity
    activity.ROOT=ROOT
    return activity.build()


def install(cfg,role):
    if cfg.get('activity_sparse_seconds'):
        return sparse(cfg,role)
    from ..dense_ar_layerwise_pd import activity
    activity.ROOT=ROOT
    return activity.install(cfg,role)


def sparse(cfg,role):
    if role not in cfg.get('activity_roles',('prefill','decode')):return
    import torch
    from vllm.v1.engine.llm_engine import LLMEngine
    from ..dense_ar_layerwise_pd.activity import stamp
    value=json.loads((ROOT/'tools/activity_build.json').read_text())
    assert digest(value['source'])==value['source_SHA256'] and digest(value['library'])==value['library_SHA256']
    lib=ctypes.CDLL(value['library'])
    lib.activity_prepare.argtypes=[ctypes.c_size_t]
    lib.activity_enable.argtypes=[ctypes.c_int]
    lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    lib.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
    for key in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes'):
        getattr(lib,'activity_'+key).restype=ctypes.c_uint64
    assert not torch.cuda.is_initialized()
    assert lib.activity_prepare(cfg.get('activity_host_pool_bytes',256*1024**2))==0
    folder=ROOT/'raw'/cfg['name'];state=dict(closed=False,active=False,done=False,next_check=0.,measurement=None)
    def stop_window():
        if not state['active']:return
        torch.cuda.synchronize();state['end']=stamp(lib)
        assert lib.activity_disable()==0
        state.update(active=False,done=True)
    original_step=LLMEngine.step
    def step(self,*a,**kw):
        now=time.monotonic()
        if not state['done'] and now>=state['next_check']:
            state['next_check']=now+.05
            if state['measurement'] is None and (folder/'measurement_begin.json').exists():
                state['measurement']=json.loads((folder/'measurement_begin.json').read_text())['time']
            if state['measurement'] is not None:
                if not state['active'] and now>=state['measurement']+cfg.get('activity_sparse_offset_s',4.):
                    torch.cuda.synchronize()
                    assert lib.activity_enable(int(cfg.get('activity_memcpy',True)))==0
                    state['start']=stamp(lib);state['active']=True
                elif state['active'] and now>=state['start']['host_midpoint']+cfg['activity_sparse_seconds']:
                    stop_window()
        return original_step(self,*a,**kw)
    LLMEngine.step=step
    def close():
        if state['closed']:return
        state['closed']=True;stop_window()
        assert 'start' in state and 'end' in state,'The diagnostic window did not start'
        result=lib.activity_end(str(folder/f'activity_{role}.csv').encode(),state['start']['cupti_ns'],state['end']['cupti_ns'])
        counters={key:getattr(lib,'activity_'+key)() for key in
            ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes')}
        write(folder/f'activity_{role}_sparse.json',dict(start=state['start'],end=state['end'],result=result,
            counters=counters,build=value,pid=os.getpid(),
            scope='Actual CUDA activity in a bounded diagnostic capture. Serving interference, record coverage, and common-window union must pass separate gates. No Clean Idle claim follows from this file alone.'))
        assert result==0 and counters['dropped']==0
    original_init=LLMEngine.__init__
    def initialized(self,*a,**kw):
        original_init(self,*a,**kw);shutdown=self.engine_core.shutdown
        def closed(*args,**kwargs):close();return shutdown(*args,**kwargs)
        self.engine_core.shutdown=closed
    LLMEngine.__init__=initialized;atexit.register(close)


if __name__=='__main__':print(build())
