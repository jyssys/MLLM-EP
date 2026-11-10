"""Capture activity before measured serving and export after serving ends."""
import ctypes
import os
import time
import threading
from .common import ROOT, write


def library(legacy=False,override=None):
    if override:path=override
    elif legacy:
        import json
        manifest=json.loads((ROOT/'tools/activity_build.json').read_text())
        path=manifest.get('library',manifest.get('binary'))
        if path is None:
            from ..fluidtoken_loop_split.activity_selftest import library as qualified
            return qualified()
    else:path=ROOT/'tools/deferred_activity.so'
    lib=ctypes.CDLL(str(path))
    lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    lib.activity_end.argtypes=[ctypes.c_char_p]
    for key in ('dropped','invalid','count','allocated'):
        if hasattr(lib,'activity_'+key):getattr(lib,'activity_'+key).restype=ctypes.c_uint64
    return lib


def anchor(lib):
    value=ctypes.c_uint64();lower=time.monotonic_ns()
    assert lib.activity_timestamp(ctypes.byref(value))==0
    upper=time.monotonic_ns()
    return dict(cupti_ns=value.value,monotonic_lower_ns=lower,monotonic_upper_ns=upper)


def install(cfg):
    assert not cfg.get('kernel_activity_seconds') and not cfg.get('sparse_device_trace')
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    previous_step=StageEngine._step;previous_close=StageEngine.close
    folder=ROOT/'raw'/cfg['name'];state=dict(started=False)

    def step(engine,*args,**kwargs):
        roles=cfg.get('full_activity_roles',('prefill','decode'))
        if engine.stage_config.role in roles and not state['started'] and (folder/'measurement_begin.json').exists():
            lib=library(cfg.get('full_collector_legacy',False),cfg.get('activity_library'))
            version=ctypes.c_uint();assert lib.cuptiGetVersion(ctypes.byref(version))==0
            assert lib.activity_begin()==0
            state.update(started=True,lib=lib,begin=anchor(lib),role=engine.stage_config.role)
            write(folder/f"full_activity_{state['role']}_begin.json",dict(
                anchor=state['begin'],pid=os.getpid(),CUPTI_API_version=version.value,
                CUPTI_mappings=[line.strip() for line in open('/proc/self/maps') if 'libcupti' in line],
                scope='Capture starts before the first measured stage step. Warm-up graphs precede CUPTI activation. Analysis reports coverage.'))
            if cfg.get('activity_periodic_flush_seconds'):
                state['flush_stop']=threading.Event();state['flushes']=[]
                cupti=lib
                def drain():
                    while not state['flush_stop'].wait(cfg['activity_periodic_flush_seconds']):
                        before=time.monotonic();code=cupti.cuptiActivityFlushAll(0)
                        state['flushes'].append(dict(time=before,seconds=time.monotonic()-before,result=code))
                state['flush_thread']=threading.Thread(target=drain,daemon=True);state['flush_thread'].start()
        return previous_step(engine,*args,**kwargs)

    def close(engine):
        if state['started']:
            if 'flush_stop' in state:
                state['flush_stop'].set();state['flush_thread'].join()
            synchronization_error=None
            try:torch.cuda.synchronize()
            except BaseException as error:synchronization_error=repr(error)
            end=anchor(state['lib']);role=state['role']
            code=state['lib'].activity_end(str(folder/f'full_activity_{role}.csv').encode())
            write(folder/f'full_activity_{role}_end.json',dict(anchor=end,result=code,
                dropped=state['lib'].activity_dropped(),invalid=state['lib'].activity_invalid(),
                kernels=state['lib'].activity_count(),
                allocated_bytes=state['lib'].activity_allocated() if hasattr(state['lib'],'activity_allocated') else None,
                periodic_flushes=state.get('flushes',[]),synchronization_error=synchronization_error,
                scope='Export occurs after measured serving. CUDA synchronization occurs only during close.'))
            state['started']=False
            assert code==0,'Deferred activity collector failed'
        return previous_close(engine)

    StageEngine._step=step;StageEngine.close=close
