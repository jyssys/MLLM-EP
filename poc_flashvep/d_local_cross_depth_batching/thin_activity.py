"""Prime before traffic; pause without a flush; export after GPU owners drain."""
import ctypes
import json
import os
import time
from .common import ROOT, SOURCE, digest, write


def build():
    import subprocess
    source=SOURCE/'thin_activity.cc';tools=ROOT/'tools';tools.mkdir(exist_ok=True)
    library=tools/('thin_activity_'+digest(source)[:16]+'.so')
    command=['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',str(source),
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(library)]
    if not library.exists():
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        write(tools/'thin_build_log.json',dict(stdout=result.stdout,stderr=result.stderr))
    write(tools/'thin_activity_build.json',dict(command=command,source_SHA256=digest(source),
        library=str(library),library_SHA256=digest(library)))


def library():
    build=json.loads((ROOT/'tools/thin_activity_build.json').read_text())
    assert digest(build['library'])==build['library_SHA256']
    handle=ctypes.CDLL(build['library'])
    handle.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    handle.activity_end.argtypes=[ctypes.c_char_p]
    for name in ('activity_dropped','activity_invalid','activity_count'):
        getattr(handle,name).restype=ctypes.c_uint64
    return handle


def clock(handle):
    timestamp=ctypes.c_uint64();lower=time.monotonic()
    assert handle.activity_timestamp(ctypes.byref(timestamp))==0
    upper=time.monotonic()
    return dict(device_clock_ns=timestamp.value,monotonic_s=(lower+upper)/2,host_uncertainty_s=upper-lower)


def prime(handle,device):
    import torch
    assert handle.activity_begin()==0
    value=torch.ones(1,device=device);value.add_(1);torch.cuda.synchronize(device)
    assert handle.activity_pause()==0 and handle.activity_reset()==0


def install(cfg):
    if not cfg.get('dlocal_activity_thin'):
        return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    initialize,step,closed=StageEngine.__init__,StageEngine._step,StageEngine.close
    folder=ROOT/'raw'/cfg['name']
    state=dict(handle=None,start=None,stop=None,origin=None)
    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        if engine.stage_config.role=='decode':
            state['handle']=library();prime(state['handle'],engine.model_runner.device)
            write(folder/'thin_collector_primed.json',dict(time=time.monotonic(),pid=os.getpid(),
                scope='Collector activation, one diagnostic add, synchronization and buffer reset before worker-ready. No model work is changed.'))
        return result
    def stepped(engine,*args,**kwargs):
        if engine.stage_config.role=='decode' and state['handle'] is not None:
            marker=folder/'measurement_begin.json'
            if state['origin'] is None and marker.exists():
                state['origin']=json.loads(marker.read_text())['time']
            if state['origin'] is not None and state['start'] is None and time.monotonic()>=state['origin']+.35*cfg['arrival_offsets'][-1]:
                begin=time.monotonic();assert state['handle'].activity_begin()==0
                state['start_call_seconds']=time.monotonic()-begin;state['start']=clock(state['handle'])
            if state['start'] is not None and state['stop'] is None and time.monotonic()>=state['start']['monotonic_s']+cfg.get('kernel_activity_seconds',1.):
                state['stop']=clock(state['handle']);begin=time.monotonic()
                assert state['handle'].activity_pause()==0
                state['pause_call_seconds']=time.monotonic()-begin
        return step(engine,*args,**kwargs)
    def close(engine):
        if engine.stage_config.role=='decode' and state['start'] is not None:
            if state['stop'] is None:
                state['stop']=clock(state['handle']);assert state['handle'].activity_pause()==0
            engine.model_runner.synchronize()
            begin=time.monotonic()
            status=state['handle'].activity_end(str(folder/'kernel_activity_decode.csv').encode())
            value=dict(**{k:v for k,v in state.items() if k!='handle'},status=status,
                dropped=state['handle'].activity_dropped(),invalid=state['handle'].activity_invalid(),
                records=state['handle'].activity_count(),flush_after_serving_seconds=time.monotonic()-begin,
                role='decode',pid=os.getpid(),scope='Activity-only timestamps. Collection pauses without GPU synchronization or file export during serving. Final flush occurs after native worker drain.')
            write(folder/'activity_capture_decode.json',value)
            assert status==0 and value['dropped']==value['invalid']==0
        return closed(engine)
    StageEngine.__init__,StageEngine._step,StageEngine.close=initialized,stepped,close


if __name__=='__main__':
    build()
