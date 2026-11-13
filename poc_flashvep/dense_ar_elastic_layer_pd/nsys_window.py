"""Bound an external Nsight capture after warm-up without collector fences."""
import atexit
from .common import *


def command(cfg,role,worker_command):
    if not cfg.get('nsys_bounded_capture') or role not in cfg.get('nsys_roles',('prefill','decode')):
        return worker_command
    output=ROOT/'raw'/cfg['name']/('nsys_'+role)
    assert not output.with_suffix('.nsys-rep').exists()
    return ['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
        '--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-graph-trace=node',
        '--cuda-memory-usage=false','--force-overwrite=false','--output='+str(output),*worker_command]


def install(cfg,role):
    if not cfg.get('nsys_bounded_capture') or role not in cfg.get('nsys_roles',('prefill','decode')):return
    import torch
    from vllm.v1.engine.llm_engine import LLMEngine
    state=dict(start=None,end=None,next_check=0.,measured=None)
    folder=ROOT/'raw'/cfg['name'];step=LLMEngine.step
    def marker(label):
        before=time.monotonic();torch.cuda.nvtx.range_push(f'ELASTIC_MONO_{label}_{before:.9f}')
        after=time.monotonic();torch.cuda.nvtx.range_pop()
        return dict(before=before,after=after)
    def stop():
        if state['start'] is not None and state['end'] is None:
            state['end']=marker('END');torch.cuda.cudart().cudaProfilerStop()
    def observed(self,*a,**kw):
        now=time.monotonic()
        if state['end'] is None and now>=state['next_check']:
            state['next_check']=now+.02
            if state['measured'] is None and (folder/'measurement_begin.json').exists():
                state['measured']=json.loads((folder/'measurement_begin.json').read_text())['time']
            if state['measured'] is not None:
                if state['start'] is None and now>=state['measured']+cfg.get('nsys_offset_s',5.):
                    torch.cuda.cudart().cudaProfilerStart();state['start']=marker('START')
                elif state['start'] is not None and now>=state['start']['after']+cfg.get('nsys_seconds',2.):stop()
        return step(self,*a,**kw)
    LLMEngine.step=observed
    def close():
        stop();write(folder/f'nsys_clock_{role}.json',dict(state,
            scope='Host anchors for external CUDA activity. No serving-thread CUDA synchronization. Interference qualification is separate.'))
    atexit.register(close)
