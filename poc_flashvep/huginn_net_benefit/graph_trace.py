"""Bounded graph-level trace. Graph enclosures are not kernel unions."""
import os
import time
from .common import ROOT, write


def install(cfg):
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    step = StageEngine._step
    initialize = StageEngine.__init__
    close = StageEngine.close
    folder = ROOT / 'raw' / cfg['name']
    state = dict(started=False, stopped=False, records=[])

    def start():
        low = time.monotonic()
        torch.cuda.cudart().cudaProfilerStart()
        now = time.monotonic()
        name = 'NET_BENEFIT_CLOCK_' + repr(now)
        torch.cuda.nvtx.mark(name)
        state.update(started=True, until=now+cfg.get('nsight_seconds', 3.))
        state['records'].append(dict(event='START', lower=low, upper=time.monotonic(),
            nvtx_name=name, monotonic=now, epoch_ns=time.time_ns()))

    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        if (cfg.get('nsight_start_before_measurement')
                and engine.stage_config.role == cfg.get('nsight_trigger_role', 'decode')):
            start()
            write(folder / f'graph_trace_clock_{os.getpid()}.json', state)
        return result

    def stop():
        low = time.monotonic()
        torch.cuda.cudart().cudaProfilerStop()
        state['stopped'] = True
        state['records'].append(dict(event='STOP', lower=low, upper=time.monotonic()))
        write(folder / f'graph_trace_clock_{os.getpid()}.json', state)

    def observed(engine, *args, **kwargs):
        if cfg.get('nsight_start_before_measurement'):
            return step(engine, *args, **kwargs)
        marker = folder / 'measurement_begin.json'
        if engine.stage_config.role == cfg.get('nsight_trigger_role', 'decode'):
            if not state['started'] and marker.exists():
                begin = __import__('json').loads(marker.read_text())['time']
                if time.monotonic() >= begin + cfg.get('nsight_delay_s', 10.):
                    start()
            elif state['started'] and not state['stopped'] and time.monotonic() >= state['until']:
                stop()
        return step(engine, *args, **kwargs)

    def closed(engine):
        if state['started'] and not state['stopped']:
            stop()
        return close(engine)
    StageEngine.__init__, StageEngine._step, StageEngine.close = initialized, observed, closed
