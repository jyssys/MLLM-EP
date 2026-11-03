"""Collect one bounded GPU window during the measured arrival interval."""
import os
import time
import json
from .common import ROOT, append


def install(cfg):
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    previous_step = StageEngine._step
    previous_close = StageEngine.close
    folder = ROOT/'raw'/cfg['name']
    state = dict(started=False, stopped=False, origin=None, until=None)

    def stop(engine, reason):
        if not state['started'] or state['stopped']:
            return
        torch.cuda.nvtx.mark('RECOVERY_CAPTURE_STOP_CLOCK')
        before = time.monotonic()
        torch.cuda.cudart().cudaProfilerStop()
        state['stopped'] = True
        append(folder/'capture_events.jsonl', dict(time=time.monotonic(), stop_begin=before,
            pid=os.getpid(), event=reason, role=engine.stage_config.role))

    def step(engine, *args, **kwargs):
        if engine.stage_config.role == cfg.get('capture_role', 'prefill'):
            marker = folder/'measurement_begin.json'
            if state['origin'] is None and marker.exists():
                state['origin'] = json.loads(marker.read_text())['time']
            now = time.monotonic()
            delay = cfg.get('capture_arrival_fraction', .35)*cfg['arrival_offsets'][-1]
            if state['origin'] is not None and not state['started'] and now >= state['origin']+delay:
                before = time.monotonic()
                torch.cuda.cudart().cudaProfilerStart()
                lower = time.monotonic()
                torch.cuda.nvtx.mark('RECOVERY_CAPTURE_CLOCK')
                upper = time.monotonic()
                state.update(started=True, until=upper+cfg['autonomous_capture_seconds'])
                append(folder/'capture_events.jsonl', dict(time=(lower+upper)/2,
                    anchor_lower=lower, anchor_upper=upper, profiler_start_call_begin=before,
                    profiler_start_call_seconds=lower-before, pid=os.getpid(), event='START',
                    device=str(engine.model_runner.device), reason='BOUNDED_MEASURED_ARRIVAL_WINDOW',
                    marker='RECOVERY_CAPTURE', wall_ns=time.time_ns()))
            if state['started'] and not state['stopped'] and time.monotonic() >= state['until']:
                stop(engine, 'STOP')
        return previous_step(engine, *args, **kwargs)

    def close(engine):
        stop(engine, 'STOP_CLOSE')
        return previous_close(engine)

    StageEngine._step = step
    StageEngine.close = close
