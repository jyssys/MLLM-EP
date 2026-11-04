"""Bound an activity-only capture on one native stage, without Nsight."""
import os
import time
from .common import *


def install(cfg):
    seconds = cfg.get('kernel_activity_seconds', 0)
    if not seconds:
        return
    if seconds > 8 and not cfg.get('allow_long_activity_diagnostic', False):
        raise RuntimeError('Long activity capture is blocked after an unexplained CUDA launch failure. Explicit diagnostic authorization is required.')
    from .activity_selftest import library, clock
    if cfg.get('nsys') or cfg.get('P_only_profiler') or cfg.get('autonomous_capture_seconds'):
        raise ValueError('Two collectors must not attach to one worker')
    from vllm_rlt.engine.stage_engine import StageEngine
    step, close = StageEngine._step, StageEngine.close
    folder = ROOT / 'raw' / cfg['name']
    state = dict(handle=None, start=None, stop=None)
    roles = cfg.get('capture_roles', [cfg.get('capture_role', 'prefill')])

    def suffix(engine):
        return '_decode' if engine.stage_config.role == 'decode' else ''

    def stop(engine):
        if state['handle'] is None or state['stop'] is not None:
            return
        handle = state['handle']
        anchor = clock(handle)
        state['stop'] = anchor
        begin = time.monotonic()
        # Finish kernels that intersect the recorded boundary before flushing.
        # The parser clips their timestamps to the earlier stop anchor.
        # This synchronization belongs to measured profiler overhead.
        import torch
        torch.cuda.synchronize(engine.model_runner.device)
        status = handle.activity_end(str(folder / ('kernel_activity' + suffix(engine) + '.csv')).encode())
        write(folder / ('activity_capture' + suffix(engine) + '.json'), dict(start=state['start'], stop=anchor,
            pid=os.getpid(), role=engine.stage_config.role,
            start_call_seconds=state['start_call_seconds'],
            stop_call_seconds=time.monotonic() - begin,
            status=status, dropped=handle.activity_dropped(), invalid=handle.activity_invalid(),
            records=handle.activity_count(), source='ACTIVITY_ONLY_CUPTI_CONCURRENT_KERNEL',
            collector_build=json.loads((ROOT / 'tools/activity_build.json').read_text()),
            scope='GPU kernel timestamps. No SM occupancy or memory-bandwidth counter.'))
        if status or handle.activity_dropped() or handle.activity_invalid():
            raise RuntimeError('Incomplete GPU activity evidence')

    def stepped(engine, *args, **kwargs):
        if engine.stage_config.role in roles:
            marker = folder / 'measurement_begin.json'
            if state['handle'] is None and marker.exists():
                origin = json.loads(marker.read_text())['time']
                due = origin + cfg.get('capture_arrival_fraction', .35) * cfg['arrival_offsets'][-1]
                if time.monotonic() >= due:
                    handle = library()
                    begin = time.monotonic()
                    status = handle.activity_begin()
                    if status:
                        raise RuntimeError(f'CUPTI activity start failed: {status}')
                    state['start_call_seconds'] = time.monotonic() - begin
                    state['handle'], state['start'] = handle, clock(handle)
                    write(folder / ('activity_start' + suffix(engine) + '.json'), dict(start=state['start'], pid=os.getpid(),
                        role=engine.stage_config.role, start_call_seconds=state['start_call_seconds']))
            if (state['handle'] is not None and state['stop'] is None
                    and time.monotonic() >= state['start']['monotonic_s'] + seconds):
                stop(engine)
        return step(engine, *args, **kwargs)

    def closed(engine):
        if engine.stage_config.role in roles:
            stop(engine)
        return close(engine)
    StageEngine._step, StageEngine.close = stepped, closed
