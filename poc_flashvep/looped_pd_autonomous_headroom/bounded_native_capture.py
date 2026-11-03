"""Use native profiler start/stop, without the failing activity-toggle path."""
import os
import time
import json
from .common import ROOT, append, write


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.profiling import ProfileConfig
    from ..looped_pd_headroom_recovery.profile_trace_export import install as export
    from ..looped_pd_headroom_recovery.cuda_only_profile import clock
    export()
    previous_step, previous_close = StageEngine._step, StageEngine.close
    folder = ROOT/'raw'/cfg['name']
    selected_role = cfg.get('capture_role', 'prefill')
    state = dict(started=False, stopped=False, origin=None, until=None)

    def stop(engine):
        if not state['started'] or state['stopped']:
            return
        before = time.monotonic()
        engine.stop_profile()
        state['stopped'] = True
        append(folder/f'cuda_only_events_{selected_role}.jsonl', dict(event='COLLECTION_DISABLED',
            time=before, call_seconds=time.monotonic()-before, **clock()))

    def step(engine, *args, **kwargs):
        if engine.stage_config.role == selected_role:
            marker = folder/'measurement_begin.json'
            if state['origin'] is None and marker.exists():
                state['origin'] = json.loads(marker.read_text())['time']
            delay = cfg.get('capture_arrival_fraction', .35)*cfg['arrival_offsets'][-1]
            if state['origin'] is not None and not state['started'] and time.monotonic() >= state['origin']+delay:
                before = time.monotonic()
                engine.start_profile(ProfileConfig(output_dir=str(folder/'cuda_only_profile'/selected_role),
                    activities=('cuda',), record_shapes=False, with_stack=False,
                    profile_memory=False, with_flops=False), role=selected_role)
                anchor = clock()
                engine.profiling.current.profiler.add_metadata_json('recovery_clock', json.dumps(anchor))
                write(folder/f'cuda_only_clock_{selected_role}.json', dict(**anchor, pid=os.getpid(),
                    role=selected_role, logical_device=engine.model_runner.device.index,
                    evidence='MEASURED_HOST_CLOCK_PAIR_FOR_NATIVE_KINETO',
                    scope='Native collector started after warm-up. No dynamic activity toggle.'))
                state.update(started=True, until=time.monotonic()+cfg['bounded_native_capture_seconds'])
                append(folder/f'cuda_only_events_{selected_role}.jsonl', dict(event='COLLECTION_ENABLED',
                    time=time.monotonic(), call_seconds=time.monotonic()-before, **clock()))
            if (state['started'] and not state['stopped']
                    and not cfg.get('native_capture_stop_at_close', False)
                    and time.monotonic() >= state['until']):
                stop(engine)
        return previous_step(engine, *args, **kwargs)

    def close(engine):
        if engine.stage_config.role == selected_role and state['started']:
            stop(engine)
            append(folder/f'cuda_only_events_{selected_role}.jsonl', dict(event='METRIC_WINDOW_END', **clock()))
            status = engine.wait_for_profile_artifacts(cfg.get('profile_finalization_timeout', 300))
            write(folder/f'cuda_only_status_{selected_role}.json', status)
        return previous_close(engine)

    StageEngine._step = step
    StageEngine.close = close
