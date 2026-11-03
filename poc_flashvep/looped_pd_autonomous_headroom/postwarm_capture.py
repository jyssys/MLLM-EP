"""Start native collection after graph warm-up and before measured arrivals."""
import os
import time
from .common import ROOT, append, write


def install(cfg):
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from ..looped_pd_atomic_prefill import rlt_worker
    previous_step = StageEngine._step
    previous_close = StageEngine.close
    previous_stats = StageEngine.stats
    previous_serve = rlt_worker.serve
    folder = ROOT/'raw'/cfg['name']
    request = folder/'postwarm_capture_requested.json'
    ready = folder/'postwarm_capture_ready.json'
    state = dict(started=False, stopped=False)

    def start_if_requested(engine):
        if engine.stage_config.role == cfg.get('capture_role', 'prefill') and not state['started'] and request.exists():
            begin = time.monotonic()
            torch.cuda.cudart().cudaProfilerStart()
            lower = time.monotonic()
            torch.cuda.nvtx.mark('RECOVERY_CAPTURE_CLOCK')
            upper = time.monotonic()
            state['started'] = True
            append(folder/'capture_events.jsonl', dict(time=(lower+upper)/2, anchor_lower=lower,
                anchor_upper=upper, profiler_start_call_begin=begin,
                profiler_start_call_seconds=lower-begin, pid=os.getpid(), event='START',
                device=str(engine.model_runner.device), reason='AFTER_GRAPH_WARMUP_BEFORE_MEASURED_ARRIVALS',
                marker='RECOVERY_CAPTURE', wall_ns=time.time_ns()))
            write(ready, dict(time=upper, pid=os.getpid()))

    def step(engine, *args, **kwargs):
        start_if_requested(engine)
        return previous_step(engine, *args, **kwargs)

    def stats(engine):
        # An idle native worker does not call step(). A native stats message
        # wakes its control loop without creating a fake model request.
        start_if_requested(engine)
        return previous_stats(engine)

    def close(engine):
        if state['started'] and not state['stopped']:
            torch.cuda.nvtx.mark('RECOVERY_CAPTURE_STOP_CLOCK')
            now = time.monotonic()
            state['stopped'] = True
            append(folder/'capture_events.jsonl', dict(time=now, stop_begin=now, pid=os.getpid(),
                event='METRIC_WINDOW_END_COLLECTION_CONTINUES_TO_APP_EXIT', role=engine.stage_config.role))
        return previous_close(engine)

    def serve(engine, inputs, args, run, warmup=False, contract=None):
        result = previous_serve(engine, inputs, args, run, warmup, contract)
        if warmup:
            write(request, dict(time=time.monotonic(), scope='All native warm-up work completed. Measured arrivals have not started.'))
            deadline = time.monotonic()+60.
            while not ready.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Post-warmup profiler handshake did not complete')
                for peer in engine.peers.values():
                    peer.client.stats()
                time.sleep(.01)
        return result

    StageEngine._step = step
    StageEngine.close = close
    StageEngine.stats = stats
    rlt_worker.serve = serve
