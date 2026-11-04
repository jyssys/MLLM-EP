"""Record infrequent Python collection intervals without changing GC policy."""
import atexit
import gc
import os
import threading
import time
from .common import ROOT, write


def install(cfg):
    if not cfg.get('observe_python_gc'):
        return
    events, starts = [], {}
    closed = [False]
    initial = dict(enabled=gc.isenabled(), thresholds=gc.get_threshold(), counts=gc.get_count())

    def callback(phase, info):
        now = time.monotonic()
        generation = info['generation']
        if phase == 'start':
            starts[generation] = now
        else:
            events.append(dict(start=starts.pop(generation, now), end=now,
                generation=generation, collected=info['collected'],
                uncollectable=info['uncollectable'], thread_id=threading.get_ident()))

    def flush():
        if closed[0]:
            return
        closed[0] = True
        gc.callbacks.remove(callback)
        write(ROOT / 'raw' / cfg['name'] / f'python_gc_{os.getpid()}.json', dict(
            pid=os.getpid(), initial=initial, events=events, final_counts=gc.get_count(),
            policy_changed=False,
            scope='Measured host GC intervals. Temporal overlap with ITL is not proof of causation. No model or scheduling operation is removed.'))

    gc.callbacks.append(callback)
    atexit.register(flush)
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.pd.engine import PDEngine
    for cls in (StageEngine, PDEngine):
        original = cls.close
        def close(engine, *args, _original=original, **kwargs):
            try:
                return _original(engine, *args, **kwargs)
            finally:
                flush()
        cls.close = close
