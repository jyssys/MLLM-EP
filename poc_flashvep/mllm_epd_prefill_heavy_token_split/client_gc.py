"""Optional bounded client startup-heap freeze with exact restoration."""
import gc
import time


class StaticHeapFreeze:
    def __init__(self):
        self.active = False

    def begin(self):
        assert gc.get_freeze_count() == 0, 'Do not unfreeze another owner\'s static heap'
        self.was_enabled = gc.isenabled()
        self.thresholds = gc.get_threshold()
        begin = time.monotonic_ns()
        collected = gc.collect()
        gc.freeze()
        self.active = True
        return dict(policy='startup_heap_freeze',begin_ns=begin,end_ns=time.monotonic_ns(),
            collected_before_arrivals=collected,frozen_objects=gc.get_freeze_count(),
            GC_enabled=gc.isenabled(),thresholds=self.thresholds,
            scope='Freeze existing client objects before planned arrivals. New allocations still use normal GC. '
                'No server, sampling, token or arrival timestamp change.')

    def close(self):
        if self.active:
            gc.unfreeze()
            self.active = False
            assert gc.get_freeze_count() == 0
            assert gc.isenabled() == self.was_enabled
            assert gc.get_threshold() == self.thresholds
