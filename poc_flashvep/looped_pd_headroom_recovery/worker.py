"""Use the qualified runner without changing tensor or transfer semantics."""
import importlib
import os
from .common import configure


def main():
    old = configure()
    cfg = old.config()
    if cfg['worker'] == 'native_mixed_worker':
        from ..looped_pd_atomic_prefill import native_mixed_worker
        native_mixed_worker.install()
    else:
        from .runtime import install
        install()
    module = importlib.import_module('poc_flashvep.looped_pd_atomic_prefill.' + cfg['worker'])
    module.ROOT = old.ROOT
    from ..looped_pd_atomic_prefill import rlt_worker
    original=rlt_worker.serve
    def measured(engine,inputs,args,run,warmup=False,contract=None):
        if not warmup:
            from .common import write,time
            write(old.ROOT/'raw'/cfg['name']/'measurement_begin.json',dict(time=time.monotonic(),run=run))
        return original(engine,inputs,args,run,warmup,contract)
    rlt_worker.serve=measured
    module.main()


if __name__ == '__main__':
    main()
