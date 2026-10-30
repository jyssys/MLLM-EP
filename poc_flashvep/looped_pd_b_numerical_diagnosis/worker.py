"""Reuse the frozen PD harness in a new output namespace, including spawn."""
import os
import sys
from .common import redirect
from .patches import install

if os.environ.get('B_NUMERICAL_RUN'):
    redirect()
    install()
    from .transfer_probe import install as transfer_install
    transfer_install()
    from .capture import install as capture_install
    capture_install()


def main():
    module=redirect()
    install()
    from .order_probe import install as order_install
    order_install(module)
    if os.environ.get('B_NUMERICAL_NVTX')=='1':
        import torch
        original=module.serve
        def serve(*args,**kwargs):
            active=not kwargs.get('warmup',False)
            if active:torch.cuda.nvtx.range_push('diagnostic_measured')
            try:return original(*args,**kwargs)
            finally:
                if active:torch.cuda.nvtx.range_pop()
        module.serve=serve
    watcher=None
    if len(sys.argv)==2:
        import json
        from pathlib import Path
        cfg=json.loads(Path(sys.argv[1]).read_text())
        if cfg.get('quiet'):
            from .common import ROOT
            from .fabric_watch import FabricWatch
            folder=ROOT/'raw'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
            watcher=FabricWatch(folder,cfg['gpus']);watcher.start()
    try:module.main()
    finally:
        if watcher:watcher.close()


if __name__=='__main__': main()
