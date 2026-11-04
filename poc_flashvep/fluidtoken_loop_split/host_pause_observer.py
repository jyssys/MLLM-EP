"""Record long host calls without changing CUDA or transport execution."""
import atexit
import os
import time
import sys
from .common import ROOT, write


def install(cfg):
    if not cfg.get('observe_host_pauses'):
        return
    threshold = .02
    events = []
    finished = [False]

    def observe(owner, method, label):
        original = getattr(owner, method)
        def measured(*args, **kwargs):
            start = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                end = time.monotonic()
                if end - start >= threshold:
                    batch = kwargs.get('batch')
                    kind = None
                    for value in args[1:]:
                        if isinstance(value, dict) and 'kind' in value:
                            kind = value['kind']
                        elif isinstance(value, str) and label.endswith('_send'):
                            kind = value
                    events.append(dict(method=label, start=start, end=end,
                        seconds=end - start, message_kind=kind,
                        path=str(args[0]) if label.startswith('evidence_') and args else None,
                        logical_rows=getattr(batch, 'num_rows', None)))
        setattr(owner, method, measured)
        return original, measured

    def flush():
        if finished[0]:
            return
        finished[0] = True
        write(ROOT / 'raw' / cfg['name'] / f'host_pauses_{os.getpid()}.json', dict(
            pid=os.getpid(), threshold_seconds=threshold, events=events,
            scope='Host-call enclosures. Nested intervals overlap and must not be added. No GPU execution time is inferred.'))

    import torch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.pd.engine import PDEngine
    from vllm_rlt.pd.transport import NixlConnector
    from vllm_rlt.pd.connector import PDStageConnector
    from vllm_rlt.engine.llm_engine import StageClient
    from vllm_rlt.engine.worker import StageWorker
    for owner, method, label in (
        (torch.cuda, 'mem_get_info', 'cuda_mem_get_info'),
        (ModelRunner, 'submit', 'model_submit'),
        (StageEngine, 'step', 'stage_step'),
        (NixlConnector, 'submit', 'nixl_submit'),
        (NixlConnector, 'poll', 'nixl_poll'),
        (PDStageConnector, 'progress', 'connector_progress'),
        (StageClient, 'send', 'client_send'),
        (StageClient, '_receive', 'client_receive'),
        (StageWorker, 'send', 'worker_send'),
        (StageWorker, 'command', 'worker_command'),
        (PDEngine, '_message', 'coordinator_message')):
        observe(owner, method, label)
    if cfg.get('observe_evidence_io'):
        from ..looped_pd_headroom_recovery import common as recovery
        from ..looped_pd_autonomous_headroom import common as autonomous
        for owner in (recovery, autonomous):
            for method in ('append', 'write'):
                original, measured = observe(owner, method, 'evidence_' + method)
                for name, module in list(sys.modules.items()):
                    if name.startswith('poc_flashvep.') and module is not None:
                        if getattr(module, method, None) is original:
                            setattr(module, method, measured)
    atexit.register(flush)
    for cls in (StageEngine, PDEngine):
        original = cls.close
        def close(engine, *args, _original=original, **kwargs):
            try:
                return _original(engine, *args, **kwargs)
            finally:
                flush()
        cls.close = close
