"""Minimal optional hooks. Qualified KV, arithmetic and native ACK stay intact."""
import os
from .common import *
INSTALLED=False


def install():
    global INSTALLED
    if INSTALLED:return
    INSTALLED=True
    cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text())
    if cfg.get('projection_graph_rows'):
        from .tile_graph import install_factory
        install_factory(cfg)
    if cfg.get('D_probe_tokens'):
        from .decode_probe import install as probe
        probe(cfg)
    if cfg.get('D_probe_warmup_tokens'):
        from .probe_warmup import install as probe_warmup
        probe_warmup(cfg)
    if cfg.get('P_prefill_graph_rows'):
        from .prefill_graph import install as prefill_graph
        prefill_graph(cfg)
    if cfg.get('observe_P_submission_readiness'):
        from .prefill_inflight import install as inflight
        inflight(cfg)
    if cfg.get('autonomous_capture_seconds'):
        from .timed_capture import install as timed_capture
        timed_capture(cfg)
    if cfg.get('postwarm_capture'):
        from .postwarm_capture import install as postwarm_capture
        postwarm_capture(cfg)
    if cfg.get('bounded_native_capture_seconds'):
        from .bounded_native_capture import install as bounded_native_capture
        bounded_native_capture(cfg)
