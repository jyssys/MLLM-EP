"""Reject an unexpected scheduler mutation. Never suppress progress guards."""
import os


def check_contract(cfg):
    if cfg.get('D_local_probe_output_pacing'):
        raise ValueError('This comparison has no local probes. Disable the unused probe pacing hook.')
    if cfg.get('wavefront'):
        raise ValueError('Native grouped continuation requires the declared non-Wavefront contract.')


def install(cfg):
    check_contract(cfg)
    from vllm_rlt.engine.stage_engine import StageEngine
    from .common import ROOT, write
    initialize = StageEngine.__init__
    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        actual = engine.scheduler.wavefront_chunked_prefill
        write(ROOT / 'raw' / cfg['name'] / f'deep_scheduler_contract_{os.getpid()}.json', dict(
            role=engine.stage_config.role, requested_wavefront=False, actual_wavefront=actual,
            unused_probe_hook=False, passed=not actual,
            scope='Runtime configuration check. The native progress guard remains unchanged.'))
        if actual:
            raise RuntimeError('The effective native scheduler differs from the declared non-Wavefront contract')
        return result
    StageEngine.__init__ = initialized
