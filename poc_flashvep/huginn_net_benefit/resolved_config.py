"""Record resolved Native PD values, including inherited defaults."""
import os
from dataclasses import asdict
from .common import ROOT, write


def install(cfg):
    from vllm_rlt.pd.engine import PDEngine
    original = PDEngine.__init__
    def initialized(engine, *args, **kwargs):
        result = original(engine, *args, **kwargs)
        write(ROOT/'raw'/cfg['name']/f'resolved_native_PD_{os.getpid()}.json', dict(
            requested_overrides=cfg.get('native_transport_limits', {}),
            resolved=asdict(engine.config),
            scope='Actual Native PDConfig after constructor wrappers resolve defaults and explicit overrides. Do not compare only a top-level experiment field.'))
        return result
    PDEngine.__init__ = initialized
