"""Record native graph counters without changing graph selection."""
import os
from .common import ROOT, write


def install(cfg):
    if not cfg.get('observe_native_graph_coverage'):
        return
    from vllm_rlt.engine.stage_engine import StageEngine
    step, close = StageEngine._step, StageEngine.close
    baseline = {}
    marker = ROOT / 'raw' / cfg['name'] / 'measurement_begin.json'

    def counters(engine):
        graphs = engine.model_runner.graphs
        return dict(captures=graphs.captures, replays=graphs.replays, fallbacks=graphs.fallbacks,
                    row_limit=graphs.execution.cuda_graph_max_batch_size) if graphs else {}

    def stepped(engine, *args, **kwargs):
        if not baseline and marker.exists():
            baseline.update(counters(engine))
        return step(engine, *args, **kwargs)

    def closed(engine):
        after = counters(engine)
        write(ROOT / 'raw' / cfg['name'] / f'native_graph_coverage_{os.getpid()}.json', dict(
            role=engine.stage_config.role, before=baseline, after=after,
            measured_delta={key: after[key] - baseline[key] for key in ('captures', 'replays', 'fallbacks') if key in baseline},
            scope='Native recurrent graph counters. No change to selection, operands, or model work.'))
        return close(engine)
    StageEngine._step, StageEngine.close = stepped, closed
