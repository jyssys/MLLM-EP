"""Avoid building the CODA descriptor that the qualified partial path discards."""
import os
from .common import ROOT, write


def unused_partial_coda(batch, total):
    return (batch.operation.value == 'prelude' and batch.loop_count > 0
        and any(span.depth+batch.loop_count < total for span in batch.spans))


def install(cfg):
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    prepare, coda, close = (ModelRunner._prepare_loop_inputs,
        ModelRunner._prepare_prefill_coda, StageEngine.close)
    counts = dict(skipped_unused_coda_descriptors=0)

    def prepared(runner, batch, packed_metadata=None):
        previous = getattr(runner, 'net_unused_partial_coda', False)
        runner.net_unused_partial_coda = unused_partial_coda(batch, cfg['depth'])
        try:
            return prepare(runner, batch, packed_metadata)
        finally:
            runner.net_unused_partial_coda = previous

    def descriptor(runner, *args, **kwargs):
        if getattr(runner, 'net_unused_partial_coda', False):
            counts['skipped_unused_coda_descriptors'] += 1
            return None, None
        return coda(runner, *args, **kwargs)

    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT/'raw'/cfg['name']/f'partial_prepare_{os.getpid()}.json', dict(
                role=engine.stage_config.role, **counts,
                scope='Only descriptor construction for a CODA forward already discarded by Native partial-path preparation is skipped. Full-depth CODA computation and every recurrent operation remain unchanged.'))
    ModelRunner._prepare_loop_inputs, ModelRunner._prepare_prefill_coda = prepared, descriptor
    StageEngine.close = closed
