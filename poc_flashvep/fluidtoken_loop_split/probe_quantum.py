"""Optional exact local-probe pacing at native recurrent boundaries.

This is a probe-only scheduling diagnostic. It does not migrate serving work.
The existing staged executor owns all hidden states and valid KV frontiers.
"""
from dataclasses import replace
import os


class OutputCredit:
    def __init__(self):
        self.credit = 1
        self.submissions = 0
        self.refreshes = 0

    def allows(self, background):
        return not background or self.credit > 0

    def consume(self, background):
        if background:
            if self.credit <= 0:
                raise RuntimeError('Probe exceeded its observed-output budget')
            self.credit -= 1
        self.submissions += 1

    def refresh(self):
        self.credit = 1
        self.refreshes += 1


def install(cfg):
    if not cfg.get('D_local_probe_output_pacing'):
        return
    from .common import ROOT, prior, write
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    prefix = 'autonomous_probe_'
    initialize, step, close = StageEngine.__init__, StageEngine.step, StageEngine.close
    ready, submit = StageEngine._execution_ready, ModelRunner.submit

    def background(engine):
        return any(not r.request_id.startswith(prefix)
                   and str(getattr(r.phase, 'value', r.phase)).lower() in
                   ('prelude', 'recurrent', 'coda', 'speculative')
                   for r in engine.scheduler.requests.values())

    def initialized(engine, *args, **kwargs):
        role = kwargs.get('stage_config')
        if role is not None and role.role == 'decode':
            scheduler = kwargs['scheduler_config']
            if scheduler.mode != 'refill':
                raise ValueError('The pacing control requires separate-phase refill scheduling')
            kwargs['scheduler_config'] = replace(scheduler, wavefront_chunked_prefill=True)
        result = initialize(engine, *args, **kwargs)
        if engine.stage_config.role == 'decode':
            engine._local_probe_credit = OutputCredit()
            engine.model_runner._local_probe_engine = engine
            write(ROOT / 'raw' / cfg['name'] / f'probe_quantum_configuration_{os.getpid()}.json', dict(
                role='decode', wavefront=engine.scheduler.wavefront_chunked_prefill,
                maximum_submissions_per_observed_output_step=1,
                scope='Independent probes only. No serving request is migrated or approximated.'))
        return result

    def eligible(engine, request):
        accepted = ready(engine, request)
        credit = getattr(engine, '_local_probe_credit', None)
        phase = str(getattr(request.phase, 'value', request.phase)).lower()
        if credit is None or not request.request_id.startswith(prefix) or not phase.startswith('prefilling_'):
            return accepted
        return accepted and credit.allows(background(engine))

    def submitted(runner, scheduled_batch=None, *, batch):
        engine = getattr(runner, '_local_probe_engine', None)
        probes = [s for s in batch.spans if s.request.request_id.startswith(prefix)
                  and str(getattr(s.request.phase, 'value', s.request.phase)).lower().startswith('prefilling_')]
        if engine is not None and probes:
            if len(probes) != 1:
                raise RuntimeError('The paced control permits only one local probe in flight')
            if batch.loop_count > 1:
                raise RuntimeError('A paced probe must not execute multiple recurrent loops in one submission')
            engine._local_probe_credit.consume(background(engine))
            prior.append(ROOT / 'raw' / cfg['name'] / 'probe_quantum_submissions.jsonl', dict(
                time=__import__('time').monotonic(), loop_count=batch.loop_count,
                rows=batch.num_rows, request_ids=[s.request.request_id for s in probes]))
        return submit(runner, scheduled_batch, batch=batch)

    def stepped(engine):
        outputs = step(engine)
        if hasattr(engine, '_local_probe_credit') and any(
                not output.request_id.startswith(prefix) and output.token_ids for output in outputs):
            engine._local_probe_credit.refresh()
        return outputs

    def closed(engine):
        if hasattr(engine, '_local_probe_credit'):
            write(ROOT / 'raw' / cfg['name'] / f'probe_quantum_accounting_{os.getpid()}.json',
                  vars(engine._local_probe_credit))
        return close(engine)

    StageEngine.__init__, StageEngine._execution_ready = initialized, eligible
    ModelRunner.submit, StageEngine.step, StageEngine.close = submitted, stepped, closed
