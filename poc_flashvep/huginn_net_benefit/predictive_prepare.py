"""Expose a checkpoint from a known-work forecast, never commit from that forecast."""
import os
import time
from .common import ROOT, write


def preparation_advertisement(ad, now, earliest_ms, config):
    if not ad or not ad.get('available') or now < ad['observed']:
        return ad
    if 1000*(now-ad['observed']) > config.get('advertisement_max_age_ms', 50.):
        return ad
    drain = ad.get('known_decode_drain_forecast_ms')
    if drain is None or drain > earliest_ms or drain <= 0:
        return ad
    budget = config.get('empty_quantum_cap_ms', 0.)
    if budget <= ad.get('future_quantum_budget_ms', ad.get('budget_ms', 0.)):
        return ad
    return dict(ad, budget_ms=budget, future_quantum_budget_ms=budget,
        admission_budget_ms=budget, decode_reserve_ms=0.,
        checkpoint_preparation_forecast_only=True)


def install(cfg):
    from vllm_rlt.engine.worker import StageWorker
    from vllm_rlt.engine.stage_engine import StageEngine
    init, closed = StageWorker.__init__, StageEngine.close
    folder = ROOT/'raw'/cfg['name']
    records = []

    def initialized(worker, *args, **kwargs):
        value = init(worker, *args, **kwargs)
        connector, engine = worker.connector, worker.engine
        if connector is None:
            return value
        if engine.stage_config.role == 'decode':
            send = connector.notify
            def notifying(kind, **fields):
                if kind == 'idle_capacity':
                    active = connector.active_decode()
                    remaining = max((max(0, request.sampling_params.max_tokens
                        - connector.output_counts.get(request.request_id, 0))
                        for request in active), default=0)
                    samples = sorted(connector.intervals)
                    cycle = max(cfg['preparation_decode_cycle_ms'],
                        samples[min(len(samples)-1, int(.95*len(samples)))] if samples else 0.)
                    fields['known_decode_drain_forecast_ms'] = remaining*cycle
                    fields['known_decode_remaining_outputs'] = remaining
                return send(kind, **fields)
            connector.notify = notifying
            return value
        scheduler = engine.scheduler
        token = cfg['intervention_method'] == 'dynamic_token'
        attribute = '_make_scheduled_request' if token else '_planned_loop_count'
        original = getattr(scheduler, attribute)
        def prepared(*args, **kwargs):
            if token:
                request = args[0]
                if getattr(request, 'idle_armed', False) is not None:
                    return original(*args, **kwargs)
                first = min(cfg['boundary_checkpoints'])/cfg['length']
            else:
                first = min(cfg['boundary_checkpoints'])/cfg['depth']
            old = connector.advertisement
            forecast = preparation_advertisement(old, time.monotonic(),
                first*cfg['preparation_source_prompt_ms'], cfg)
            if forecast is old:
                return original(*args, **kwargs)
            connector.advertisement = forecast
            records.append(dict(time=time.monotonic(),
                event='CHECKPOINT_PREPARATION_FORECAST_NOT_ADMISSION',
                known_decode_remaining_outputs=old['known_decode_remaining_outputs'],
                known_decode_drain_forecast_ms=old['known_decode_drain_forecast_ms'],
                estimated_source_boundary_ms=first*cfg['preparation_source_prompt_ms']))
            try:
                return original(*args, **kwargs)
            finally:
                connector.advertisement = old
        setattr(scheduler, attribute, prepared)
        return value

    def close(engine):
        try:
            return closed(engine)
        finally:
            write(folder/f'predictive_checkpoint_preparation_{os.getpid()}.json', dict(
                role=engine.stage_config.role, records=records,
                scope='Known Decode work and past cycle times predict checkpoint preparation only. Every offer, commit, and D execution uses the current unmodified admission rules. Future arrivals remain unknown.'))
    StageWorker.__init__, StageEngine.close = initialized, close
