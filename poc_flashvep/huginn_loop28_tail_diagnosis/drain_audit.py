"""Observe the actual advertised forecast and its known Decode cohort."""
import os,time
from .common import *


def install(cfg):
    from vllm_rlt.engine.worker import StageWorker
    send=StageWorker.send
    records=[]
    def observed(worker,kind,**fields):
        if worker.engine.stage_config.role=='decode':
            connector=worker.connector
            if kind=='idle_capacity' and 'known_decode_drain_forecast_ms' in fields:
                cohort=[dict(request_id=r.request_id,seed=r.sampling_params.seed,
                    cap=r.sampling_params.max_tokens,phase=r.phase.value,
                    observed_outputs=connector.output_counts.get(r.request_id,0))
                    for r in connector.active_decode()]
                records.append(dict(event='ADVERTISED_FORECAST',time=time.monotonic(),
                    observed=fields['observed'],prediction_ms=fields['known_decode_drain_forecast_ms'],
                    remaining_outputs=fields['known_decode_remaining_outputs'],cohort=cohort))
            elif kind=='output':
                output=fields['output']
                records.append(dict(event='OUTPUT',time=time.monotonic(),request_id=output.request_id,
                    tokens=len(output.token_ids),finished=output.finished))
            elif kind=='stopped':
                write(ROOT/'raw'/cfg['name']/f'drain_forecast_audit_{os.getpid()}.json',dict(records=records,
                    scope='Actual outgoing D advertisements and D output-send receipts. Evaluate only the request identities known in each advertisement. This is Host completion prediction, not physical GPU idle or admission safety.'))
        return send(worker,kind,**fields)
    StageWorker.send=observed


def prepare():
    initialize();jobs=[]
    for cap in (16,4):
        source=f'forecast8b_output16_loop_seed2026110940' if cap==16 else 'forecast8_output4_loop_seed2026110940'
        cfg=json.loads((OLD_ROOT/'configs'/(source+'.json')).read_text())
        parent=OLD_ROOT/'raw'/cfg['required_qualification']/'qualification_with_frozen_reference.json'
        assert json.loads(parent.read_text())['passed']
        cfg.update(name=f'idle_drain_output{cap}_audit_n64',count=64,
            request_caps=cfg['request_caps'][:64],prompt_ids=cfg['prompt_ids'][:64],
            arrival_offsets=cfg['arrival_offsets'][:64],required_qualification=None,
            diagnosis_parent=dict(configuration=source,qualification=str(parent),qualification_sha256=digest(parent)),
            diagnosis_events=False,drain_forecast_audit=True,clean_confirmation=False,
            phase='SHORT_EXISTING_FORECAST_OBSERVATION_NOT_NEW_POLICY',timeout=420,startup_timeout=240)
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/drain_audit_plan.json',dict(jobs=jobs))
    print(' '.join(jobs))


if __name__=='__main__':prepare()
