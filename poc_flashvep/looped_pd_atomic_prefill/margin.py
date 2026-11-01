"""Validate finite staggered D service before a strict prefill-heavy label."""
import copy
from .common import *
from .analyze import serving,percentile
from .evidence import status,records,rows,full_work,transfer


def configurations():
    from .classify import main as classify
    candidates=classify();runs={r['name']:r for r in serving()};output=[];selection={}
    for model in MODEL_PINS:
        eligible=[r for r in candidates if r['model']==model and runs[r['name']]['phase']=='final_repeat' and
            r['chunk']==r['length'] and r['screen_P_candidate'] and r['P_near_strongest_finite_service_rate']]
        if not eligible:
            selection[model]=dict(status='NOT_RUN_NO_QUALIFIED_P_PRESSURE_SCREEN');continue
        selected=max(eligible,key=lambda r:r['actual_P_completion_rate']);source=runs[selected['name']]
        cfg=json.loads((ROOT/'configs'/(selected['name']+'.json')).read_text());count=32 if model=='qwen8' else 8 if model=='ouro' else 3
        rate=1.2*selected['actual_P_completion_rate'];cfg=copy.deepcopy(cfg)
        cfg.update(name=f'diagnostic_{model}_staggered_D_reserve_v1',phase='D_margin_diagnostic',count=count,
            request_caps=[cfg['cap']]*count,prompt_ids=cfg['prompt_ids'][:count],arrival_offsets=[0.]*count,
            prepared_D_arrival_offsets=[i/rate for i in range(count)],D_ready_pattern='FINITE_REGULAR_1P2X_OBSERVED_P_RATE',
            worker='ar_staggered_d' if model=='qwen8' else 'prepared_d_worker',prepare_D_only=model!='qwen8',
            P_slots=count,verify_transfer=model!='qwen8',state_probe=False,timeout=900,nsys=False,
            reference_serving_run=selected['name'],diagnostic_ready_rate=rate,
            diagnostic_scope='Finite native prepared-KV stress. No migration or Dynamic A. Does not certify arbitrary continuation SLOs.')
        if model=='qwen8':cfg['gpus']=cfg['gpus'][-1:]
        output.append(cfg);selection[model]=dict(source=selected['name'],diagnostic=cfg['name'],count=count,rate=rate,
            note='Small legal resident-KV cohort. Repeated short cohorts do not establish long-horizon sustainable capacity.')
    write(ROOT/'state/D_margin_selection.json',selection);return output


def analyze():
    selection_path=ROOT/'state/D_margin_selection.json';selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    verdicts={}
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text());summary=[]
    for model,choice in selection.items():
        if 'diagnostic' not in choice:continue
        name=choice['diagnostic'];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        values=records(ROOT/'raw'/name/'prepared_D_requests.jsonl');reference={r['sample_id']:r for r in rows(choice['source'])}
        slo=slos[model+'_L'+str(cfg['length'])]
        parity=len(values)==cfg['count'] and all(r['sample_id'] in reference and
            r['output_ids']==reference[r['sample_id']]['output_ids'] for r in values)
        tp=percentile([r['tpot_ms'] for r in values],.95) if values else None
        lag=percentile([1000*(r['actual']-r['scheduled']) for r in records(ROOT/'raw'/name/'prepared_D_release_events.jsonl')],.95)
        # Readiness-to-first-token delay is an additional diagnostic. It is
        # not the normal request TTFT, because preparation precedes readiness.
        ready=percentile([r['ttft_ms'] for r in values],.95) if values else None
        work=full_work(name) if model!='qwen8' and status(name)=='COMPLETE' else None
        byte=transfer(name) if model!='qwen8' and status(name)=='COMPLETE' else None
        passed=(status(name)=='COMPLETE' and parity and tp is not None and tp<=.9*slo['TPOT_ms'] and
            lag is not None and lag<=1000/cfg['diagnostic_ready_rate'] and (model=='qwen8' or work['passed'] and byte['passed']))
        value=dict(model=model,source=choice['source'],diagnostic=name,passed=bool(passed),count=cfg['count'],
            diagnostic_ready_rate=cfg['diagnostic_ready_rate'],exact_output_parity=parity,request_TPOT_p95_ms=tp,
            TPOT_margin_limit_ms=.9*slo['TPOT_ms'],release_lag_p95_ms=lag,ready_to_first_token_p95_ms=ready,
            full_work=work,valid_KV_hidden_byte_checks=byte,
            evidence='MEASURED_FINITE_PREPARED_STAGGERED_D',
            scope='Finite resident-KV stress at 1.2x observed P completion. This is not online sustainable capacity or a Prefill-offload safety guarantee.')
        summary.append(value)
        # Share only with the same input/output/chunk, whose measured P
        # completion rate does not exceed the rate actually stressed here.
        for r in serving():
            if (r['model']==model and r['length']==cfg['length'] and r['output_cap']==cfg['cap'] and
                r['arm']=='PD_G1' and r['chunk']==cfg['chunk'] and isinstance(r.get('P_completion_rate_central'),(int,float)) and
                1.2*r['P_completion_rate_central']<=cfg['diagnostic_ready_rate']*(1+1e-9)):verdicts[r['name']]=value
    write(ROOT/'state/D_margin_validation.json',verdicts);write(ROOT/'D_margin_summary.json',summary);return summary
