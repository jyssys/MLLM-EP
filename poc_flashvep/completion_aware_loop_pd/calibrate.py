"""Freeze empirical GPU-enclosure costs and separate diagnostic configurations."""
from collections import defaultdict
from .prepare import *
from .analyze import percentile,metrics


def calibration(model):
    configure();folder=ROOT/'raw'/f'baseline_{model}_events128_calibration'
    groups=defaultdict(list)
    for path in folder.glob('sparse_device_*.json'):
        value=json.loads(path.read_text())
        if value['role']=='decode':
            for r in value['rows']:
                if not r['prompt']:groups[r['operation']].append(r['CUDA_enclosure_ms'])
    assert groups['recurrent'] and groups['prelude'] and groups['coda']
    clean=metrics(f'baseline_{model}_clean128_calibration')
    instrumented=metrics(f'baseline_{model}_events128_calibration')
    loop=percentile(groups['recurrent'],.99)
    boundary=percentile(groups['prelude'],.99)+percentile(groups['coda'],.99)+2.
    cfg=config(model,'unused','loop')
    result=dict(model=model,loop_p99_ms=loop,boundary_guard_ms=boundary,
        native_ITL_target_ms=min(clean['frozen_SLO']['ITL_ms'],1.05*clean['ITL_p99_ms']),
        wall_ratio=None,TTFT_p95_ratio=instrumented['TTFT_p95_ms']/clean['TTFT_p95_ms'],
        ITL_p99_ratio=instrumented['ITL_p99_ms']/clean['ITL_p99_ms'],
        maximum_ITL_ratio=instrumented['maximum_ITL_ms']/clean['maximum_ITL_ms'],
        source_run=folder.name,counts={k:len(v) for k,v in groups.items()},
        scope='Empirical CUDA Event stream enclosures, including launch gaps and waits. Not exclusive kernel costs or a future safety guarantee.')
    write(ROOT/'data'/f'{model}_decode_enclosure_calibration.json',result)
    return result


def new_config(model,name,method='loop',count=24,seed=2026110910):
    cfg=config(model,name,method=method,count=count,seed=seed)
    c=json.loads((ROOT/'data'/f'{model}_decode_enclosure_calibration.json').read_text())
    cfg.update(source_monitor_boundaries=True,completion_reservations=True,
        require_whole_suffix_budget=False,max_continuations=2,negotiation_ttl_ms=50.,
        native_ITL_target_ms=c['native_ITL_target_ms'],feedback_ITL_limit_ms=c['native_ITL_target_ms'],
        device_decode_costs={k:c[k] for k in ('loop_p99_ms','boundary_guard_ms')},
        completion_deadline_ms=500. if model=='ouro' else 700.,
        initial_token_budget_selection=False,minimum_P_pressure=2,
        policy_revision='completion-reservation-v1',phase='COMPLETION_AWARE_CALIBRATION_NOT_CONFIRMATION')
    return cfg


def main():
    prepare();values={m:calibration(m) for m in ('ouro','huginn')};jobs=[]
    for model in ('ouro','huginn'):
        # A short physical-kernel diagnostic uses the existing qualified CUPTI collector.
        cfg=config(model,f'physical_{model}_both_kernel2s128',trace=False)
        cfg.update(kernel_activity_seconds=2.,capture_roles=['prefill','decode'],capture_arrival_fraction=.35,
            phase='CUPTI_KERNEL_DIAGNOSTIC_WITH_MATCHED_CLEAN_CONTROL')
        save(cfg);jobs.append(cfg['name'])
        for method in ('loop','token'):
            cfg=new_config(model,f'q1_{model}_{method}_state8',method,count=8)
            old=json.loads((OLD/'configs'/f'r9_{model}_actual_mixed_state8.json').read_text())
            for k in ('state_probe','verify_transfer','async_prompt_state_probe','async_verify_handoff',
                'state_probe_measured_only','state_hash_defer_until_close','state_pinned_slots',
                'qualification_ignore_observer_tail_latch'):
                cfg[k]=old[k]
            cfg.update(arrival_offsets=[0.]*8,arrival_rate=None,arrival_pattern='SIMULTANEOUS_STATE_DIAGNOSTIC',
                phase='N8_NEW_EXACT_STATE_QUALIFICATION_NOT_PERFORMANCE',completion_deadline_ms=5000.,
                timeout=900,startup_timeout=300)
            save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/calibration_jobs.json',jobs)
    print(json.dumps(dict(calibration=values,jobs=jobs),indent=2))


if __name__=='__main__':main()
