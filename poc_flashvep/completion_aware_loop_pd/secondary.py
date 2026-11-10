"""Prepare qualified recovery diagnostics without replacing frozen primary starts."""
import argparse
from collections import defaultdict
from .common import *
from .prepare import prepare,config
from .calibrate import new_config
from .analyze import percentile


def cycles(name):
    groups=defaultdict(list)
    for p in (ROOT/'raw'/name).glob('sparse_device_*.json'):
        v=json.loads(p.read_text())
        if v['role']!='decode':continue
        starts={}
        for row in v['rows']:
            if row['prompt']:continue
            for request in row['requests']:
                key=(request['tid'],request['start'])
                if row['operation']=='prelude':starts[key]=(row['begin']['time'],row['rows'])
                elif row['operation']=='coda' and key in starts:
                    start,batch=starts.pop(key)
                    groups[max(batch,row['rows'])].append(1000*(row['end']['time']-start))
    values=[t for group in groups.values() for t in group]
    assert values,name
    return dict(name=name,p99_ms=percentile(values,.99),count=len(values),
        batch_p99_ms={str(k):percentile(v,.99) for k,v in groups.items()},
        scope='GPU-clock Prelude-start to Coda-end enclosure. It includes host submission gaps and waits. It is not a kernel union.')


def recovery_config(model,name,arm,count=32,seed=2026110920):
    cfg=(config(model,name,'native',count,seed) if arm=='native' else new_config(model,name,arm,count,seed))
    cfg.update(no_measured_graph_capture=True,phase='ABSOLUTE_SLO_ENVELOPE_RECOVERY_DIAGNOSTIC',
        secondary_arm=arm,secondary_scope='Not a Native-relative non-inferiority success. Frozen absolute SLOs remain unchanged.')
    if model=='huginn':cfg.update(D_kv_blocks=24832,minimum_HBM_free_bytes=3*1024**3)
    if arm!='native':
        source=(f'secondary_huginn_pool6_guarded_events128_calibration' if model=='huginn'
            else 'baseline_ouro_events128_calibration')
        cycle=cycles(source)
        stage=json.loads((ROOT/'data'/f'{model}_decode_enclosure_calibration.json').read_text())
        boundary=stage['boundary_guard_ms']
        per_loop=max(stage['loop_p99_ms'],(cycle['p99_ms']-boundary)/cfg['depth'])
        absolute=1112.9925646819174 if model=='ouro' else 1963.9920117333531
        cfg.update(device_decode_costs=dict(loop_p99_ms=per_loop,boundary_guard_ms=boundary,
            scope='Conservative cycle-enclosure allocation. Not an exclusive GPU per-loop measurement.'),
            cycle_reserve_calibration=cycle,
            native_ITL_target_ms=65.08287642151119 if model=='ouro' else 192.8212538175284,
            feedback_ITL_limit_ms=65.08287642151119 if model=='ouro' else 192.8212538175284,
            additional_work_TPOT_limit_ms=48.3363260853697 if model=='ouro' else 167.91435532271862,
            reserve_known_decode_transitions=True,
            recover_empty_decode_epoch=True,
            source_local_admission_filter=True,
            source_shared_completion_filter=True,
            completion_deadline_ms=1000. if model=='ouro' else 1200.,
            policy_revision='absolute-envelope-cycle-reserve-diagnostic-v2')
        if arm=='token':cfg.update(initial_token_budget_selection=True,token_single_offer_fallback=True)
    return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('calibration','N8','N32','selection'));a=p.parse_args()
    prepare();jobs=[]
    if a.stage=='calibration':
        for suffix,trace in (('clean',False),('events',True)):
            cfg=config('huginn',f'secondary_huginn_pool6_guarded_{suffix}128_calibration',trace=trace,seed=2026110920)
            cfg.update(D_kv_blocks=24832,no_measured_graph_capture=True,minimum_HBM_free_bytes=3*1024**3,
                phase='SIX_REQUEST_KV_POOL_NATIVE_CALIBRATION_NOT_PRIMARY')
            save(cfg);jobs.append(cfg['name'])
    else:
        count=8 if a.stage=='N8' else 32
        for model in ('ouro','huginn'):
            for arm in (('native','loop','token') if a.stage=='selection' else ('loop','token')):
                suffix=f'state{count}' if a.stage in ('N8','N32') else f'selection{count}'
                name=f'secondary_{model}_{arm}_{suffix}'
                cfg=recovery_config(model,name,arm,count=count)
                if a.stage in ('N8','N32'):
                    q=json.loads((ROOT/'configs'/f'q1_{model}_{arm}_state8.json').read_text())
                    for k in ('state_probe','verify_transfer','async_prompt_state_probe','async_verify_handoff',
                        'state_probe_measured_only','state_hash_defer_until_close','qualification_ignore_observer_tail_latch'):
                        cfg[k]=q[k]
                    cfg.update(state_pinned_slots=count+2,completion_deadline_ms=20000.,
                        arrival_offsets=[0.]*count,
                        arrival_seed=None,arrival_rate=None,timeout=1800,
                        require_actual_offload_qualification=True,
                        native_transport_limits=dict(shutdown_timeout=300),
                        phase=f'RECOVERY_{a.stage}_EXACT_STATE_NOT_PERFORMANCE')
                    if a.stage=='N32':cfg['required_qualification']=f'secondary_{model}_{arm}_state8'
                elif arm!='native':cfg['required_qualification']=f'secondary_{model}_{arm}_state32'
                save(cfg);jobs.append(name)
    write(ROOT/'state'/f'secondary_{a.stage}_jobs.json',jobs);print(' '.join(jobs))


if __name__=='__main__':main()
