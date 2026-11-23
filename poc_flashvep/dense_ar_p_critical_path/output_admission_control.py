"""Discriminate decode-residency and P-work hypotheses at fixed Output192."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,bind,dispatch_allowed,preserve

PLAN='output192_D_cap_intervention_v1.json'


def prepare():
    from .cap_intervention import template
    runner=bind();mapping={'NativeStrong':'NativeStrong','NativeCommon2':'NativeCommon2','NativeCommon4':'NativeCommon4',
        'StockToken':'StockToken','OriginalLayer':'OriginalLayer','OptimizedLayer':'Layer8Cap4',
        'OriginalHybrid':'OriginalHybrid','OptimizedHybrid':'HybridIsoCap4'}
    names={};data_name='balanced400_v2_heldout.json';data=json.loads((ROOT/'data'/data_name).read_text())
    seed=2026112381;count=len(data)
    for label,parent in mapping.items():
        cfg=template(parent);name=f'PC_Output192_DCap_s0_{label}'
        cfg.update(name=name,gpus=[4,5],dataset=data_name,dataset_SHA256=digest(ROOT/'data'/data_name),count=count,
            correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
            P_event_timing=False,kernel_activity=False,coordinator_trace=False,ignore_eos=True,output_cap=192,
            replay_output_lengths=[192]*count,diagnostic_controlled_output_replay=True,
            qps=2.,seed=seed,arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(.5,count-1))].tolist(),
            scope='Predeclared N1 Output192 D-cap control. Same-axis cap2→4 changes only D admission. '
                'Keep the same source credit,geometry,compiler path and output work. No N3 or serving-success claim.')
        if cfg.get('axis_kind')=='Token':
            from ..dense_ar_output_aware_v2.work_contract import regenerate
            regenerate(cfg,data)
        runner.save(cfg);names[label]=name
    target=ROOT/'jobs'/PLAN;assert not target.exists()
    write(target,dict(time=time.time(),arms=list(mapping),lanes=[[],[],list(names.values()),[]],stop_on_failure=True,
        blocks=[dict(mode='Fixed192',start=0,seed=seed,qps=2.,pair=[4,5],arms=names,order=list(mapping))],
        additional_decode_references=['NativeCommon4'],
        analysis_scope='N1 controlled Output192 D admission intervention. The short-answer C1-replay cap4 Discovery failed its TTFT screen. '
            'That result is retained and is not extrapolated to the longer decode-residency regime. '
            'Fresh cap2 and cap4 controls share the physical pair,prompts,arrival,exact output192,source limit,cut and compiler contract. '
            'Keep NativeStrong,Common2,exact Common4 and strong uncapped compiled Stock. '
            'This is a separate original-runtime family,not the new common warm-cache policy. '
            'N1 cannot confirm relative5percent decode safety or a serving win. Do not pool historical cap2 starts into N3. '
            'No safe-window predictor or active-decode insertion is implemented.'))
    log=ROOT/'logs/output192_D_cap_intervention_v1.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.output_admission_control','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/output192_D_cap_controller_v1.json',dict(time=time.time(),pid=child.pid));return child.pid


def run():
    runner=bind();parent=json.loads((ROOT/'state/p_output_stress_controller.json').read_text())['pid']
    complete=ROOT/'state/p_output_stress_c3_v1_lane2_complete.json'
    def stop(reason):
        write(ROOT/'state/output192_D_cap_not_run.json',dict(time=time.time(),status='NOT_RUN',reason=reason))
    while not complete.exists():
        try:
            proc=Path('/proc',str(parent));assert proc.stat().st_uid==os.getuid()
            assert b'poc_flashvep.dense_ar_p_critical_path.output_stress' in (proc/'cmdline').read_bytes()
        except (OSError,AssertionError):
            if complete.exists():break
            stop('The original Output stress handle is missing before terminal completion. Inspect it; do not restart it.');return
        if not dispatch_allowed():stop('Dispatch budget ended while the original stress experiment retained the pair.');return
        time.sleep(15)
    old=json.loads((ROOT/'jobs/p_output_stress_c3_v1.json').read_text())
    assert all((ROOT/'raw'/name/'dispatcher_result.json').exists() and
        json.loads((ROOT/'raw'/name/'dispatcher_result.json').read_text())['complete'] for name in old['ordered_jobs'])
    budget=json.loads((ROOT/'state/budget.json').read_text())
    if time.time()+125*60>budget['deadline_epoch']:
        stop('Less than125minutes remain for this eight-arm diagnostic and resource/publication closeout. No partial new grid is launched.');return
    runner.lane(PLAN,2);preserve()


def realization_analysis():
    """Check whether a larger configured cap changes actual decode batches."""
    bind();preserve()
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from ..dense_ar_longbench400_pd import registration_deferral_analysis as registration
    from .critical_analysis import csv_save
    registration.ROOT=ROOT
    plan=json.loads((ROOT/'jobs'/PLAN).read_text());rows=[];contracts={}
    for label,name in plan['blocks'][0]['arms'].items():
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());contracts[label]=cfg
        metric=summarize(name);runtime=json.loads((folder/'actual_runtime_decode.json').read_text())
        histogram=metric['actual_geometry']['decode']['prefill_decode_request_histogram']
        pure=[value for value in histogram if value['prefill']==0 and value['decode']>0]
        steps=sum(value['steps'] for value in pure)
        deferred=bool(cfg.get('defer_D_request_registration'))
        queue=registration.transition(name)['D_host_registration_queue'] if deferred else None
        rows.append(dict(label=label,name=name,evidence='TRACE_ANALYSIS',independent_starts=1,
            source_SHA256=cfg['source_capsule_SHA256'],dataset_SHA256=cfg['dataset_SHA256'],GPU_pair=cfg['gpus'],
            QPS=cfg['qps'],seed=cfg['seed'],axis_cut=cfg.get('axis_cut'),hybrid_suffix_fraction=cfg.get('hybrid_suffix_fraction'),
            source_credit=cfg.get('continuous_source_credit_limit',0),configured_D_cap=cfg.get('D_running_admission_cap'),
            actual_scheduler_max_running=runtime['runtime_max_num_running_requests'],
            pure_decode_steps=steps,pure_decode_maximum_requests=max((value['decode'] for value in pure),default=None),
            pure_decode_step_weighted_mean_requests=sum(value['steps']*value['decode'] for value in pure)/steps if steps else None,
            prefill_decode_histogram=histogram,registration_policy_enabled=deferred,
            registration_queue=queue,registration_scope='Separate host-registration queue. Not a scheduler queue or GPU idle interval.'
                if deferred else 'NOT_APPLICABLE_POLICY_DISABLED',
            TTFT_p95_ms=metric['TTFT_p95_ms'],ITL_p99_ms=metric['ITL_p99_ms'],TPOT_p95_ms=metric['TPOT_p95_ms'],
            scope='N1. A configured running cap is not an achieved decode batch. Do not interpret a weak cap response as proof of physical compute saturation.'))
    comparisons=[];by_label={row['label']:row for row in rows}
    controls=('axis_cut','hybrid_suffix_fraction','continuous_source_credit_limit','source_capsule_SHA256',
        'P_compile_prefix','D_compile_partial','defer_D_remote_setup','defer_D_request_registration','d_compute_throttle',
        'dataset_SHA256','arrival_offsets','replay_output_lengths','ignore_eos','output_cap')
    for axis in ('Layer','Hybrid'):
        left,right='Optimized'+axis,'Original'+axis
        if left not in by_label or right not in by_label:continue
        assert all(contracts[left].get(key)==contracts[right].get(key) for key in controls)
        a,b=by_label[left],by_label[right]
        comparisons.append(dict(candidate=left,control=right,controlled_fields_equal=True,
            configured_caps=[b['configured_D_cap'],a['configured_D_cap']],
            actual_decode_maxima=[b['pure_decode_maximum_requests'],a['pure_decode_maximum_requests']],
            actual_decode_step_means=[b['pure_decode_step_weighted_mean_requests'],a['pure_decode_step_weighted_mean_requests']],
            TTFT_ratio=a['TTFT_p95_ms']/b['TTFT_p95_ms'],independent_starts=1,
            interpretation='The running cap changed. Inspect realized batches and host registration before identifying a service-capacity cause. '
                'No independent N3,exclusive GPU cost,or safe-window benefit is established.'))
    value=dict(time=time.time(),rows=rows,comparisons=comparisons,plan_SHA256=digest(ROOT/'jobs'/PLAN),
        registration_source_SHA256=digest(Path(registration.__file__)),
        scope='Read-only realization check of the predeclared Output192 intervention. No new GPU run or runtime change. '
            'A cap that does not produce larger decode batches cannot be treated as a decode-batch intervention. '
            'The registration policy waits while an output-started request remains running. Its duration is host waiting,not GPU kernel time.')
    write(ROOT/'analysis/output192_D_admission_realization.json',value)
    csv_save(ROOT/'figures/output192_D_admission_realization.csv',rows)
    preserve();print('OUTPUT192_ADMISSION_REALIZATION',len(rows),len(comparisons),flush=True);return value


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':print(prepare(),flush=True)
    elif sys.argv[1]=='analyze':
        from .confirmation_analysis import analyze
        analyze(PLAN,'output192_D_cap_intervention')
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='realization':realization_analysis()
    else:raise SystemExit('Expected prepare,run,analyze,or realization')
