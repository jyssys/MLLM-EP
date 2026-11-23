"""Test one bounded lower-observer-cost capture bundle after the fixed controls."""
import json
import time
import numpy as np
from .common import ROOT, write, digest, dispatch_allowed
from .c3_campaign import qualified_registry, performance, launch, wait_file
from .c3_activity import observation
from .c0_analysis import csv_save


def prepare():
    target=ROOT/'jobs/c3_native_single_role_activity_recovery_v11.json';assert not target.exists()
    item=qualified_registry()['NativeStrong'];tests=[];lanes=[[],[],[],[]]
    for role in ('prefill','decode'):
        jobs=[]
        for stage in ('Clean1','Profile','Clean2'):
            name=f'V2_C3ActivityRecovery_q2.0_{role}_{stage}_V11'
            flags=observation(stage)
            flags.update(kernel_activity_roles=[role],activity_ROI_seconds=5,quiescent_witness=False,
                observer_scope='Bounded recovery bundle: one role,5-second ROI,no quiescent/deadline witness. '
                    'Keep the original5% full-trace and exposed-request gate. This is not an isolated attribution to one observer feature.')
            performance(item,name,'longbench_balanced400_v2.json',[4,5],2.0,2026112301,observation=flags)
            jobs.append(name);lanes[2].append(name)
        tests.append(dict(label='NativeStrong',qps=2.0,role=role,pair=[4,5],jobs=jobs))
    write(target,dict(time=time.time(),lanes=lanes,tests=tests,stop_on_failure=False,
        original_failed_capture_retained=True,full_trace_and_ROI_limit_fraction=.05,
        output_work_limit_fraction=.05,
        hypotheses=[
            'If dual-device collection contributes,the single-role bundle can reduce the exposed-request latency change.',
            'If extra state observation contributes,removing that witness can reduce the observed change.',
            'If natural batch/output variability dominates,even the bounded bundle can fail the unchanged gate.'],
        scope='Prospective recovery after the dual-role Native2.0/4.8 observer gates failed. '
            'Test each role once. Do not repeat until a gate passes. The observer bundle changes three instrumentation choices. '
            'Do not attribute a passing bundle to one of those choices. P/D are separate independent captures,not one simultaneous window. '
            'Keep all model,scheduler,source,sampling,dataset,arrival and client timestamps unchanged. '
            'No active-decode offload is introduced. Unvalidated executable windows remain NOT_IDENTIFIED.'))
    return target.name


def analyze():
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.window_analysis import role_analysis,request_metrics
    from .c3_analysis import actual_quality
    plan=json.loads((ROOT/'jobs/c3_native_single_role_activity_recovery_v11.json').read_text());results=[]
    fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms')
    for case in plan['tests']:
        receipts=[ROOT/'raw'/name/'dispatcher_result.json' for name in case['jobs']]
        values=[json.loads(p.read_text()) for p in receipts if p.exists()]
        if len(values)!=3 or not all(v['complete'] for v in values):
            status='FAILED' if any(not v['complete'] for v in values) else 'NOT_RUN' if not values else 'PARTIAL_OR_NOT_RUN'
            results.append(dict(case,evidence=status));continue
        try:
            role=role_analysis(case['jobs'][1],case['role'])
            roi=json.loads((ROOT/'raw'/case['jobs'][1]/f'activity_{case["role"]}/actual_ROI.json').read_text())
            left=roi['left']['host_lower_ns']/1e9;right=roi['right']['host_upper_ns']/1e9
            queries=json.loads((ROOT/'raw'/case['jobs'][1]/f'actual_queries_{case["role"]}.json').read_text())['rows']
            ids={r['request_id'] for r in queries if left<=r['time']<=right};assert ids
            metrics=[];local=[];quality=[]
            for name in case['jobs']:
                metrics.append(json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text()))
                requests=[json.loads(s) for s in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
                local.append(request_metrics([r for r in requests if 'q'+str(r['sample_id']) in ids]))
                quality.append(actual_quality(name)['passed'])
            ratios={f:metrics[1][f]/((metrics[0][f]+metrics[2][f])/2) for f in fields}
            local_ratios={f:local[1][f]/((local[0][f]+local[2][f])/2) for f in fields}
            work=metrics[1]['output_tokens']/((metrics[0]['output_tokens']+metrics[2]['output_tokens'])/2)
            passed=all(abs(v-1)<=.05 for v in [*ratios.values(),*local_ratios.values(),work]) and all(quality)
            gate=dict(passed=passed,status='PASS' if passed else 'DIAGNOSTIC',
                full_trace_profile_over_mean_clean=ratios,matched_exposed_requests_profile_over_mean_clean=local_ratios,
                profile_output_over_mean_clean=work,quality_passed=quality,exposed_request_ids=sorted(ids),independent_triplets=1)
            results.append(dict(case,role_result=role,observer_gate=gate,
                evidence='MEASURED_KERNEL_ACTIVITY' if passed else 'DIAGNOSTIC',
                decode_safe_executable_window='NOT_IDENTIFIED'))
        except Exception as error:results.append(dict(case,evidence='FAILED',error=repr(error)))
    write(ROOT/'analysis/c3_native_single_role_activity_recovery_v11.json',dict(time=time.time(),results=results,
        plan_SHA256=digest(ROOT/'jobs/c3_native_single_role_activity_recovery_v11.json'),
        scope=plan['scope']+' One CPC triplet is not a serving-performance CI. No operator-exclusive,SM/HBM or executable-window claim.'))
    csv_save(ROOT/'figures/c3_native_single_role_activity_recovery_v11.csv',[
        dict(role=r['role'],QPS=r['qps'],evidence=r['evidence'],
            qualified_kernel_busy_fraction=r['role_result']['kernel_busy_fraction'] if r.get('observer_gate',{}).get('passed') else None,
            diagnostic_kernel_busy_fraction=r.get('role_result',{}).get('kernel_busy_fraction'),observer_gate=r.get('observer_gate'))
        for r in results])
    print('BOUNDED_SINGLE_ROLE_CPC',[(r['role'],r['evidence']) for r in results],flush=True)


def run():
    name='c3_native_single_role_activity_recovery_v11.json'
    if not (ROOT/'jobs'/name).exists():prepare()
    wait_file(ROOT/'state/c3_knob_confirmation_complete_v2.json')
    # The same pair serves the prospectively required equal-output diagnostic.
    # It has priority over this optional observer recovery.
    wait_file(ROOT/'state/c3_primary_knee_replay_complete_v5.json')
    if not dispatch_allowed():return
    result=launch(name,allow_partial_budget=True);analyze()
    write(ROOT/'state/c3_native_single_role_activity_recovery_complete_v11.json',dict(time=time.time(),complete=result['complete']))


if __name__=='__main__':run()
