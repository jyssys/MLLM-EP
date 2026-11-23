"""Qualify short kernel/copy captures against clean continuous-serving runs."""
import argparse
import csv
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .c3_campaign import qualified_registry,performance,launch,wait_file
from .c3_policy import policy_registry
from .c0_analysis import csv_save

def observation(stage):
    return dict(kernel_activity=stage=='Profile',kernel_activity_roles=['prefill','decode'],
        collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,activity_copies=True,
        quiescent_witness=stage=='Profile',
        operator_correlation=False,graph_enclosures=False,activity_delay_seconds=30,activity_ROI_seconds=20,
        observer_scope='Kernel/copy-only20-second ROI after30 seconds. No operator hooks or CUDA API tracing. '
            'Require clean/profile/clean latency ratios within5% on the full trace and ROI-exposed request IDs. '
            'Kernel gaps are not guaranteed all-engine idle or decode-safe executable windows.')

def prepare():
    selection=json.loads((ROOT/'configs/c3_final_static_selection_v2.json').read_text());registry=policy_registry();original=qualified_registry()
    grid=json.loads((ROOT/'configs/c3_common_qps_grid_v2.json').read_text());knee=grid['reference_knee']
    loads=grid['QPS_grid']
    tests=[];lanes=[[],[],[],[]]
    cases=[]
    primary=json.loads((ROOT/'configs/c3_budget_priority_amendment_v4.json').read_text())['primary_confirmation_QPS']
    qps=json.loads((ROOT/'configs/c3_confirmation_knee_refinement_v3.json').read_text())['important_confirmation_QPS'][-1]
    for load in (primary,qps):cases.append(('NativeStrong',original['NativeStrong'],load))
    qps=json.loads((ROOT/'configs/c3_confirmation_knee_refinement_v3.json').read_text())['important_confirmation_QPS'][-1]
    for axis,value in selection['chosen'].items():
        if axis=='Deflection':continue
        cases.append((axis,value['item'],qps))
    for cap in sorted({v['item']['cap'] for v in selection['chosen'].values()}):
        key=f'NativeCommon_Cap{cap}_Credit0';cases.append(('NativeCommon'+str(cap),registry[key],qps))
    for load in sorted((v for v in loads if v not in (primary,qps)),key=lambda v:(v==min(loads),v)):
        cases.append(('NativeStrong',original['NativeStrong'],load))
    for index,(label,item,qps) in enumerate(cases):
        # Reserve the other secondary pair for controlled knob interventions
        # and the prospectively triggered primary-knee equal-output replay.
        lane=3;pair=[6,7];jobs=[]
        for stage in ('Clean1','Profile','Clean2'):
            name=f'V2_C3Activity_q{qps}_{label}_{stage}'
            flags=observation(stage)
            performance(item,name,'longbench_balanced400_v2.json',pair,qps,2026112301,observation=flags)
            jobs.append(name);lanes[lane].append(name)
        tests.append(dict(label=label,qps=qps,jobs=jobs,pair=pair))
    filename='c3_kernel_copy_cpc_v2.json';write(ROOT/'jobs'/filename,dict(time=time.time(),lanes=lanes,tests=tests,stop_on_failure=False,
        limits=[.95,1.05],output_work_difference_limit=.05,
        secondary_priority_SHA256=digest(ROOT/'configs/c3_secondary_resource_priority_v6.json'),
        scope='One independent clean/profile/clean triplet per method/load initially. '
            'The full-trace gate and matched ROI-exposed-request gate must both pass. '
            'Zero dropped/invalid CUPTI records are mandatory. A failed capture remains diagnostic. '
            'Do not use NVML utilization as kernel busy. The collector does not establish SM/HBM metrics or a safe insertion window.'))
    return filename

def early_native():
    from .common import dispatch_allowed
    assert dispatch_allowed()
    assert not (ROOT/'configs/c3_final_static_selection_v2.json').exists()
    assert (ROOT/'state/c3_continuous_iso8_output_bridge_complete_v2.json').exists()
    target=ROOT/'jobs/c3_native_activity_early_v10.json';assert not target.exists()
    item=qualified_registry()['NativeStrong']
    keys=json.loads((ROOT/'configs/c3_confirmation_knee_refinement_v3.json').read_text())['important_confirmation_QPS']
    tests=[];lanes=[[],[],[],[]]
    for qps in (keys[0],keys[-1]):
        jobs=[]
        for stage in ('Clean1','Profile','Clean2'):
            name=f'V2_C3Activity_q{qps}_NativeStrong_{stage}'
            performance(item,name,'longbench_balanced400_v2.json',[6,7],qps,2026112301,observation=observation(stage))
            jobs.append(name);lanes[3].append(name)
        tests.append(dict(label='NativeStrong',qps=qps,jobs=jobs,pair=[6,7]))
    write(ROOT/'configs/c3_native_activity_early_priority_v10.json',dict(time=time.time(),
        changed='Execution order only. Run the two already declared NativeStrong CPC triplets on the idle6/7 pair.',
        source_dataset_pair_trace_gate_changed=False,static_selection_uses_these_results=False,
        scope='The final full activity plan reuses these exact names,configs and receipts. '
            'Do not create another independent-start count by listing the same capture twice. '
            'Preserve the unchanged5% full-trace/ROI observer gate. Do not inspect Main/heldout timing before static selection freezes.'))
    write(target,dict(time=time.time(),lanes=lanes,tests=tests,stop_on_failure=False,
        scope='Early dispatch of the two unchanged NativeStrong CPC triplets. No candidate selection input.'))
    result=launch(target.name,allow_partial_budget=True)
    write(ROOT/'state/c3_native_activity_early_complete_v10.json',dict(time=time.time(),complete=result['complete']))

def analyze():
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.window_analysis import role_analysis,observer
    plan=json.loads((ROOT/'jobs/c3_kernel_copy_cpc_v2.json').read_text());results=[];flat=[]
    for test in plan['tests']:
        receipts=[ROOT/'raw'/name/'dispatcher_result.json' for name in test['jobs']]
        existing=[json.loads(path.read_text()) for path in receipts if path.exists()]
        if len(existing)!=3 or not all(r['complete'] for r in existing):
            status='FAILED' if any(not r['complete'] for r in existing) else 'NOT_RUN' if not existing else 'PARTIAL_OR_NOT_RUN'
            results.append(dict(test,evidence=status,completed_runs=sum(r['complete'] for r in existing),
                reason='A complete clean/profile/clean triplet is required. Do not label an unlaunched capture as a measured failure.'))
            continue
        try:
            roles=[role_analysis(test['jobs'][1],role) for role in ('prefill','decode')]
            gate=observer(test['qps'],roles,test['jobs'])
            metrics=[json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text()) for name in test['jobs']]
            clean_output=(metrics[0]['output_tokens']+metrics[2]['output_tokens'])/2
            work_ratio=metrics[1]['output_tokens']/clean_output
            gate['profile_output_over_mean_clean']=work_ratio;gate['output_work_gate_passed']=abs(work_ratio-1)<=.05
            from .c3_analysis import actual_quality
            gate['per_run_quality_passed']={name:actual_quality(name)['passed'] for name in test['jobs']}
            gate['passed'] &= gate['output_work_gate_passed'] and all(gate['per_run_quality_passed'].values())
            gate['status']='PASS' if gate['passed'] else 'DIAGNOSTIC'
            evidence='MEASURED_KERNEL_ACTIVITY' if gate['passed'] else 'DIAGNOSTIC'
            for role in roles:
                role['evidence']=evidence
                role['scope']='Kernel/copy activity uses the actual collector ROI. Host observations remain separate. '
                role['scope']+='A gap does not establish a ready partition,a free stream,KV credit or a future deadline. '
                role['scope']+='NIXL host overlap does not establish the cause of a gap.'
                if role['role']=='decode':
                    folder=ROOT/'raw'/test['jobs'][1]
                    roi=json.loads((folder/'activity_decode/actual_ROI.json').read_text())
                    left=roi['left']['host_lower_ns']/1e9;right=roi['right']['host_upper_ns']/1e9
                    snapshots=json.loads((folder/'scheduler_queues_decode.json').read_text())['rows']
                    samples=[r for r in snapshots if left<=r['time']<=right]
                    role['deadline_proxy_samples']=[dict(time=r['time'],
                        residual_budget_ms=1000*(r['earliest_worker_decode_deadline']-r['time']),
                        output_started_requests=r['output_started_requests'],active_decode=r['active_decode'])
                        for r in samples if r.get('earliest_worker_decode_deadline') is not None]
                    role['deadline_proxy_scope']='Observed scheduler snapshots only. Use the last worker emission plus34ms. '
                    role['deadline_proxy_scope']+='No future EOS or arrival is used. This proxy does not establish candidate readiness,stream availability or executable safety.'
                    role['no_known_active_deadline_snapshots']=sum(r.get('earliest_worker_decode_deadline') is None for r in samples)
                flat.append(dict(label=test['label'],QPS=test['qps'],role=role['role'],evidence=evidence,observer_passed=gate['passed'],
                    qualified_kernel_busy_fraction=role['kernel_busy_fraction'] if gate['passed'] else None,
                    diagnostic_raw_kernel_busy_fraction=role['kernel_busy_fraction'],ROI_seconds=role['ROI_seconds'],
                    interior_kernel_gap_ms=role['interior_kernel_gap_ms'],NIXL_host_enclosure_union_ms=role['NIXL_host_enclosure_union_ms'],
                    observed_copy_union_ms=role['observed_copy_union_ms'],queue=role['queue'],
                    decode_safe_executable_window='NOT_IDENTIFIED',SM_utilization='NOT_MEASURED',HBM_bandwidth='NOT_MEASURED'))
            results.append(dict(test,roles=roles,observer_gate=gate,evidence=evidence))
        except Exception as error:results.append(dict(test,evidence='FAILED',error=repr(error)))
    write(ROOT/'analysis/c3_kernel_copy_cpc_results_v2.json',dict(time=time.time(),results=results,
        scope='GPU-clock kernel/copy intervals are clipped to the actual measured ROI. Host timing and GPU timing remain separate. '
            'Qualified kernel busy requires the unchanged5% clean/profile/clean gate. One triplet is not a start-level serving CI. '
            'A physical kernel gap does not establish readiness,KV credit,stream availability or future deadline safety. '
            'NIXL API overlap is not proof that the API caused a gap. Online dynamic insertion remains NOT_RUN without a verified tail/window predictor.'))
    csv_save(ROOT/'figures/c3_kernel_busy_and_queue_v2.csv',flat);figures(results)
    print('C3_KERNEL_ACTIVITY_CPC_RESULTS',[(r['label'],r['qps'],r.get('observer_gate',{}).get('status',r['evidence'])) for r in results],flush=True)

def figures(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for test in results:
        if 'roles' not in test:continue
        records=[];gap_records=[]
        common_left=min(min(a for a,b in role['kernel_intervals_ns']+role['kernel_inactivity_intervals_ns']) for role in test['roles'])
        fig,axes=plt.subplots(2,1,figsize=(12,4.5),sharex=True)
        for ax,role in zip(axes,test['roles']):
            spans=role['kernel_intervals_ns'];allspans=spans+role['kernel_inactivity_intervals_ns']
            left=min(a for a,b in allspans);right=max(b for a,b in allspans)
            ax.broken_barh([((a-common_left)/1e9,(b-a)/1e9) for a,b in spans],(.2,.6),facecolors='#0072b2')
            for kind,values in (('kernel_union',spans),('kernel_inactivity',role['kernel_inactivity_intervals_ns'])):
                for a,b in values:
                    record=dict(role=role['role'],kind=kind,start_CUPTI_ns=a,end_CUPTI_ns=b,
                        start_seconds_from_common_ROI=(a-common_left)/1e9,duration_ms=(b-a)/1e6,
                        boundary_censored=a==left or b==right,evidence=test['evidence'])
                    records.append(record)
                    if kind=='kernel_inactivity' and not record['boundary_censored']:gap_records.append(record)
            ax.set(ylim=(0,1),yticks=[.5],yticklabels=[role['role']]);ax.grid(axis='x',alpha=.2)
        axes[-1].set_xlabel('Seconds from the earlier actual P/D collector ROI start')
        fig.suptitle(f'{test["label"]} | QPS{test["qps"]} | {test["evidence"]} | kernel intervals')
        fig.text(.02,.015,'One clean/profile/clean triplet. Capture is diagnostic unless the5% observer gate passes.\n'
            'Blank kernel intervals are not guaranteed all-engine idle. No validated executable-window or dynamic-speedup claim.',fontsize=8)
        fig.tight_layout(rect=(0,.1,1,.92));stem=f'c3_kernel_timeline_q{test["qps"]}_{test["label"]}'
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
        csv_save(ROOT/'figures'/f'{stem}.csv',records)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7.5,4.8))
        for role in test['roles']:
            values=sorted(r['duration_ms'] for r in gap_records if r['role']==role['role'])
            if values:ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=role['role'])
        ax.set(xscale='log',xlabel='Interior kernel inactivity interval (ms)',ylabel='Empirical CDF',
            title=f'{test["label"]} | QPS{test["qps"]} | {test["evidence"]}')
        ax.grid(alpha=.2);ax.legend()
        fig.text(.02,.015,'Boundary-censored gaps are excluded. One ROI per role. Kernel inactivity is not a decode-safe window.',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1));stem=f'c3_kernel_gap_CDF_q{test["qps"]}_{test["label"]}'
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
        csv_save(ROOT/'figures'/f'{stem}.csv',gap_records);plt.close(fig)
        deadline=next((r.get('deadline_proxy_samples',[]) for r in test['roles'] if r['role']=='decode'),[])
        if deadline:
            fig,ax=plt.subplots(figsize=(7.5,4.8));ax.hist([r['residual_budget_ms'] for r in deadline],bins=30)
            ax.set(xlabel='Observed worker-deadline residual (ms)',ylabel='Scheduler snapshots',
                title=f'{test["label"]} | QPS{test["qps"]} | {test["evidence"]}')
            fig.text(.02,.015,'Last worker emission+34ms proxy. Snapshot counts are not a time-weighted window distribution.\n'
                'No validated candidate-ready executable window or future-arrival protection follows from this histogram.',fontsize=8)
            fig.tight_layout(rect=(0,.09,1,1));stem=f'c3_worker_deadline_residual_q{test["qps"]}_{test["label"]}'
            for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
            csv_save(ROOT/'figures'/f'{stem}.csv',deadline);plt.close(fig)
    overview=[]
    for test in results:
        for role in test.get('roles',[]):
            overview.append(dict(label=test['label'],QPS=test['qps'],role=role['role'],
                kernel_busy_percent=100*role['kernel_busy_fraction'],evidence=test['evidence'],
                **role['queue']['time_weighted_mean']))
    for field,ylabel in (('kernel_busy_percent','GPU kernel busy (%)'),('waiting_total','Observed waiting requests'),
            ('decode_ready_waiting','Observed decode-ready waiting requests'),('remote_KV_waiting','Observed remote-KV waiting requests')):
        fig,ax=plt.subplots(figsize=(8,5));plotted=[]
        for label in sorted({r['label'] for r in overview}):
            for role,marker in (('prefill','o'),('decode','^')):
                values=[r for r in overview if r['label']==label and r['role']==role and field in r]
                for evidence in ('MEASURED_KERNEL_ACTIVITY','DIAGNOSTIC'):
                    rows=[r for r in values if r['evidence']==evidence]
                    if not rows:continue
                    ax.scatter([r['QPS'] for r in rows],[r[field] for r in rows],marker=marker,
                        label=f'{label}/{role}'+(' diagnostic' if evidence=='DIAGNOSTIC' else ''),
                        facecolors='none' if evidence=='DIAGNOSTIC' else None)
                    plotted+=rows
        ax.set(xlabel='Offered QPS (requests/s)',ylabel=ylabel,title='C3 short ROI observations')
        ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
        fig.text(.02,.015,'One clean/profile/clean triplet per point. Hollow markers fail the observer or quality gate.\n'
            'Queue states are host observations. The sampled ROI does not establish sustained capacity or a safe window.',fontsize=8)
        fig.tight_layout(rect=(0,.08,1,1));stem='c3_activity_'+field+'_vs_qps_v2'
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
        csv_save(ROOT/'figures'/f'{stem}.csv',plotted);plt.close(fig)

def run():
    wait_file(ROOT/'state/c3_continuous_iso8_output_bridge_complete_v2.json')
    # The bridge starts after geometry; selection can finish while it runs.
    while not (ROOT/'configs/c3_final_static_selection_v2.json').exists():
        from .common import dispatch_allowed
        assert dispatch_allowed();time.sleep(5)
    name=prepare();launch(name);analyze()
    write(ROOT/'state/c3_kernel_copy_cpc_complete_v2.json',dict(time=time.time(),complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','analyze'));args=p.parse_args();globals()[args.action]()
