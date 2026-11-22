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

def prepare():
    selection=json.loads((ROOT/'configs/c3_final_static_selection_v2.json').read_text());registry=policy_registry();original=qualified_registry()
    grid=json.loads((ROOT/'configs/c3_common_qps_grid_v2.json').read_text());knee=grid['reference_knee']
    loads=sorted({min(grid['QPS_grid'],key=lambda q:abs(q-knee)),min(grid['QPS_grid'],key=lambda q:abs(q-1.2*knee))})
    tests=[];lanes=[[],[],[],[]]
    cases=[]
    for qps in loads:cases.append(('NativeStrong',original['NativeStrong'],qps))
    qps=loads[-1]
    for axis,value in selection['chosen'].items():
        if axis=='Deflection':continue
        cases.append((axis,value['item'],qps))
    for cap in sorted({v['item']['cap'] for v in selection['chosen'].values()}):
        key=f'NativeCommon_Cap{cap}_Credit0';cases.append(('NativeCommon'+str(cap),registry[key],qps))
    for index,(label,item,qps) in enumerate(cases):
        lane=2+index%2;pair=[2*lane,2*lane+1];jobs=[]
        for stage in ('Clean1','Profile','Clean2'):
            name=f'V2_C3Activity_q{qps}_{label}_{stage}'
            flags=dict(kernel_activity=stage=='Profile',kernel_activity_roles=['prefill','decode'],
                collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,activity_copies=True,
                operator_correlation=False,graph_enclosures=False,activity_delay_seconds=30,activity_ROI_seconds=20,
                observer_scope='Kernel/copy-only20-second ROI after30 seconds. No operator hooks or CUDA API tracing. '
                    'Require clean/profile/clean latency ratios within5% on the full trace and ROI-exposed request IDs. '
                    'Kernel gaps are not guaranteed all-engine idle or decode-safe executable windows.')
            performance(item,name,'longbench_balanced400_v2.json',pair,qps,2026112301,observation=flags)
            jobs.append(name);lanes[lane].append(name)
        tests.append(dict(label=label,qps=qps,jobs=jobs,pair=pair))
    filename='c3_kernel_copy_cpc_v2.json';write(ROOT/'jobs'/filename,dict(time=time.time(),lanes=lanes,tests=tests,stop_on_failure=False,
        limits=[.95,1.05],output_work_difference_limit=.05,
        scope='One independent clean/profile/clean triplet per method/load initially. '
            'The full-trace gate and matched ROI-exposed-request gate must both pass. '
            'Zero dropped/invalid CUPTI records are mandatory. A failed capture remains diagnostic. '
            'Do not use NVML utilization as kernel busy. The collector does not establish SM/HBM metrics or a safe insertion window.'))
    return filename

def analyze():
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.window_analysis import role_analysis,observer
    plan=json.loads((ROOT/'jobs/c3_kernel_copy_cpc_v2.json').read_text());results=[];flat=[]
    for test in plan['tests']:
        try:
            roles=[role_analysis(test['jobs'][1],role) for role in ('prefill','decode')]
            gate=observer(test['qps'],roles,test['jobs'])
            metrics=[json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text()) for name in test['jobs']]
            clean_output=(metrics[0]['output_tokens']+metrics[2]['output_tokens'])/2
            work_ratio=metrics[1]['output_tokens']/clean_output
            gate['profile_output_over_mean_clean']=work_ratio;gate['output_work_gate_passed']=abs(work_ratio-1)<=.05
            gate['passed'] &= gate['output_work_gate_passed'];gate['status']='PASS' if gate['passed'] else 'DIAGNOSTIC'
            evidence='MEASURED_KERNEL_ACTIVITY' if gate['passed'] else 'DIAGNOSTIC'
            for role in roles:
                role['evidence']=evidence
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
        fig,axes=plt.subplots(2,1,figsize=(12,4.5),sharex=True)
        for ax,role in zip(axes,test['roles']):
            spans=role['kernel_intervals_ns'];allspans=spans+role['kernel_inactivity_intervals_ns'];left=min(a for a,b in allspans)
            ax.broken_barh([((a-left)/1e9,(b-a)/1e9) for a,b in spans],(.2,.6),facecolors='#0072b2')
            ax.set(ylim=(0,1),yticks=[.5],yticklabels=[role['role']]);ax.grid(axis='x',alpha=.2)
        axes[-1].set_xlabel('Seconds from actual collector ROI start')
        fig.suptitle(f'{test["label"]} | QPS{test["qps"]} | {test["evidence"]} | kernel intervals')
        fig.text(.02,.015,'One clean/profile/clean triplet. Capture is diagnostic unless the5% observer gate passes.\n'
            'Blank kernel intervals are not guaranteed all-engine idle. No validated executable-window or dynamic-speedup claim.',fontsize=8)
        fig.tight_layout(rect=(0,.1,1,.92));stem=f'c3_kernel_timeline_q{test["qps"]}_{test["label"]}'
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
        plt.close(fig)

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
