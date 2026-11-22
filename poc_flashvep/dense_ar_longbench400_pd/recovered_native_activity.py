"""Retest observer overhead after the independently qualified READ policy."""
import argparse
import json
import time
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .next_jobs import plan
from .run import execute
from .window_analysis import observer,role_analysis
from .confirmation_analysis import ratio


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('lane',type=int,choices=(0,1))
    args=parser.parse_args()
    qps,first=((1.8,0),(2.8,4))[args.lane]
    capsule,sha=freeze('recovered_native_activity_3triplets_v1_20261121')
    triplets=[]
    for start,seed in enumerate((2026112471,2026112472,2026112473)):
        names=[]
        for label in ('Clean1','KernelCopiesROI','Clean2'):
            profile=label=='KernelCopiesROI'
            names.append(create(f'RecoveredNative400_q{qps}_s{start}_{label}_v1','Native',
                'longbench_balanced400_v1.json',pair=(first,first+1),qps=qps,seed=seed,
                execution_family='INSTALLED_DEFAULT_COMPILE',continuous_source_credit_limit=0,
                d_compute_throttle=True,defer_D_remote_setup=True,quiescent_witness=True,
                source_capsule=capsule,source_capsule_SHA256=sha,
                kernel_activity=profile,kernel_activity_roles=['prefill','decode'],
                collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,
                activity_copies=True,operator_correlation=False,graph_enclosures=False,
                activity_delay_seconds=30,activity_ROI_seconds=10,
                observer_scope='Three new independent Clean/Profile/Clean triplets. The qualified common READ policy is fixed. '
                    'Both GPU roles in one execution. GPU kernel/copy activity only. '
                    'Full-trace and ROI-exposed request metrics must stay within5percent. '
                    'CUPTI kernel/copy inactivity is not all-engine physical idle or a predicted safe window.'))
        triplets.append(names)
    contract=dict(triplets=triplets,QPS=qps,source_SHA256=sha,limits=[.95,1.05],
        scope='A new observer test after the policy intervention. Preserve all earlier collector failures. '
            'No relaxed criterion. The prospective future window and ready partial-state requirements remain separate.',
        evidence='FROZEN_OBSERVER_DESIGN')
    write(ROOT/'configs'/f'recovered_native_activity_q{qps}_v1.json',contract)
    results=[]
    for names in triplets:
        for name in names:
            assert execute(name),name
        roles=[role_analysis(names[1],role) for role in ('prefill','decode')]
        gate=observer(qps,roles,names)
        results.append(dict(names=names,observer=gate,roles=roles))
        write(ROOT/'analysis'/f'recovered_native_activity_q{qps}_v1.json',
            dict(results=results,evidence='DIAGNOSTIC_PENDING_THREE_TRIPLETS'))
        print('RECOVERED_NATIVE_OBSERVER',qps,len(results),gate['status'],flush=True)
    ci={}
    for scope in ('full_trace_profile_over_mean_clean','matched_exposed_requests_profile_over_mean_clean'):
        ci[scope]={field:ratio([r['observer'][scope][field] for r in results],[1.]*3)
            for field in results[0]['observer'][scope]}
    passed=all(v['CI95'][0]>=.95 and v['CI95'][1]<=1.05 for values in ci.values() for v in values.values())
    # All per-triplet counters and ROI anchors already passed role_analysis.
    write(ROOT/'analysis'/f'recovered_native_activity_q{qps}_v1.json',dict(
        time=time.time(),results=results,observer_ratio_CIs=ci,passed=passed,
        evidence='MEASURED_KERNEL_ACTIVITY' if passed else 'DIAGNOSTIC',
        scope='Three independent paired triplets. Pointwise unadjusted start-level empirical CIs. '
            'Do not report failed-observer kernel activity as clean primary evidence. '
            'No ready partition was inserted. No decode-safe executable future window is established.'))
    print('RECOVERED_NATIVE_OBSERVER_FINAL',qps,passed,flush=True)


if __name__=='__main__':
    main()
