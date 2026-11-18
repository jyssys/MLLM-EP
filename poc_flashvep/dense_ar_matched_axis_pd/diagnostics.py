"""Run equal-admission controls and bounded observer A/B/A checks."""
import argparse
import json
import time
from .common import ROOT,write
from .campaign import job,run_job
from .host_paths import analyze


def profiles():
    """Every axis/output has its own clean-before/profile/clean-after."""
    records=[]
    for cap in (16,200):
        for method in ('Layer','Token','Agg','Native'):
            values=[]
            names=[]
            failures=[]
            for mode in ('before','profile','after'):
                roles=['replica0','replica1'] if method=='Agg' else ['prefill','decode']
                cfg=job(method,12 if method in ('Layer','Token') else 0,36,cap,6.,2026111786,
                    stage='observer_'+mode,dataset='matched144.json',
                    nsys_bounded_capture=mode=='profile',nsys_roles=roles,
                    nsys_offset_s=5.,nsys_seconds=8.,clean_timing=mode!='profile',
                    runtime_observation=mode=='profile',runtime_observation_roles=roles)
                names.append(cfg['name'])
                try:
                    values.append(run_job(cfg))
                    analyze(cfg['name'])
                except BaseException as error:
                    values.append(None)
                    failures.append(dict(mode=mode,name=cfg['name'],error=repr(error)))
                    if mode!='profile':
                        raise
            tests={}
            for key in (() if failures else ('TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')):
                before,observed,after=[r[key] for r in values]
                control=(before+after)/2
                tests[key]=dict(before=before,profiled=observed,after=after,
                    profiled_over_control=observed/control,
                    controls_within_5pct=abs(after/before-1)<=.05,
                    passed=abs(observed/control-1)<=.05 and abs(after/before-1)<=.05)
            records.append(dict(method=method,cap=cap,names=names,tests=tests,failures=failures,
                passed=not failures and all(t['passed'] for t in tests.values()),
                evidence='TRACE_ANALYSIS',
                scope='Same seed, prompt pool and configuration. These are observer controls, not independent performance confirmation. Failed 5% gates restrict all associated online kernel claims to Diagnostic.'))
            write(ROOT/'analysis/online_profiler_observer_gates.json',records)


def admission():
    # This is a controlled ablation, not a replacement for the cap8 primary.
    manifest=[]
    for cap in (16,200):
        for layers in (4,8,12):
            for limit in (1,8):
                for method in ('Layer','Token'):
                    cfg=job(method,layers,36,cap,6.,2026111787,
                        stage='admission_control',dataset='matched144.json',admission=limit)
                    manifest.append(cfg['name'])
                    run_job(cfg);analyze(cfg['name'])
    write(ROOT/'analysis/admission_control_manifest.json',dict(names=manifest,
        scope='Same B8192/S1024. Change the common admission cap on both axes. N36 exactly matched rows. Cap1 removes the between-axis cohort-size difference. This ablation does not replace the cap8 primary.'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profiles',action='store_true')
    parser.add_argument('--admission',action='store_true')
    parser.add_argument('--wait-probes',action='store_true')
    args=parser.parse_args()
    if args.wait_probes:
        while not (ROOT/'state/probe_complete.json').exists():
            time.sleep(15)
    if args.profiles:
        profiles()
    if args.admission:
        admission()
    write(ROOT/'state/diagnostics_complete.json',dict(time=time.time(),
        profiles=args.profiles,admission=args.admission))


if __name__=='__main__':
    main()
