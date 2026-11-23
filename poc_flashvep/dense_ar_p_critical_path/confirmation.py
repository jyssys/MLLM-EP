"""Confirm one safe credit intervention against unchanged strong references."""
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind,preserve,resources,dispatch_allowed
from .bootstrap import baseline
from .capsule import build

ARMS=('NativeStrong','NativeCommon2','StockToken','OriginalLayer','OptimizedLayer','OriginalHybrid','OptimizedHybrid')


def prepare():
    runner=bind();blocks=[];jobs=[]
    for mode in ('EqualReplay','Natural'):
        for start,seed in enumerate((2026112321,2026112322,2026112323)):
            names={}
            for arm in ARMS:
                label=arm.removeprefix('Original').removeprefix('Optimized')
                name=f'PC_C3_{mode}_s{start}_{arm}'
                cfg=baseline(label,name,[4,5],seed=seed,natural=mode=='Natural',save_cfg=False)
                if label in ('Layer','Hybrid'):
                    source=build(cfg['source_capsule'].removeprefix('pc_'))
                    enabled=arm.startswith('Optimized');policy='retirement' if enabled else 'control'
                    cfg.update(source_capsule=source.name,source_capsule_SHA256=digest(source),
                        qualification_gate_v2=f'pc_{label.lower()}_{policy}_composition.json',
                        P_critical_trace=True,P_event_timing=False,source_credit_release_on_retirement=enabled)
                    qcfg=json.loads((ROOT/'configs'/f'PC_Q_{label}_{"Retirement" if enabled else "Control"}_Natural400.json').read_text())
                    assert runner.signature(cfg)==runner.signature(qcfg),(arm,'Different tested composition')
                cfg.update(confirmation=True,request_holdout_scope='Frozen192 heldout requests. New paired arrival seeds. '
                    'Each arm uses an independent server start. Do not combine the R0 reproduction or CPC starts with this N3.')
                runner.save(cfg);names[arm]=name
            order=list(ARMS)
            if start==1:order.reverse()
            elif start==2:order=order[2:]+order[:2]
            if mode=='Natural':order=order[3:]+order[:3]
            jobs.extend(names[arm] for arm in order)
            blocks.append(dict(mode=mode,start=start,seed=seed,qps=2.0,pair=[4,5],arms=names,order=order))
    path=ROOT/'jobs/c3_credit_retirement_confirmation_r1.json';assert not path.exists()
    write(path,dict(time=time.time(),blocks=blocks,ordered_jobs=jobs,arms=ARMS,
        target_P_critical_path_reduction_fraction=.05,paired_starts=3,
        scope='Same capsule and credit/admission values for Original/Optimized. Only safe source-retirement release changes. '
            'Original here is the newly qualified same-source false-switch control. The root-only R0 reproduction remains separate. '
            'Strong Stock Token retains the original compiled source and unbounded coordinator window. '
            'No future output/decode window prediction and no new active-decode insertion. All guards and SLO remain frozen.'))
    log=ROOT/'logs/c3_credit_retirement_confirmation_r1.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.confirmation','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/c3_credit_confirmation_controller.json',dict(time=time.time(),pid=child.pid,log=str(log.relative_to(ROOT))))
    return child.pid


def ready_gate(cfg,owner,pair):
    path=ROOT/'state'/cfg['qualification_gate_v2'];restored=False
    while not path.exists():
        if not dispatch_allowed():return 'NOT_RUN_TIME_BUDGET'
        label=cfg.get('axis_kind');lane=0 if label=='Layer' else 1
        failure=ROOT/'state'/f'retirement_qualification_r2_lane{lane}_failure.json'
        if failure.exists():return 'NOT_RUN_CORRECTNESS_FAILURE'
        if not restored:owner.restore(pair);restored=True
        time.sleep(10)
    proof=json.loads(path.read_text())
    return 'READY' if proof.get('passed') else 'NOT_RUN_CORRECTNESS_GATE'


def run():
    runner=bind();pair=[4,5];owner=resources();results=[]
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from ..dense_ar_longbench400_pd.service_accounting import analyze as service
    from .quality import compare as quality
    plan=json.loads((ROOT/'jobs/c3_credit_retirement_confirmation_r1.json').read_text())
    with (ROOT/'state/pair_4_5.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for name in plan['ordered_jobs']:
                cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());status=ready_gate(cfg,owner,pair)
                result=dict(name=name,status=status)
                if status=='READY':
                    try:
                        owner.stop(pair);ok=runner.execute(name)
                        result['status']='COMPLETE' if ok else 'FAILED'
                        if ok:
                            metric=summarize(name);path=service(name)
                            result['metrics']={k:metric[k] for k in ('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms',
                                'output_tokens','SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second',
                                'decode_SLO_attainment','max_ITL_ms') if k in metric}
                            result['P_arrival_to_ready_mean_ms']=sum(path['means'][k] for k in
                                ('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms','P_prepare_to_ready_ms'))
                            if not cfg['ignore_eos']:
                                result['quality_passed']=quality(name)['passed']
                                if not result['quality_passed']:result['status']='INVALID_QUALITY'
                    except Exception as error:result.update(status='FAILED',error=repr(error))
                results.append(result)
                write(ROOT/'state/c3_credit_confirmation_progress.json',dict(time=time.time(),results=results))
                print('PC_C3_CONFIRMATION',json.dumps(result),flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state/c3_credit_confirmation_complete.json',dict(time=time.time(),results=results,
        complete=len(results)==len(plan['ordered_jobs']),all_executed_passed=all(v['status']=='COMPLETE' for v in results)))


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)
