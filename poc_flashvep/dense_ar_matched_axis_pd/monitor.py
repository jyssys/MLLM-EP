"""Record resumable progress and refresh CPU analysis between milestones."""
import argparse
import collections
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,QUIET_LOCK,resources,write,digest
from .recovery import acknowledged_failure,acknowledged_controller_stop

FLAGS=('discovery_complete','confirmation_complete','diagnostics_complete',
       'realized_shapes_complete','realized_shapes_all_pairs_complete','baseline_bridge_complete',
       'admission_confirmation_complete')


def read(path,default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return default


def snapshot():
    counts=collections.Counter()
    critical=[];optional=[];interrupted=[];conditional=[]
    prototype_failures=read(ROOT/'state/preserved_fixed_cohort_prototype_failures.json',{}).get('runs',{})
    quick_closeout=read(ROOT/'state/priority_actual_suffix_closeout.json',{})
    known_observer_startup_failures=set(quick_closeout.get('optional_observer_startup_failures',[]))
    for cfg_path in (ROOT/'configs').glob('*.json'):
        cfg=read(cfg_path,{})
        if not isinstance(cfg,dict) or cfg.get('name')!=cfg_path.stem:
            continue
        receipt=read(ROOT/'raw'/cfg['name']/'dispatcher_result.json')
        if receipt is None:
            continue
        if not receipt.get('complete'):
            acknowledged_prototype=prototype_failures.get(cfg['name'],{})
            prototype_matches=(cfg.get('fixed_cohort_prototype') and cfg.get('correctness_only')
                and acknowledged_prototype.get('eligible_for_performance') is False
                and acknowledged_prototype.get('dispatcher_SHA256')==digest(ROOT/'raw'/cfg['name']/'dispatcher_result.json')
                and acknowledged_prototype.get('configuration_SHA256')==digest(ROOT/'raw'/cfg['name']/'configuration.json'))
            observer_startup=(cfg.get('priority_projection_events') and cfg['name'] in known_observer_startup_failures
                and not (ROOT/'raw'/cfg['name']/'window.json').exists())
            if acknowledged_failure(cfg['name']):
                interrupted.append(cfg['name'])
            elif prototype_matches:
                conditional.append(cfg['name'])
            elif cfg.get('nsys_bounded_capture') or observer_startup:
                optional.append(cfg['name'])
            elif cfg.get('source_chunk_packing'):
                critical.append(cfg['name'])
            counts['failed_preserved']+=1
            continue
        if cfg.get('source_chunk_packing') and not cfg.get('nsys_bounded_capture'):
            for suffix in ('_state_gate.json','_output_gate.json'):
                gate=read(ROOT/'analysis'/(cfg['name']+suffix))
                if gate is not None and not gate.get('passed'):
                    critical.append(cfg['name'])
        if cfg.get('correctness_only'):
            counts['correctness_starts']+=1
        elif cfg.get('auxiliary_only'):
            counts['isolated_starts']+=1
        else:
            counts[cfg.get('matrix_stage','other')]+=1
    flags={key:read(ROOT/'state'/(key+'.json')) is not None for key in FLAGS}
    if (ROOT/'state/last2_supplement_design.json').exists() and not (ROOT/'state/last2_supplement_cancelled.json').exists():
        flags['last2_supplement_complete']=read(ROOT/'state/last2_supplement_complete.json') is not None
    flags['safe_observer_analysis_complete']=read(ROOT/'analysis/safe_observer_API_placement.json') is not None
    stopped=read(ROOT/'state/controller_stopped.json',{})
    design=read(ROOT/'state/design_freeze_v3_before_performance.json',{})
    if stopped.get('time',0)>design.get('frozen_epoch',float('inf')) and not acknowledged_controller_stop():
        critical.append('CONTROLLER_STOPPED: '+stopped.get('error','Unknown error'))
    stopped=read(ROOT/'state/last2_supplement_stopped.json',{})
    if stopped and not (ROOT/'state/last2_supplement_cancelled.json').exists():
        critical.append('LAST2_SUPPLEMENT_STOPPED: '+stopped.get('error','Unknown error'))
    family='qualification_reserved' if (ROOT/'state/design_freeze_v3_before_performance.json').exists() else 'qualification_chunked'
    qualification={str(n):bool((read(ROOT/'analysis'/f'{family}_N{n}.json',{}) or {}).get('passed')) for n in (8,32,128)}
    qualification['Output200_N8']=bool((read(ROOT/'analysis'/f'{family}_Output200_N8.json',{}) or {}).get('passed'))
    return dict(epoch=time.time(),counts=dict(counts),flags=flags,
        qualification=qualification,critical_failures=sorted(set(critical)),optional_profiler_failures=optional,
        preserved_external_interruptions=interrupted,
        preserved_unqualified_prototype_failures=conditional,
        GPU_lease=read(ROOT/'state/GPU_lease_current.json'),
        ready_for_final_analysis=all(flags.values()) and all(qualification.values()))


def refresh():
    with (ROOT/'logs/CPU_analysis_refresh.log').open('a') as log:
        for module in ('statistics','exports','batch_operator_models','observer_audit','root_analysis','figures'):
            result=subprocess.run([str(PYTHON),'-u','-m',
                'poc_flashvep.dense_ar_matched_axis_pd.'+module],cwd=REPO,
                stdout=log,stderr=subprocess.STDOUT,timeout=600)
            assert result.returncode==0,'CPU analysis failed: '+module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    refreshed=-1
    while True:
        value=snapshot()
        write(ROOT/'state/resume_progress_manifest.json',value)
        print(json.dumps(value),flush=True)
        if value['critical_failures']:
            raise RuntimeError('Preserved critical run failed. Diagnose before resuming: '+str(value['critical_failures']))
        completed=sum(value['counts'].get(k,0) for k in ('discovery','motivation_odd','confirmation','admission_control'))
        if completed//8!=refreshed or value['ready_for_final_analysis']:
            refresh();refreshed=completed//8
        if args.once:
            return
        if value['ready_for_final_analysis']:
            with QUIET_LOCK.open('a+') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                owner=resources()
                if not owner.processes():owner.restore(list(range(8)))
            write(ROOT/'state/final_analysis_ready.json',dict(time=time.time(),
                scope='The prespecified GPU matrices completed. This does not mark the campaign closed or publish a success verdict. Inspect results, figures and resource restoration before publication.'))
            return
        time.sleep(60)


if __name__=='__main__':
    main()
