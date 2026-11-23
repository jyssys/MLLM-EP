"""Freeze the discovery screen before creating independent heldout jobs."""
import fcntl
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,bind,resources,dispatch_allowed,preserve

PLAN='p_geometry_confirmation_v1.json'


def prepare_after_discovery():
    runner=bind()
    from .balance_analysis import analyze
    from ..dense_ar_output_aware_v2.work_contract import regenerate
    discovery=json.loads((ROOT/'jobs/p_balance_discovery208_v1.json').read_text())
    while not all((ROOT/'raw'/r['name']/'dispatcher_result.json').exists() for r in discovery['arms']):
        assert dispatch_allowed(),'Discovery did not finish before the dispatch deadline'
        time.sleep(15)
    frozen=ROOT/'state/p_service_balance_selection_frozen.json'
    if not frozen.exists():analyze(freeze=True)
    selection=json.loads(frozen.read_text());selected=selection['selected']
    if not selected:
        write(ROOT/'state/p_geometry_confirmation_not_run.json',dict(time=time.time(),status='NOT_RUN',
            reason='Neither predefined geometry candidate passed the frozen Discovery208 screen.'))
        return None
    mapping={'NativeStrong':'NativeStrong','NativeCommon2':'NativeCommon2','StockToken':'StockToken'}
    for candidate in selected:
        axis='Layer' if candidate=='Layer8' else 'Hybrid'
        mapping['Original'+axis]='Original'+axis;mapping['Optimized'+axis]=candidate
    data_name='balanced400_v2_heldout.json';data=json.loads((ROOT/'data'/data_name).read_text());count=len(data)
    reference_cfg=json.loads((ROOT/'configs/PC_C3_EqualReplay_s0_OriginalLayer.json').read_text())
    lengths=reference_cfg['replay_output_lengths'];assert len(lengths)==count
    blocks=[];ordered=[]
    for mode in ('EqualReplay','Natural'):
        for index,seed in enumerate((2026112361,2026112362,2026112363)):
            names={}
            for label,original in mapping.items():
                name=f'PC_Geometry_{mode}_s{index}_{label}'
                cfg=json.loads((ROOT/'configs'/('PC_BalanceDiscovery208_'+original+'.json')).read_text())
                cfg.update(name=name,gpus=[6,7],dataset=data_name,dataset_SHA256=digest(ROOT/'data'/data_name),
                    count=count,seed=seed,qps=2.,confirmation=True,discovery_only=False,
                    arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(.5,count-1))].tolist(),
                    ignore_eos=mode=='EqualReplay',request_holdout_scope='Frozen192 geometry holdout. '
                        'Selection used only Discovery208. New paired arrival seeds. Independent server starts. '
                        'Do not pool this family with the credit-retirement confirmation.')
                if mode=='EqualReplay':cfg.update(replay_output_lengths=lengths,diagnostic_controlled_output_replay=True)
                else:cfg.pop('replay_output_lengths',None);cfg.pop('diagnostic_controlled_output_replay',None)
                regenerate(cfg,data);runner.save(cfg);names[label]=name
            order=list(mapping)
            if index==1:order.reverse()
            elif index==2:order=order[2:]+order[:2]
            if mode=='Natural':order=order[2:]+order[:2]
            ordered.extend(names[label] for label in order)
            blocks.append(dict(mode=mode,start=index,seed=seed,qps=2.,pair=[6,7],arms=names,order=order))
    target=ROOT/'jobs'/PLAN;assert not target.exists()
    write(target,dict(time=time.time(),blocks=blocks,arms=list(mapping),ordered_jobs=ordered,
        selected_before_confirmation=mapping,selection_SHA256=digest(frozen),
        analysis_scope='Geometry confirmation only. OptimizedLayer means Last8,not a same-work compiler change. '
            'OptimizedHybrid means Last2 with suffix3/17. Credit limits16/8,D cap2 and numerical paths remain unchanged. '
            'Strong Stock WorkCut8 keeps its original compiled path,cap0 and credit0. Both Native references remain. '
            'All independent starts and failures remain visible. Natural output work can vary under BI0. '
            'No dynamic safe window is claimed. N3 paired95% intervals use starts,not requests or tokens.'))
    return target


def run():
    runner=bind();path=prepare_after_discovery()
    if path is None:return
    plan=json.loads(path.read_text());pair=[6,7];owner=resources();results=[]
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from .quality import compare
    with (ROOT/'state/pair_6_7.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for name in plan['ordered_jobs']:
                row=dict(name=name,status='NOT_RUN')
                if dispatch_allowed():
                    try:
                        owner.stop(pair);ok=runner.execute(name);row['status']='COMPLETE' if ok else 'FAILED'
                        if ok:
                            row['metrics']=summarize(name)
                            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                            if not cfg['ignore_eos']:
                                row['quality_passed']=compare(name)['passed']
                                if not row['quality_passed']:row['status']='INVALID_QUALITY'
                    except Exception as error:row.update(status='FAILED',error=repr(error))
                results.append(row)
                write(ROOT/'state/p_geometry_confirmation_progress.json',dict(time=time.time(),results=results))
                print('GEOMETRY_CONFIRMATION',name,row['status'],flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state/p_geometry_confirmation_complete.json',dict(time=time.time(),results=results,
        complete=len(results)==len(plan['ordered_jobs'])))


def launch():
    bind();log=ROOT/'logs/p_geometry_confirmation_controller.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.geometry_confirmation','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/p_geometry_confirmation_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


if __name__=='__main__':
    import sys
    if sys.argv[1]=='launch':print(launch(),flush=True)
    elif sys.argv[1]=='analyze':
        from .confirmation_analysis import analyze
        analyze(PLAN,'p_geometry_confirmation')
    else:run()
