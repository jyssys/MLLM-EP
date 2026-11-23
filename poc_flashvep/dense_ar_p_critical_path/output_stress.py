"""Separate output length from input and arrival in a limited C3 stress control."""
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,bind,dispatch_allowed
from .bootstrap import baseline

PLAN='p_output_stress_c3_v1.json'


def prepare():
    runner=bind();blocks=[];ordered=[]
    mapping={'NativeStrong':'NativeStrong','NativeCommon2':'NativeCommon2','StockToken':'StockToken',
        'OriginalLayer':'OriginalLayer','OptimizedLayer':'Layer8','OriginalHybrid':'OriginalHybrid',
        'OptimizedHybrid':'HybridIso8Approx'}
    for output in (16,64,128,192):
        for index,seed in enumerate((2026112381,2026112382,2026112383) if output==192 else (2026112381,)):
            names={}
            for label,parent in mapping.items():
                name=f'PC_OutputStress_Fixed{output}_s{index}_{label}'
                if parent in ('Layer8','HybridIso8Approx'):
                    cfg=json.loads((ROOT/'configs'/f'PC_Balance_{parent}_Natural400.json').read_text())
                    cfg.update(qualification_gate_v2='pc_balance_'+parent.lower()+'_composition.json')
                elif parent.startswith('Original'):
                    cfg=json.loads((ROOT/'configs'/f'PC_C3_EqualReplay_s0_{parent}.json').read_text())
                else:cfg=baseline(parent,name,[4,5],save_cfg=False)
                dataset='balanced400_v2_heldout.json';count=192
                cfg.update(name=name,gpus=[4,5],dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=count,
                    ignore_eos=True,output_cap=output,replay_output_lengths=[output]*count,diagnostic_controlled_output_replay=True,
                    correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
                    qps=2.,seed=seed,arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(.5,count-1))].tolist(),
                    confirmation=output==192,discovery_only=False,
                    scope='Controlled fixed-output stress test. Original prompts stay frozen. This is not Natural EOS. '
                        'Output16/64/128 each have one start. Output192 has three predeclared starts. '
                        'Do not select a favorable output or static partition after this stress test.')
                if cfg.get('axis_kind')=='Token':
                    from ..dense_ar_output_aware_v2.work_contract import regenerate
                    regenerate(cfg,json.loads((ROOT/'data'/dataset).read_text()))
                runner.save(cfg);names[label]=name
            order=list(mapping)
            if index==1:order.reverse()
            elif index==2:order=order[2:]+order[:2]
            ordered.extend(names[label] for label in order)
            blocks.append(dict(mode='Fixed'+str(output),start=index,seed=seed,qps=2.,pair=[4,5],arms=names,order=order))
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),blocks=blocks,arms=list(mapping),ordered_jobs=ordered,
        lanes=[[],[],ordered,[]],stop_on_failure=True,
        analysis_scope='Controlled Output16/64/128/192 stress on the unchanged heldout192 prompts. '
            'The same arrival trace is used across outputs for start0. Only Output192 has predeclared N3. '
            'OptimizedLayer denotes the Last8 geometry control. OptimizedHybrid denotes Last2/suffix3/17. '
            'Both use false early-credit flags. Source credit16/8 and D cap2 remain fixed. '
            'Strong Stock remains WorkCut8,compiled,credit0,cap0. These are not iso-work or common-policy axis-only comparisons. '
            'No output bucket is substituted for Natural EOS. Every fixed-output sample emits the specified token count. '
            'No active-decode insertion or predicted safe window is added. Single-start gates are unconfirmed.'))
    log=ROOT/'logs/p_output_stress_controller.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.output_stress','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/p_output_stress_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    runner=bind();parent=json.loads((ROOT/'jobs/c3_credit_retirement_confirmation_r1.json').read_text())
    while not all((ROOT/'raw'/name/'dispatcher_result.json').exists() for name in parent['ordered_jobs']):
        if not dispatch_allowed():
            write(ROOT/'state/p_output_stress_not_run.json',dict(time=time.time(),status='NOT_RUN',reason='Dispatch budget expired while awaiting primary N3.'))
            return
        time.sleep(15)
    for name in parent['ordered_jobs']:
        if not json.loads((ROOT/'raw'/name/'dispatcher_result.json').read_text())['complete']:
            write(ROOT/'state/p_output_stress_not_run.json',dict(time=time.time(),status='NOT_RUN',reason='Primary GPU run failed. Diagnose it before the optional stress campaign.',name=name))
            return
    runner.lane(PLAN,2)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':print(prepare(),flush=True)
    elif sys.argv[1]=='analyze':
        from .confirmation_analysis import analyze
        analyze(PLAN,'p_output_stress')
    else:run()
