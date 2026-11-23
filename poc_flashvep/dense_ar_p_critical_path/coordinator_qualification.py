"""Qualify the common coordinator change before primary serving tests."""
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,bind,dispatch_allowed
from .bootstrap import baseline
from .capsule import build as retirement_source
from .coordinator_capsule import build

PLAN='coordinator_hygiene_qualification_r2.json'


def prepare():
    runner=bind();items=[]
    labels=('NativeStrong','NativeCommon2','StockToken','Layer','Hybrid','Layer8','HybridIso8Approx','HybridGCControl')
    for label in labels:
        axis='Layer' if label=='Layer8' else 'Hybrid' if label in ('HybridIso8Approx','HybridGCControl') else label
        if label in ('Layer8','HybridIso8Approx'):
            cfg=json.loads((ROOT/'configs'/f'PC_Balance_{label}_Natural400.json').read_text())
            parent=ROOT/'references/source_capsules'/cfg['source_capsule']
        else:
            cfg=baseline(axis,'PC_CoordQualTemplate_'+label,[2,3],save_cfg=False)
            parent=retirement_source(cfg['source_capsule'].removeprefix('pc_'))
        source=build(parent.name,'r2');defer_gc=label!='HybridGCControl'
        cfg.update(gpus=[2,3],source_capsule=source.name,source_capsule_SHA256=digest(source),
            P_event_timing=False,P_critical_trace=True,source_credit_release_on_retirement=False,
            coordinator_trace=False,buffer_coordinator_evidence=True,defer_coordinator_cyclic_gc=defer_gc)
        teacher=strict=None
        if axis in ('Layer','Hybrid','StockToken') and label!='HybridGCControl':
            base_name='PC_Balance_'+label+'_Strict8' if label in ('Layer8','HybridIso8Approx') else f'PC_Q_{axis}_Original_Strict8_ArrivalFix2'
            strict_base=json.loads((ROOT/'configs'/(base_name+'.json')).read_text())
            names=[]
            for enabled in (False,True):
                name=f'PC_CoordQual_{label}_{"DeferredGC" if enabled else "NormalGC"}_Strict8'
                value=dict(strict_base);value.update(name=name,gpus=[2,3],source_capsule=source.name,
                    source_capsule_SHA256=digest(source),P_event_timing=False,P_critical_trace=True,
                    source_credit_release_on_retirement=False,coordinator_trace=False,
                    buffer_coordinator_evidence=True,defer_coordinator_cyclic_gc=enabled)
                value.pop('qualification_gate_v2',None);runner.save(value);names.append(name)
            teacher,strict=names
        jobs=[]
        for count,data in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            name=f'PC_CoordQual_{label}_Natural{count}';value=dict(cfg)
            value.update(name=name,dataset=data,dataset_SHA256=digest(ROOT/'data'/data),count=count,
                correctness_only=True,axis_state_capture=True,axis_capture_hidden_last_only=True,
                axis_import_hash_after_model=True,gpu_state_digest=True,verify_GPU_digest_with_CPU=False,
                ignore_eos=False,qps=3.2,seed=2026112370,
                arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(2026112370).exponential(1/3.2,count-1))].tolist())
            for key in ('replay_output_lengths','diagnostic_controlled_output_replay','qualification_gate_v2'):value.pop(key,None)
            if axis=='StockToken':
                from ..dense_ar_output_aware_v2.work_contract import regenerate
                regenerate(value,json.loads((ROOT/'data'/data).read_text()))
            runner.save(value);jobs.append(name)
        items.append(dict(label=label,arm='NativeCommon' if label=='NativeCommon2' else axis,lane=1,pair=[2,3],
            teacher=teacher,strict=strict,jobs=jobs,gate='pc_coord_'+label.lower()+'_composition_r2.json',
            signature=runner.signature(value),source_capsule_SHA256=digest(source),
            strict_scope='Same-source NormalGC/DeferredGC exact state differential. Original-weight teacher evidence remains inherited. '
                'Native has no new strict8. Its fresh Natural32/400 state,coverage,lifetime and quality are tested.'))
    path=ROOT/'configs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),items=items,state_prefix='pc_coordinator_hygiene',source_capsule_SHA256=None,
        scope='Common coordinator evidence-write and cyclic-GC timing change. Worker model/transport bytes stay unchanged from the R1 parent. '
            'Reference counting remains enabled. Restore the prior coordinator GC state after measurement or in finally. '
            'All eight compositions get fresh Natural32/400 lifetime,coverage and frozen task-quality gates. '
            'Five axis compositions also get fresh exact8 False/True differential tests on GPU2/3. '
            'No primary performance claim before these gates. No new active-decode insertion or timestamp correction.'))
    log=ROOT/'logs/coordinator_hygiene_qualification_r2.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.coordinator_qualification','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/coordinator_hygiene_qualification_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    bind();plan=json.loads((ROOT/'jobs/coordinator_GC_diagnostic_v1.json').read_text())
    while not all((ROOT/'raw'/name/'dispatcher_result.json').exists() for case in plan['cases'] for name in case['jobs']):
        assert dispatch_allowed();time.sleep(15)
    for case in plan['cases']:
        for name in case['jobs']:
            assert json.loads((ROOT/'raw'/name/'dispatcher_result.json').read_text())['complete'],name
    from ..dense_ar_output_aware_v2.c3_family import run as qualify
    qualify(1,PLAN)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)
