"""Qualify two cost-motivated geometry controls,not another full cut sweep."""
import json
import subprocess
import time
import numpy as np
from .common import ROOT,PRIOR,REPO,PYTHON,write,digest,copy_data,bind
from .bootstrap import baseline,reroot
from .capsule import build


def prepare():
    runner=bind();items=[]
    cases=(dict(label='Layer8',axis='Layer',cut=8,lane=3),
        dict(label='HybridIso8Approx',axis='Hybrid',cut=2,fraction=3/17,lane=1))
    for case in cases:
        label,axis,lane=case['label'],case['axis'],case['lane'];pair=[2*lane,2*lane+1]
        cfg=baseline(axis,'PC_BalanceTemplate_'+label,pair,save_cfg=False)
        source=build(cfg['source_capsule'].removeprefix('pc_'))
        cfg.update(axis_cut=case['cut'],source_capsule=source.name,source_capsule_SHA256=digest(source),
            P_critical_trace=True,P_event_timing=False,source_credit_release_on_retirement=False)
        if axis=='Hybrid':cfg['hybrid_suffix_fraction']=case['fraction']
        if axis=='Layer':teacher_old='V2_DecodeScope_Layer_OriginalTeacher8'
        else:teacher_old='V2_C3Policy_Hybrid_FusedShort_DCompiledInterval_Cap2_Credit8_LocalThreeWaveTeacher8_ContextFixV15'
        teacher_cfg=json.loads((PRIOR/'configs'/(teacher_old+'.json')).read_text())
        tsource=reroot(teacher_cfg['source_capsule']);dataset=teacher_cfg['dataset'];copy_data(dataset)
        teacher='PC_Balance_'+label+'_IndependentTeacher8'
        teacher_cfg.update(name=teacher,gpus=pair,source_capsule=tsource.name,source_capsule_SHA256=digest(tsource),arrival_offsets=[0.]*8)
        if axis=='Hybrid':teacher_cfg['local_hybrid_teacher_fraction']=case['fraction']
        runner.save(teacher_cfg)
        strict='PC_Balance_'+label+'_Strict8';qcfg=dict(cfg)
        qcfg.update(name=strict,dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=8,
            output_cap=16,ignore_eos=True,serial=True,controlled_pipeline=True,continuous_source_credit_limit=0,
            correctness_only=True,axis_state_capture=True,axis_capture_hidden_last_only=False,
            axis_import_hash_after_model=False,gpu_state_digest=False,verify_GPU_digest_with_CPU=False,arrival_offsets=[0.]*8)
        for key in ('replay_output_lengths','diagnostic_controlled_output_replay','qualification_gate_v2'):qcfg.pop(key,None)
        runner.save(qcfg);jobs=[]
        for count,dataset in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            value=dict(cfg);name='PC_Balance_'+label+f'_Natural{count}'
            value.update(name=name,dataset=dataset,dataset_SHA256=digest(copy_data(dataset)),count=count,
                correctness_only=True,axis_state_capture=True,axis_capture_hidden_last_only=True,
                axis_import_hash_after_model=True,gpu_state_digest=True,verify_GPU_digest_with_CPU=False,
                ignore_eos=False,qps=3.2,seed=2026112340)
            value['arrival_offsets']=np.r_[0.,np.cumsum(np.random.default_rng(value['seed']).exponential(1/value['qps'],count-1))].tolist()
            for key in ('replay_output_lengths','diagnostic_controlled_output_replay','qualification_gate_v2'):value.pop(key,None)
            runner.save(value);jobs.append(name)
        items.append(dict(**case,arm=axis,pair=pair,teacher=teacher,strict=strict,jobs=jobs,
            gate='pc_balance_'+label.lower()+'_composition.json',signature=runner.signature(value),
            source_capsule_SHA256=digest(source)))
    path=ROOT/'configs/p_service_balance_qualification_v1.json';assert not path.exists()
    write(path,dict(time=time.time(),items=items,state_prefix='pc_service_balance',source_capsule_SHA256=None,
        selection_basis='Only two additional controls. Layer2→8 reduces nominal P GEMM and Attention work by6/34 before queue effects. '
            'Hybrid2 at suffix3/17 gives nominal P28L/D8L rows,close to StockToken WorkCut8. '
            'The hybrid P Attention coefficient is23.0588 versus Stock21.7778 before finite-length/alignment terms. '
            'Current Hybrid2/f1/8 has P29.75L rows and P Attention coefficient26.03125. '
            'All formulas are structural accounting,not measured speedup. Exact actual aligned rows remain in runtime records.',
        scope='Change only Layer cut or Hybrid token fraction relative to the same-axis false-retirement control. '
            'Keep source credit16/8 and D cap2 unchanged. These are not D-policy-matched comparisons against Stock cap0/credit0. '
            'Independent original-weight teachers,N8 exact state,N32/N400 lifetime/coverage and frozen task-quality gates are mandatory. '
            'No full cut/QPS sweep,no output or SLO change,no active-decode insertion. '
            'Performance selection must use Discovery208 before any new candidate heldout confirmation.'))
    children=[]
    for item in items:
        log=ROOT/'logs'/f'service_balance_qualification_lane{item["lane"]}.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.service_balance_probe',
                'run',str(item['lane'])],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(label=item['label'],lane=item['lane'],pid=child.pid))
    write(ROOT/'state/service_balance_qualification_controller.json',dict(time=time.time(),children=children))
    return children


def run(lane):
    bind()
    from ..dense_ar_output_aware_v2.c3_family import run as original
    original(lane,'p_service_balance_qualification_v1.json')


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(int(sys.argv[2])),flush=True)
