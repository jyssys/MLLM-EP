"""Test a recurrent decode-tail symptom without changing the frozen main test."""
import argparse
import hashlib
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .final_confirmation_jobs import clone,signature
from .final_continuous_selection import policy
from .run import save,execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .metrics import summarize


PLAN='hybrid_READ_repair_frozen_v1.json'


def build():
    target=ROOT/'configs'/PLAN
    assert not target.exists(),'Keep the repair design fixed before new timings'
    main=json.loads((ROOT/'configs/final_continuous_execution_plan_v1.json').read_text())
    observed=next(c for c in main['cells'] if c['workload']=='Main400' and c['QPS']==1.8 and c['start']==2)
    entries={a:json.loads((ROOT/'configs'/(n+'.json')).read_text()) for a,n in observed['jobs'].items()}
    base=entries['BestHybrid'];assert base['fuse_short_hybrid'] and not base.get('defer_D_remote_setup')
    assert base['continuous_source_credit_limit']==4 and base['hidden_source_slots']==4
    repair=dict(base,defer_D_remote_setup=True)
    wanted=dict(NativeStrong=(False,False,False,None,False),
        NativeCompute=(True,False,False,None,False),
        NativeREAD=(True,True,False,None,False),
        NativeCompletion=(True,True,False,None,True))
    prototypes={}
    for label,p in wanted.items():
        candidates=[c for a,c in entries.items() if a.startswith('Native') and policy(c)==p]
        assert len(candidates)==1,(label,p)
        prototypes[label]=candidates[0]
    prototypes.update(HybridCompute=base,HybridREAD=repair,StockToken=entries['BestToken'])
    capsule,sha=main['source_capsule'],main['source_capsule_SHA256']
    key=hashlib.sha256(json.dumps(signature(repair),sort_keys=True).encode()).hexdigest()[:12]
    gate='Final_composition_'+key+'_v1.json'
    assert not (ROOT/'state'/gate).exists(),'This design requires a fresh combined fused/READ proof'
    proofs=[]
    for count,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        name=f'RepairQ_FusedHybridREAD_Natural{count}_v1'
        cfg=clone(repair,name,dataset,2.8,2026112201,(6,7),capsule,sha,correctness=True)
        save(cfg);proofs.append(name)
    pilot_name='RepairPilot_Main400_q1.8_FusedHybridREAD_ObservedTrace_v1'
    cfg=clone(repair,pilot_name,observed['dataset'],observed['QPS'],observed['seed'],
        observed['GPU_pair'],capsule,sha)
    assert cfg['arrival_offsets']==base['arrival_offsets']
    cfg.update(final_composition_gate=gate,repair_diagnostic=True,
        scope='Post-observation same-trace READ-only intervention. Original main result remains. '
            'This pilot is not an independent held-out confirmation or a source of new main static choices.')
    save(cfg)
    pilot=dict(stage='OBSERVED_TRACE_DIAGNOSTIC',QPS=1.8,start=2,seed=observed['seed'],
        GPU_pair=observed['GPU_pair'],dataset=observed['dataset'],jobs=dict(HybridREAD=pilot_name,
            HybridCompute=observed['jobs']['BestHybrid'],NativeStrong=observed['jobs']['NativeStrong'],
            NativeCompute=next(observed['jobs'][a] for a,c in entries.items()
                if a.startswith('Native') and policy(c)==wanted['NativeCompute']),
            NativeREAD=next(observed['jobs'][a] for a,c in entries.items()
                if a.startswith('Native') and policy(c)==wanted['NativeREAD'])))
    cells=[pilot];jobs=[pilot_name]
    for index,seed in enumerate((2026112661,2026112662,2026112663)):
        arms={}
        for label,prototype in prototypes.items():
            name=f'RepairConfirm_Main400_q1.8_s{index}_{label}_v1'
            cfg=clone(prototype,name,'longbench_balanced400_v1.json',1.8,seed,(6,7),capsule,sha)
            if label=='HybridREAD':cfg['final_composition_gate']=gate
            else:cfg['final_composition_gate']=prototype['final_composition_gate']
            cfg.update(repair_confirmation=True,
                scope='Predeclared follow-up after a main-test single-start tail symptom. '
                    'All seven controls run three new arrival-seed blocks. Do not retune the original main choices. '
                    'HybridCompute and HybridREAD differ only in large remote-READ setup deferral. '
                    'Keep strong compiled Token,unmodified Native and all exact Native D policies. '
                    'Natural BI0 outputs can differ;task quality and actual token work remain mandatory.')
            save(cfg);arms[label]=name
        ordered=list(arms.values());np.random.default_rng(seed).shuffle(ordered);jobs.extend(ordered)
        cells.append(dict(stage='INDEPENDENT_REPAIR_CONFIRMATION',QPS=1.8,start=index,seed=seed,
            GPU_pair=[6,7],dataset='longbench_balanced400_v1.json',jobs=arms))
    symptom_name=observed['jobs']['BestHybrid']
    symptom=json.loads((ROOT/'analysis'/(symptom_name+'_metrics.json')).read_text())
    write(target,dict(time=time.time(),source_capsule=capsule,source_capsule_SHA256=sha,
        original_main_plan_SHA256=digest(ROOT/'configs/final_continuous_execution_plan_v1.json'),
        observed_symptom=dict(name=symptom_name,TTFT_p95_ms=symptom['TTFT_p95_ms'],
            ITL_p99_ms=symptom['ITL_p99_ms'],TPOT_p95_ms=symptom['TPOT_p95_ms'],independent_starts=1),
        hypotheses=['Large remote READ setup delays decode submission even when large prefill compute is deferred.',
            'Actual decode batching or runtime host gaps explain the symptom without a READ-specific effect.',
            'The apparent regression is start/trace variation and does not persist on three new seeds.'],
        proof_jobs=proofs,composition_gate=gate,signature=signature(repair),cells=cells,performance_jobs=jobs,
        evidence='ADAPTIVE_REPAIR_DESIGN_FROZEN_BEFORE_NEW_TIMINGS',
        scope='No original main cell is removed or renamed. This is a separate adaptive follow-up,not the original blind test. '
            'Change only READ setup deferral within fused Hybrid. No model,output,timestamp,SLO,credit,pool or cut change. '
            'Fresh Natural32/400 combined-policy proof precedes timing. Failed proof means NOT_RUN performance. '
            'All future confirmation seeds and all controls are fixed before the pilot. Run them regardless of the pilot direction. '
            'READ timing also changes associated transfer/stream timing. A benefit is not exclusive host or DMA attribution. '
            'No repeated search until a preferred confidence interval appears.'))
    print('HYBRID_READ_REPAIR_FROZEN',len(jobs),'new serving starts; original140 unchanged',flush=True)


def run():
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    path=ROOT/'state'/plan['composition_gate']
    if not path.exists():
        results=[]
        try:
            for name in plan['proof_jobs']:
                assert execute(name),name
                state=audit(name)
                assert state['passed'] and state['decode_full36_observation_complete']
                assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
                score=quality(name);assert score['passed']
                results.append(dict(name=name,state=state,quality=score,compute_rule=no_mixed(name)))
            proof=dict(passed=True,full400_passed=True,name=plan['proof_jobs'][-1],
                source_capsule_SHA256=plan['source_capsule_SHA256'],signature=plan['signature'],
                results=results,evidence='CORRECTNESS_ONLY')
        except Exception as error:
            proof=dict(passed=False,results=results,error=repr(error),signature=plan['signature'],
                source_capsule_SHA256=plan['source_capsule_SHA256'],evidence='FAILED')
        write(path,proof);print('FUSED_HYBRID_READ_COMPOSITION',proof['passed'],flush=True)
    proof=json.loads(path.read_text());results=[]
    for name in plan['performance_jobs']:
        if not proof['passed']:
            results.append(dict(name=name,status='NOT_RUN',reason='BLOCKED_CORRECTNESS'))
            continue
        success=execute(name)
        if success:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            state=audit(name) if cfg.get('axis_kind') else None
            if state:assert state['passed'] and state['decode_full36_observation_complete']
            score=quality(name);m=summarize(name)
            result=dict(name=name,status='MEASURED_ONLINE' if score['passed'] else 'FAILED_TASK_QUALITY',
                quality=score,ITL_p99_ms=m['ITL_p99_ms'],TPOT_p95_ms=m['TPOT_p95_ms'],TTFT_p95_ms=m['TTFT_p95_ms'])
        else:result=dict(name=name,status='FAILED')
        results.append(result)
        write(ROOT/'state/hybrid_READ_repair_progress_v1.json',dict(results=results))
        print('HYBRID_READ_REPAIR',result,flush=True)
    write(ROOT/'state/hybrid_READ_repair_complete_v1.json',dict(complete=True,results=results,
        scope='Operational terminal outcomes. Scientific safety and performance require separate paired analyses.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('build','run'));args=parser.parse_args()
    build() if args.action=='build' else run()
