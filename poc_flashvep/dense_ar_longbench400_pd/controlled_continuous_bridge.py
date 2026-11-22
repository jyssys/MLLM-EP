"""Test an existing C0 interior winner in actual continuous scheduling."""
import argparse
import hashlib
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .freeze_capsule import freeze
from .final_confirmation_jobs import clone,signature
from .run import save,execute
from .qualification import audit
from .decode_cap_discovery import full_native_state
from .compute_throttle_qualification import no_mixed
from .quality_gate import compare as quality
from .metrics import summarize


PLAN='controlled_continuous_I3_bridge_frozen_v1.json'


def prototypes(cap):
    names=dict(NativeStrong='RemoteSetupAB_Discovery200_q2.8_NativeStrong_v1',
        NativeCommon=f'DcapTune_Discovery200_q2.8_Native_Cap{cap}_v1',
        Layer='CompoundTune_Discovery200_q2.8_Layer_C8_Cap2_v1',
        StockToken='CompoundTune_Discovery200_q2.8_Token_C8_Cap2_v1',
        Hybrid='CompoundTune_Discovery200_q2.8_Hybrid_C8_Cap2_v1')
    result={k:json.loads((ROOT/'configs'/(v+'.json')).read_text()) for k,v in names.items()}
    for label in ('Layer','StockToken','Hybrid'):
        result[label].update(D_running_admission_cap=cap,
            D_cap_qualification_gate=f'D_admission_cap_{cap}_{"StockToken8" if label=="StockToken" else "Layer8" if label=="Layer" else "Hybrid4"}_gate_v1.json')
    token=dict(result['StockToken'],token_stock_model=False,execution_family='MATCHED_NO_COMPILE',
        qualification_gate='C3_continuous_actions_gate_v1.json')
    for field in ('remote_family_gate','compute_family_gate','source_credit_qualification_gate'):
        token.pop(field,None)
    result['TokenMode0Control']=token
    return result


def plan():
    path=ROOT/'configs'/PLAN
    assert not path.exists()
    capsule,sha=freeze('final_continuous_strong_confirmation_v1_20261121')
    sets=json.loads((ROOT/'data/iso8_hybrid_sets_frozen_v1.json').read_text())['rows']
    selected=next(r for r in sets if r['split']=='heldout' and r['input_bucket']==3)
    assert selected['dataset']=='iso8_input3_heldout_exact16.json'
    proofs={};cells=[];jobs=[]
    for cap,output in ((1,16),(1,64),(2,64)):
        for label,base in prototypes(cap).items():
            key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
            if key in proofs:continue
            prior=ROOT/'state'/('Final_composition_'+key+'_v1.json')
            if prior.exists():
                p=json.loads(prior.read_text())
                assert p.get('signature')==signature(base) and p.get('source_capsule_SHA256')==sha
                if p.get('full400_passed') and p.get('passed'):
                    proofs[key]=dict(signature=signature(base),gate=prior.name,prequalified=True,label=label)
                    continue
            gate='C3Bridge_composition_'+key+'_v1.json'
            names=[]
            for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
                name=f'C3BridgeQ_{label}_Cap{cap}_{key}_Natural{n}_v1'
                cfg=clone(base,name,dataset,2.8,2026112201,(4,5),capsule,sha,correctness=True)
                save(cfg);names.append(name)
            proofs[key]=dict(signature=signature(base),gate=gate,prequalified=False,jobs=names,label=label)
        for start,seed in enumerate((2026112551,2026112552,2026112553)):
            arms={}
            for label,base in prototypes(cap).items():
                key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
                name=f'C3Bridge_I3_O{output}_Cap{cap}_s{start}_{label}_v1'
                cfg=clone(base,name,selected['dataset'],1.253873,seed,(4,5),capsule,sha)
                cfg.update(output_cap=output,ignore_eos=True,bridge_composition_gate=proofs[key]['gate'],
                    iso_work_frontier=True,target_D_rows=selected['D_rows_each'],target_P_rows=selected['P_rows_each'],
                    bridge_controlled_output=True,bridge_discovery_only=False,
                    scope='Predeclared controlled-output continuous bridge from the existing long-input C0 iso-work Hybrid candidate. '
                        'All three conditions and all six arms run three new paired starts. Retain losses and safety failures. '
                        'P remains continuous. D running cap1/2 is not a forced batch count or the whole-request C0 barrier. '
                        'Stock Token keeps the strong original compiled full model. Mode0 Token is a diagnostic runtime-family control,not its replacement. '
                        'Cross-runtime mode0/stock comparison can include adapter and numerical-path differences,not only a compile flag. '
                        'This small16-request test is not natural400 throughput or sustainable capacity. Do not tune the main static choices from it.')
                if label=='Hybrid':cfg['suffix_by_request']=selected['Hybrid_suffix_by_request']
                elif label in ('StockToken','TokenMode0Control'):
                    cfg.update(suffix_by_request=selected['Token_suffix_by_request'],aggregate_work_audit=dict(
                        target_D_rows=selected['D_rows_each'],actual_D_rows=selected['D_rows_each'],
                        signed_error_rows=0,exact_match=True,alignment_tokens=16))
                save(cfg);arms[label]=name
            block=list(arms.values());np.random.default_rng(seed+cap*100+output).shuffle(block)
            jobs.extend(block);cells.append(dict(D_cap=cap,output=output,start=start,seed=seed,QPS=1.253873,
                GPU_pair=[4,5],dataset=selected['dataset'],jobs=arms))
    write(path,dict(proofs=proofs,cells=cells,performance_jobs=jobs,source_capsule_SHA256=sha,
        dataset_SHA256=digest(ROOT/'data'/selected['dataset']),
        evidence='FROZEN_CONTINUOUS_BRIDGE_CONFIRMATION_DESIGN',
        scope='Choose the existing C0 I3/output64 iso-work Hybrid candidate before new C3 bridge timing. '
            'Also test its output16 counterexample and common cap1→2 intervention. No cell is dropped after outcomes. '
            'All three independent starts use4/5. The old C0 comparison used6/7 and other seeds/source. Do not compute a paired C0→C3 CI. '
            'Within each new C3 cell,all arms share the same source,pair,requests,arrival and exact output count. '
            'Layer8,Token8 and Hybrid4 have equal logical aggregate P/D rows. Padding and actual batches need separate audit. '
            'Natural32/400 proof is mandatory before timing any new execution/cap composition. All final-source cached proofs require exact signatures. '
            'This bridge is not candidate tuning for the main Natural400 evaluation or a global best-Hybrid claim.'))
    print('CONTROLLED_C3_BRIDGE_FROZEN',len(jobs),'starts',len(proofs),'composition keys',flush=True)


def run():
    protocol=json.loads((ROOT/'configs'/PLAN).read_text())
    for key,item in protocol['proofs'].items():
        if item['prequalified']:continue
        target=ROOT/'state'/item['gate']
        if target.exists():continue
        results=[]
        try:
            for name in item['jobs']:
                assert execute(name),name
                cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
                state=audit(name) if cfg.get('axis_kind') else full_native_state(name,False)
                assert state['passed']
                if cfg.get('axis_kind'):
                    assert state['decode_full36_observation_complete']
                    assert all(r['status']=='PASS' for r in state['imported_prefix_KV'])
                score=quality(name);assert score['passed'],score
                results.append(dict(name=name,state=state,quality=score,
                    compute_rule=no_mixed(name) if cfg.get('d_compute_throttle') else None))
            proof=dict(passed=True,full400_passed=True,signature=item['signature'],
                source_capsule_SHA256=protocol['source_capsule_SHA256'],results=results,evidence='CORRECTNESS_ONLY')
        except Exception as error:
            proof=dict(passed=False,signature=item['signature'],results=results,error=repr(error),evidence='FAILED')
        write(target,proof)
        print('C3_BRIDGE_COMPOSITION',key,proof['passed'],flush=True)
    results=[]
    for name in protocol['performance_jobs']:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        proof=json.loads((ROOT/'state'/cfg['bridge_composition_gate']).read_text())
        if not proof['passed']:
            results.append(dict(name=name,status='NOT_RUN',reason='BLOCKED_CORRECTNESS'))
            continue
        assert signature(cfg)==proof['signature'] and proof['source_capsule_SHA256']==cfg['source_capsule_SHA256']
        success=execute(name)
        if success:
            coverage=audit(name) if cfg.get('axis_kind') else None
            if coverage:assert coverage['passed'] and coverage['decode_full36_observation_complete']
            m=summarize(name)
            result=dict(name=name,status='MEASURED_ONLINE',TTFT_p95_ms=m['TTFT_p95_ms'],ITL_p99_ms=m['ITL_p99_ms'])
        else:result=dict(name=name,status='FAILED')
        results.append(result)
        write(ROOT/'state/controlled_continuous_bridge_progress_v1.json',dict(results=results))
        print('C3_BRIDGE_SERVING',result,flush=True)
    write(ROOT/'state/controlled_continuous_bridge_complete_v1.json',dict(complete=True,results=results,
        scope='Completion records outcomes,not a successful scientific decision. Failed and blocked arms stay visible.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','run'));args=parser.parse_args()
    plan() if args.action=='plan' else run()
