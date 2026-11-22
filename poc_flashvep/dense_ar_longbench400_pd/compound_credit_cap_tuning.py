"""Test the declared credit/concurrency interaction before static freezing."""
import hashlib
import json
from .common import ROOT,write
from .freeze_capsule import freeze
from .final_confirmation_jobs import clone,signature
from .run import save,execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .metrics import summarize


def main():
    assert not (ROOT/'configs/final_continuous_static_choices_frozen_v1.json').exists()
    capsule,sha=freeze('final_continuous_strong_confirmation_v1_20261121')
    sources={'Layer':'CreditTune_Discovery200_q2.8_Layer8_C8_Pool16_v1',
        'Token':'CreditTune_Discovery200_q2.8_Token8_C8_Pool16_v1',
        'Hybrid':'CreditTune_Discovery200_q2.8_Hybrid4_C8_Pool16_v1'}
    plan=[]
    for axis,name in sources.items():
        base=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        assert base['continuous_source_credit_limit']==8 and base['hidden_source_slots']==16
        base.update(D_running_admission_cap=2,
            D_cap_qualification_gate=f'D_admission_cap_2_{"StockToken8" if axis=="Token" else "Layer8" if axis=="Layer" else "Hybrid4"}_gate_v1.json')
        key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
        gate='Final_composition_'+key+'_v1.json'
        proof_name=f'FinalPreQ400_Compound{axis}_{key}_v1'
        cfg=clone(base,proof_name,'longbench_balanced400_v1.json',2.8,2026112201,(2,3),capsule,sha,correctness=True)
        save(cfg)
        performance=clone(base,f'CompoundTune_Discovery200_q2.8_{axis}_C8_Cap2_v1',
            'balanced400_discovery.json',2.8,2026112291,(2,3),capsule,sha)
        performance.update(discovery_only=True,final_composition_gate=gate,
            scope='Before final choices,combine source credit8/fixed pool16 with common D running cap2. '
                'Separate sweeps already tested both controls. This targeted interaction can improve safety or regress latency. '
                'Keep unrestricted and previous cap controls. All3 axes use the same compound policy. '
                'Source,model,dataset,arrival,B8192/S1024 and frozen SLO remain fixed. No held-out timing has been observed.')
        save(performance)
        plan.append(dict(axis=axis,proof=proof_name,gate=gate,performance=performance['name'],signature=signature(cfg)))
    write(ROOT/'configs/compound_credit_cap_discovery_frozen_v1.json',dict(jobs=plan,
        source_capsule_SHA256=sha,evidence='FROZEN_TARGETED_CONTROL_INTERACTION',
        hypothesis='Credit8 reduces source stalls. A common running cap2 may prevent excessive decode/partial-prefill concurrency. '
            'The combination need not help. Retain all failures and regressions. Require fresh full400 proof before n1 timing.'))
    results=[]
    for job in plan:
        name=job['proof']
        try:
            assert execute(name),name
            state=audit(name)
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
            score=quality(name)
            assert score['passed'],score
            proof=dict(passed=True,full400_passed=True,name=name,signature=job['signature'],
                source_capsule_SHA256=sha,state=state,quality=score,compute_rule=no_mixed(name),
                evidence='CORRECTNESS_ONLY',scope='New final-source compound-policy Full400 stress. '
                    'Inherited independent N8 action proofs remain separate. No performance or decode confidence claim.')
        except Exception as error:
            proof=dict(passed=False,name=name,error=repr(error),signature=job['signature'],
                source_capsule_SHA256=sha,evidence='FAILED')
        path=ROOT/'state'/job['gate']
        assert not path.exists()
        write(path,proof)
        if not proof['passed']:
            results.append(dict(job=job,status='FAILED_CORRECTNESS'))
            continue
        assert execute(job['performance']),job['performance']
        m=summarize(job['performance'])
        score=quality(job['performance'])
        results.append(dict(job=job,status='MEASURED_ONLINE',quality=score,metrics=m))
        print('COMPOUND_DISCOVERY',job['axis'],m['TTFT_p95_ms'],m['ITL_p99_ms'],m['TPOT_p95_ms'],score['passed'],flush=True)
    write(ROOT/'state/compound_credit_cap_discovery_complete_v1.json',dict(complete=True,results=results))


if __name__=='__main__':
    main()
