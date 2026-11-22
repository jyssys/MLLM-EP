"""Use an available pair for final-source control proofs,not test timing."""
import hashlib
import json
from .common import ROOT,write
from .freeze_capsule import freeze
from .final_confirmation_jobs import clone,signature
from .run import save,execute
from .decode_cap_discovery import full_native_state
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed


def main():
    assert not (ROOT/'configs/final_continuous_static_choices_frozen_v1.json').exists(), 'This is a pre-choice control proof'
    capsule,sha=freeze('final_continuous_strong_confirmation_v1_20261121')
    controls=('RemoteSetupAB_Discovery200_q2.8_NativeStrong_v1',
        'RemoteSetupAB_Discovery200_q2.8_NativeSetup_v1',
        'DcapTune_Discovery200_q2.8_Native_Cap2_v1','NativeCompute_Discovery200_q2.8_v1')
    results=[]
    for original in controls:
        base=json.loads((ROOT/'configs'/(original+'.json')).read_text())
        key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
        name='FinalPreQ400_Native_'+key+'_v1'
        cfg=clone(base,name,'longbench_balanced400_v1.json',2.8,2026112201,(2,3),capsule,sha,correctness=True)
        save(cfg)
        try:
            assert execute(name),name
            state=full_native_state(name,False)
            score=quality(name)
            assert score['passed'],score
            result=dict(passed=True,full400_passed=True,name=name,signature=signature(cfg),
                source_capsule_SHA256=sha,state=state,quality=score,
                compute_rule=no_mixed(name) if cfg.get('d_compute_throttle') else None,
                evidence='CORRECTNESS_ONLY',
                scope='Final-source Full400 control proof before static choices or held-out timing. '
                    'The pair was available after its credit sweep. No test latency selects a static action. '
                    'The final plan may reuse only this exact source/signature proof. '
                    'New-source offloading compositions still need separate Full400 proofs.')
        except Exception as error:
            result=dict(passed=False,name=name,error=repr(error),source_capsule_SHA256=sha,
                signature=signature(cfg),evidence='FAILED')
        path=ROOT/'state'/('Final_composition_'+key+'_v1.json')
        assert not path.exists(), 'Preserve an earlier proof or failure'
        write(path,result)
        results.append(result)
        print('FINAL_SOURCE_NATIVE_CONTROL_PROOF',name,result['passed'],flush=True)
    write(ROOT/'state/final_native_prequalification_complete_v1.json',dict(complete=True,results=results))


if __name__=='__main__':
    main()
