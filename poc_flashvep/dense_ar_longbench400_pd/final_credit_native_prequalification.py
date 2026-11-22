"""Use the idle pair for existing credit-control composition proofs only."""
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
    assert not (ROOT/'configs/final_continuous_static_choices_frozen_v1.json').exists()
    capsule,sha=freeze('final_continuous_strong_confirmation_v1_20261121')
    results=[]
    for credit in (8,16):
        original=f'CreditTune_Discovery200_q2.8_Native8_C{credit}_Pool16_v1'
        base=json.loads((ROOT/'configs'/(original+'.json')).read_text())
        key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
        name=f'FinalPreQ400_NativeCredit{credit}_{key}_v1'
        cfg=clone(base,name,'longbench_balanced400_v1.json',2.8,2026112201,(6,7),capsule,sha,correctness=True)
        save(cfg)
        try:
            assert execute(name),name
            state=full_native_state(name,False)
            score=quality(name)
            assert score['passed'],score
            result=dict(passed=True,full400_passed=True,name=name,signature=signature(cfg),
                source_capsule_SHA256=sha,state=state,quality=score,compute_rule=no_mixed(name),
                evidence='CORRECTNESS_ONLY',scope='New-source proof of an already declared Native credit control. '
                    'Use no proof latency for candidate selection. This is not another performance grid. '
                    'A selected composition can reuse this proof only when source and signature are identical.')
        except Exception as error:
            result=dict(passed=False,name=name,error=repr(error),source_capsule_SHA256=sha,
                signature=signature(cfg),evidence='FAILED')
        path=ROOT/'state'/('Final_composition_'+key+'_v1.json')
        assert not path.exists(),'Keep an earlier proof or failure'
        write(path,result)
        results.append(result)
        print('FINAL_SOURCE_NATIVE_CREDIT_PROOF',credit,result['passed'],flush=True)
    write(ROOT/'state/final_credit_native_prequalification_complete_v1.json',dict(complete=True,results=results))


if __name__=='__main__':
    main()
