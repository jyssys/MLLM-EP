"""Validate promising compositions before final choices,without test timing."""
import hashlib
import json
from .common import ROOT,write,digest
from .freeze_capsule import freeze
from .final_confirmation_jobs import clone,signature
from .run import save,execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed


def main():
    assert not (ROOT/'configs/final_continuous_static_choices_frozen_v1.json').exists()
    preview=ROOT/'analysis/final_continuous_selection_preview_v1.json'
    selection=json.loads(preview.read_text())
    capsule,sha=freeze('final_continuous_strong_confirmation_v1_20261121')
    jobs=[]
    seen=set()
    for cell in selection['selected'].values():
        for family in ('Layer','Token','Hybrid'):
            row=cell[family]
            if row is None:
                continue
            base=json.loads((ROOT/'configs'/(row['name']+'.json')).read_text())
            key=hashlib.sha256(json.dumps(signature(base),sort_keys=True).encode()).hexdigest()[:12]
            if key in seen:
                continue
            seen.add(key)
            name=f'FinalPreQ400_{family}_{key}_v1'
            cfg=clone(base,name,'longbench_balanced400_v1.json',2.8,2026112201,(6,7),capsule,sha,correctness=True)
            save(cfg)
            jobs.append(dict(key=key,name=name,source_candidate=row['name'],signature=signature(cfg)))
    write(ROOT/'configs/final_candidate_prequalification_v1.json',dict(jobs=jobs,
        discovery_preview_SHA256=digest(preview),source_capsule_SHA256=sha,
        evidence='CORRECTNESS_ONLY_PRE_TEST_PLAN',
        scope='Use a currently available pair for new-source composition proofs. Static choices remain unfrozen. '
            'Do not use proof latencies for tuning. A later different candidate still needs its own proof.'))
    results=[]
    for item in jobs:
        name=item['name']
        try:
            assert execute(name),name
            state=audit(name)
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(r['status']=='PASS' for r in state['imported_prefix_KV'])
            score=quality(name)
            assert score['passed'],score
            proof=dict(passed=True,full400_passed=True,name=name,signature=item['signature'],
                source_capsule_SHA256=sha,state=state,quality=score,compute_rule=no_mixed(name),
                evidence='CORRECTNESS_ONLY',
                scope='New final-source Full400 candidate proof before held-out latency. '
                    'No performance or safety confidence claim. Exact source/signature is required for cache reuse.')
        except Exception as error:
            proof=dict(passed=False,name=name,error=repr(error),signature=item['signature'],
                source_capsule_SHA256=sha,evidence='FAILED')
        path=ROOT/'state'/('Final_composition_'+item['key']+'_v1.json')
        assert not path.exists(),'Do not replace an earlier proof or failure'
        write(path,proof)
        results.append(proof)
        print('FINAL_SOURCE_CANDIDATE_PROOF',name,proof['passed'],flush=True)
    write(ROOT/'state/final_candidate_prequalification_complete_v1.json',dict(complete=True,results=results))


if __name__=='__main__':
    main()
