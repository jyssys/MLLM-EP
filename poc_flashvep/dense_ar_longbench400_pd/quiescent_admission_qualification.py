"""Qualify the admission-only intervention before any serving claim."""
import argparse
import json
import time
import torch
from .common import ROOT,write
from .experiment import create
from .run import execute
from .freeze_capsule import freeze
from .qualification import audit
from .qualification_gate import compare
from .quality_gate import compare as quality


def strict_first(teacher,actual):
    tf=ROOT/'raw'/teacher;af=ROOT/'raw'/actual
    expected=json.loads((tf/'prompt_state_single.json').read_text())['requests']['q0']
    state=json.loads((af/'axis_events_decode.json').read_text())['states']['q0']
    gold=torch.load(tf/'numerical_single_q0.pt',map_location='cpu',weights_only=True)['hidden_last']
    hidden=torch.load(state['tensor_file'],map_location='cpu',weights_only=False)['hidden'][-1]
    gold_tokens=next(json.loads(s)['output_ids'] for s in (tf/'requests.jsonl').read_text().splitlines() if json.loads(s)['sample_id']==0)
    tokens=json.loads((af/'requests.jsonl').read_text().splitlines()[0])['output_ids']
    return dict(full_KV_exact=expected['KV']==state['KV'],last_hidden_exact=torch.equal(gold,hidden),greedy_exact=tokens==gold_tokens)


def check_admission(name):
    folder=ROOT/'raw'/name
    certificates=[json.loads(s) for s in (folder/'idle_certificates.jsonl').read_text().splitlines()]
    admissions=[json.loads(s) for s in (folder/'D_quiescent_admissions.jsonl').read_text().splitlines()]
    proof=audit(name)
    assert len(admissions)==proof['n'] and len(certificates)==proof['n']
    assert all(c['passed'] and c['scheduler_requests']==0 for c in certificates)
    assert proof['mixed_batches']==0 and proof['maximum_actual_D_batch']==1
    assert proof['passed'] and proof['decode_full36_observation_complete']
    assert all(r['status']=='PASS' for r in proof['imported_prefix_KV'])
    return dict(passed=True,n=proof['n'],device_certificates=len(certificates),
        maximum_actual_D_batch=1,mixed_batches=0,full36_decode=True,imported_prefix_KV=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('lane',type=int,choices=range(3));args=parser.parse_args()
    torch.set_num_threads(1)
    axis=('Layer','Token','Hybrid')[args.lane];first=args.lane*2
    capsule,sha=freeze('quiescent_admission_correctness_v1_20261121')
    extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if axis=='Hybrid' else {}
    common=dict(cut=4,pair=(first,first+1),correctness=True,d_quiescent_admission=True,
        continuous_source_credit_limit=4,source_capsule=capsule,source_capsule_SHA256=sha,
        axis_hash_workers=8,wall_timeout=3600,serving_timeout=3300,**extra)
    stages=[]
    for count in (1,8):
        stages.append(create(f'Q16_Guard_{axis}4_N{count}_v1',axis,'qualification8.json',
            count=count,output=16,fixed=True,serial=True,qps=20,**common))
    for count,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        stages.append(create(f'Q16_Guard_{axis}4_Natural{count}_v1',axis,dataset,
            qps=2.8,seed=2026112201,axis_capture_hidden_last_only=True,
            gpu_state_digest=True,axis_import_hash_after_model=True,**common))
    write(ROOT/'configs'/f'D_quiescent_qualification_lane{args.lane}_v1.json',dict(jobs=stages,
        evidence='FROZEN_CORRECTNESS_DESIGN',source_capsule_SHA256=sha,
        scope='Change D admission only. P remains continuous with source credit4. '
            'N1 then N8 strict full-state,then natural32 and full400 coverage/import/quality. '
            'No serving or safety claim follows from this observer-heavy validation.'))
    teacher={'Layer':'Q4_Full36_Teacher_N8','Token':'Q4_TokenTeacher4_N8',
        'Hybrid':'Q4_HybridBoundaryTeacher_f0125_N8'}[axis]
    results=[]
    for index,name in enumerate(stages):
        assert execute(name),name
        proof=check_admission(name)
        if index==0:
            numerical=strict_first(teacher,name);assert all(numerical.values()),numerical
        elif index==1:
            numerical=compare(teacher,name);assert numerical['passed']
        else:
            numerical=quality(name);assert numerical['passed'],numerical
        results.append(dict(name=name,admission=proof,numerical=numerical))
        write(ROOT/'state'/f'D_quiescent_{axis}_qualification_v1.json',dict(
            time=time.time(),axis=axis,results=results,passed=len(results)==len(stages),
            full400_passed=len(results)==len(stages),source_capsule_SHA256=sha,
            evidence='CORRECTNESS_ONLY'))
    print(axis,'ADMISSION_QUALIFICATION_PASS',flush=True)


if __name__=='__main__':main()
