"""Validate original-model whole-prefill deflection as a separate strong control."""
import json
import time
import torch
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .quality_gate import compare as quality


def strict(teacher,actual):
    tf=ROOT/'raw'/teacher;af=ROOT/'raw'/actual
    gold=json.loads((tf/'prompt_state_single.json').read_text())['requests']
    states=json.loads((af/'prompt_state_decode.json').read_text())['requests']
    gg={r['sample_id']:r for r in map(json.loads,(tf/'requests.jsonl').read_text().splitlines())}
    aa={r['sample_id']:r for r in map(json.loads,(af/'requests.jsonl').read_text().splitlines())}
    rows=[]
    for i in aa:
        rid='q'+str(i)
        h=torch.load(states[rid]['hidden_tensor_file'],map_location='cpu',weights_only=True)['hidden_last']
        g=torch.load(tf/f'numerical_single_{rid}.pt',map_location='cpu',weights_only=True)['hidden_last']
        rows.append(dict(request_id=rid,KV_exact=states[rid]['KV']==gold[rid]['KV'],
            hidden_exact=torch.equal(h.flatten(),g.flatten()),greedy_exact=aa[i]['output_ids']==gg[i]['output_ids']))
    return dict(rows=rows,passed=all(all(r[k] for k in ('KV_exact','hidden_exact','greedy_exact')) for r in rows))


def lifetime(name):
    folder=ROOT/'raw'/name
    for role in ('prefill','decode'):assert json.loads((folder/f'KV_lifetime_{role}.json').read_text())['passed']
    cfg=json.loads((folder/'configuration.json').read_text())
    records=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    states=json.loads((folder/'prompt_state_decode.json').read_text())['requests']
    states={rid:value for rid,value in states.items() if rid.startswith('q') and rid[1:].isdigit()}
    if cfg.get('deflection_force_local_control'):
        assert len(states)==len(records)
        assert all(s['query_coverage_end']==r['input_tokens'] and len(s['KV'])==36
            for r in records for s in [states['q'+str(r['sample_id'])]])
    deflections=[json.loads(s) for s in (folder/'whole_prefill_deflections.jsonl').read_text().splitlines()]
    assert deflections,'The new path must execute,not remain a no-op'
    return dict(passed=True,n=len(records),deflections=len(deflections),D_local_full_KV_states=len(states))


def main():
    torch.set_num_threads(1)
    capsule,sha=freeze('whole_prefill_deflection_qualification_v1_20261121')
    common=dict(pair=(6,7),correctness=True,execution_family='INSTALLED_DEFAULT_COMPILE',
        source_capsule=capsule,source_capsule_SHA256=sha,wall_timeout=4500,serving_timeout=4000)
    results=[]
    for n in (1,8):
        teacher=create(f'Q21_OriginalFull36_DeflectionTeacher_N{n}_v1','Native','qualification8.json',count=n,
            reference_single=True,serial=True,output=16,fixed=True,state_capture=True,
            axis_state_capture=False,numerical_probe=True,numerical_probe_sync=True,kv_layout='HND',**common)
        actual=create(f'Q21_DirectD_Full36_N{n}_v1','Native','qualification8.json',count=n,
            serial=True,output=16,fixed=True,state_capture=True,axis_state_capture=False,
            whole_prefill_deflection=True,deflection_force_local_control=True,d_quiescent_admission=True,**common)
        assert execute(teacher) and execute(actual)
        proof=strict(teacher,actual);life=lifetime(actual)
        write(ROOT/'analysis'/(actual+'_strict_validation.json'),proof)
        assert proof['passed'],proof
        results.append(dict(name=actual,strict=proof,lifetime=life))
    for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        actual=create(f'Q21_WholeDeflection_C3_Natural{n}_v1','Native',dataset,
            qps=2.8,seed=2026112201,state_capture=True,axis_state_capture=False,gpu_state_digest=True,
            whole_prefill_deflection=True,d_quiescent_admission=True,continuous_source_credit_limit=4,**common)
        assert execute(actual),actual
        life=lifetime(actual);score=quality(actual);assert score['passed'],score
        results.append(dict(name=actual,lifetime=life,quality=score))
        write(ROOT/'state/whole_prefill_deflection_gate_v1.json',dict(time=time.time(),
            passed=len(results)==4,full400_passed=len(results)==4,results=results,evidence='CORRECTNESS_ONLY',
            scope='Original full36 D-local prefill. No partition adapter,no partial-state handoff,no output correction. '
                'The online rule uses current source outstanding work and a D idle certificate. '
                'Requests assigned to P retain normal Native PD handoff. ITL and performance need clean independent tests.'))
    print('WHOLE_PREFILL_DEFLECTION_QUALIFIED',flush=True)


if __name__=='__main__':main()
