"""Qualify the zero-offload2D boundary with the normal Native handoff geometry."""
import json
import time
import torch
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import audit
from .qualification_gate import KV_equal


def compare(teacher,actual):
    tf=ROOT/'raw'/teacher;af=ROOT/'raw'/actual
    gold=json.loads((tf/'prompt_state_decode.json').read_text())['requests']
    measured=json.loads((af/'axis_events_decode.json').read_text())['states']
    ga={r['sample_id']:r for r in map(json.loads,(tf/'requests.jsonl').read_text().splitlines())}
    aa={r['sample_id']:r for r in map(json.loads,(af/'requests.jsonl').read_text().splitlines())}
    rows=[]
    for i in aa:
        rid='q'+str(i);g=gold[rid];a=measured[rid]
        gh=torch.load(g['hidden_tensor_file'],map_location='cpu',weights_only=True)['hidden_last']
        ah=torch.load(a['tensor_file'],map_location='cpu',weights_only=False)['hidden'][-1]
        rows.append(dict(request_id=rid,full_valid_KV_exact=KV_equal(g['KV'],a['KV']),
            hidden_exact=torch.equal(gh.flatten(),ah.flatten()),greedy_exact=ga[i]['output_ids']==aa[i]['output_ids']))
    return dict(rows=rows,passed=all(all(r[k] for k in ('full_valid_KV_exact','hidden_exact','greedy_exact')) for r in rows))


def main():
    torch.set_num_threads(1)
    capsule,sha=freeze('native_zero_boundary_qualification_v1_20261121')
    common=dict(pair=(2,3),correctness=True,controlled=True,serial=True,output=16,fixed=True,
        source_capsule=capsule,source_capsule_SHA256=sha,wall_timeout=3600,serving_timeout=3300)
    teacher=create('Q24_OriginalNativeProxy_Teacher_N8_v1','Native','qualification8.json',
        state_capture=True,axis_state_capture=False,**common)
    jobs=[teacher]
    for n in (1,8):
        jobs.append(create(f'Q24_HybridZeroBoundary_N{n}_v1','Hybrid','qualification8.json',
            cut=0,count=n,hybrid_suffix_fraction=0.,partition_aligned_chunks=True,
            axis_state_capture=True,axis_hash_workers=8,**common))
    write(ROOT/'configs/native_zero_boundary_qualification_v1.json',dict(jobs=jobs,
        evidence='FROZEN_CORRECTNESS_DESIGN',
        scope='Delegate m=s=0 remote matching to the original connector. The original Native proxy recomputes its final prompt query. '
            'Compare the complete post-query KV,hidden and greedy output with an unchanged original-model Native PD teacher. '
            'Before that query,compare only the valid L-1 imported prefix. No model or token correction.'))
    results=[]
    for i,name in enumerate(jobs):
        assert execute(name),name
        if i==0:continue
        coverage=audit(name);strict=compare(teacher,name)
        assert coverage['passed'] and coverage['decode_full36_observation_complete'] and strict['passed'],strict
        assert all(r['status']=='PASS' for r in coverage['imported_prefix_KV'])
        results.append(dict(name=name,coverage=coverage,strict=strict))
    prior=json.loads((ROOT/'state/C0_family_q7_gate.json').read_text())
    actions=list(prior['qualified_actions'])
    actions.extend([dict(axis='Hybrid',cut=0,suffix_fraction=0.),dict(axis='Native',cut=0,suffix_fraction=None)])
    write(ROOT/'state/C0_family_with_native_q24_gate_v1.json',dict(time=time.time(),passed=True,
        qualified_axes=['Native','Layer','Token','Hybrid'],qualified_actions=actions,
        native_zero_boundary=True,mixed_rows_passed=False,results=results,evidence='CORRECTNESS_ONLY',
        scope='Original Q7 action-family proofs plus the separately proven zero-offload Native boundary. '
            'The extra Native last-prompt query is explicit. Heterogeneous continuous400 needs a separate gate. '
            'This is not an online selector,speedup,or decode-safety result.'))
    print('NATIVE_ZERO_BOUNDARY_FULL_STATE_PASS',flush=True)


if __name__=='__main__':main()
