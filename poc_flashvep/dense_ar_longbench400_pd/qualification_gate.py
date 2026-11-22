"""Compare all full-state qualifications with their legal full-model teachers."""
import hashlib
import json
import torch
from .common import ROOT,write,digest,SOURCE
from .qualification import audit


def KV_equal(expected,actual):
    assert len(expected)==len(actual),'Different valid layer counts'
    for a,b in zip(expected,actual):
        alg_a=a.get('digest_algorithm','RAW_SHA256');alg_b=b.get('digest_algorithm','RAW_SHA256')
        if alg_a!=alg_b:raise ValueError(('Incompatible KV digest algorithms. This is not tensor inequality.',alg_a,alg_b))
        assert a.get('chunk_bytes')==b.get('chunk_bytes'),'Different digest leaf sizes'
    return expected==actual


def compare(teacher,actual):
    tf=ROOT/'raw'/teacher;af=ROOT/'raw'/actual
    expected=json.loads((tf/'prompt_state_single.json').read_text())['requests']
    states=json.loads((af/'axis_events_decode.json').read_text())['states']
    gold={r['sample_id']:r for r in map(json.loads,(tf/'requests.jsonl').read_text().splitlines())}
    output={r['sample_id']:r for r in map(json.loads,(af/'requests.jsonl').read_text().splitlines())}
    rows=[]
    count=json.loads((af/'configuration.json').read_text())['count']
    assert len(gold)==len(output)==count
    for index in range(count):
        rid='q'+str(index);state=states[rid]
        hidden=torch.load(tf/f'numerical_single_{rid}.pt',map_location='cpu',weights_only=True)['hidden_last']
        measured=torch.load(state['tensor_file'],map_location='cpu',weights_only=False)['hidden'][-1]
        rows.append(dict(request_id=rid,full_KV_exact=KV_equal(expected[rid]['KV'],state['KV']),
            last_hidden_exact=torch.equal(hidden,measured),greedy_exact=gold[index]['output_ids']==output[index]['output_ids']))
    return dict(teacher=teacher,actual=actual,rows=rows,
        passed=all(all(r[key] for key in ('full_KV_exact','last_hidden_exact','greedy_exact')) for r in rows))


def main():
    torch.set_num_threads(1);results=[]
    for cut in (2,4,8,12):
        for axis in ('Layer','Token'):
            actual=f'Q4_{axis}{cut}_N8';audit(actual)
            teacher='Q4_Full36_Teacher_N8' if axis=='Layer' else f'Q4_TokenTeacher{cut}_N8'
            result=compare(teacher,actual);results.append(result)
            write(ROOT/'analysis/full_state_q4_comparisons.json',dict(results=results,evidence='CORRECTNESS_ONLY'))
            print(actual,result['passed'],flush=True)
    for actual,teacher in (('Q4_Hybrid4_f0125_Aligned_N8','Q4_HybridBoundaryTeacher_f0125_N8'),
            ('Q4_Hybrid0_f0125_TokenBoundary_N8','Q4_HybridBoundaryTeacher_f0125_N8'),
            ('Q4_Hybrid4_f0_LayerBoundary_N8','Q4_Full36_Teacher_N8')):
        audit(actual);result=compare(teacher,actual);results.append(result)
        write(ROOT/'analysis/full_state_q4_comparisons.json',dict(results=results,evidence='CORRECTNESS_ONLY'))
        print(actual,result['passed'],flush=True)
    passed=all(r['passed'] for r in results)
    write(ROOT/'state/C0_full_state_q4_gate.json',dict(passed=passed,qualified_axes=['Layer','Token','Hybrid'] if passed else [],
        qualified_cuts=[2,4,8,12],hybrid_fractions=[.125,0.],n=8,mixed_rows_passed=False,
        qualifier='C0 serial full-state gate. C3 natural mixed requests need their separate validation.',
        results=results,evidence='CORRECTNESS_ONLY'))
    assert passed,'Full-state mismatch. Do not use this gate for performance.'


if __name__=='__main__':main()
