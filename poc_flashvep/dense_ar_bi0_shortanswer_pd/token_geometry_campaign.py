"""Require strict same-partition output and full KV equality before timing."""
import hashlib
import json
import time
from .common import ROOT,write,save
from .qualification import configuration,records
from .run import execute


def main(tag='v1',qualification_tag='shared128_v2',asynchronous=False):
    declaration=ROOT/'configs'/f'frozen_token_same_geometry_teacher_{tag}.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),N=8,cuts=[4,8,12,1,2,6],KV_block=128,
            B=8192,S=1024,execution=f'Compile0, full decode graph, async={asynchronous}, BI0.',
            gate='Full valid prompt KV, last hidden, greedy sequence and query spans must equal the same-partition single-GPU teacher.',
            reason='Full-query teacher geometry differs from Token prefix/suffix geometry. Exact transport alone does not verify all causal continuation math.',
            scope='Correctness only. Lower per-step token budget creates the same boundary on one GPU. No timing or throughput eligibility by itself.'))
    results=[]
    for cut in (4,8,12,1,2,6):
        while True:
            file=ROOT/'analysis'/f'BI0_axis_state_qualification_{qualification_tag}.json'
            prior=json.loads(file.read_text())['results'] if file.exists() else []
            if any(r['axis']=='Token' and r['cut']==cut and r['passed'] for r in prior):break
            for failed in ('shared_axis_qualification_hold.json',f'axis_state_mismatch_{qualification_tag}.json',
                           f'axis_qualification_{qualification_tag}_failed.json','async_qualification_state_hold.json','STOP_REQUESTED.json'):
                assert not (ROOT/'state'/failed).exists(),failed
            time.sleep(3)
        name=f'qual_TokenGeometryTeacher_Last{cut}_N8_BI0_{tag}'
        cfg=configuration(name,'Native','qualification8.json',reference_single=True,gpus=[1],
            execution_family='MATCHED_NO_COMPILE',async_scheduling=asynchronous,numerical_probe=True,
            numerical_probe_sync=True,token_teacher_cut=cut,KV_block_tokens=128,restore_official_UCX_init=True)
        assert execute(save(cfg)),name
        import torch
        torch.set_num_threads(1)
        teacher=json.loads((ROOT/'raw'/name/'prompt_state_single.json').read_text())['requests']
        actual_name=f'qual_Token_Last{cut}_ShortAnswer_N8_BI0_{qualification_tag}'
        actual=json.loads((ROOT/'raw'/actual_name/'axis_events_decode.json').read_text())['states']
        expected={r['sample_id']:r['output_ids'] for r in records(ROOT/'raw'/name/'requests.jsonl')}
        outputs={r['sample_id']:r['output_ids'] for r in records(ROOT/'raw'/actual_name/'requests.jsonl')}
        teacher_queries=json.loads((ROOT/'raw'/name/'actual_queries_single.json').read_text())['rows']
        token_queries={role:json.loads((ROOT/'raw'/actual_name/f'actual_queries_{role}.json').read_text())['rows'] for role in ('prefill','decode')}
        rows=[]
        for index in range(8):
            rid=f'q{index}';hidden=torch.load(ROOT/'raw'/name/f'numerical_single_{rid}.pt',weights_only=True)['hidden_last']
            digest=hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            lengths=json.loads((ROOT/'data/qualification8.json').read_text());length=lengths[index]['input_tokens']
            a=[(q['start'],min(q['count'],length-q['start'])) for q in teacher_queries if q['request_id']==rid and q['start']<length]
            b=[(q['start'],min(q['count'],length-q['start'])) for role in ('prefill','decode') for q in token_queries[role] if q['request_id']==rid and q['start']<length]
            rows.append(dict(request_id=rid,query_spans_exact=a==b,teacher_query_spans=a,actual_query_spans=b,
                full_KV_exact=teacher[rid]['KV']==actual[rid]['KV'],last_hidden_exact=digest==actual[rid]['last_hidden_SHA256'],
                greedy_sequence_exact=expected[index]==outputs[index]))
        passed=all(r['query_spans_exact'] and r['full_KV_exact'] and r['last_hidden_exact'] and r['greedy_sequence_exact'] for r in rows)
        results.append(dict(cut=cut,passed=passed,rows=rows,teacher=name,actual=actual_name))
        write(ROOT/'analysis'/f'BI0_Token_same_geometry_gate_{tag}.json',dict(results=results,evidence='CORRECTNESS_ONLY'))
        if not passed:
            write(ROOT/'state'/f'token_geometry_mismatch_{tag}.json',dict(time=time.time(),cut=cut,status='BLOCKED_CORRECTNESS'));return
    flag='token_geometry_teacher_complete.json' if tag=='v1' else f'token_geometry_teacher_{tag}_complete.json'
    write(ROOT/'state'/flag,dict(time=time.time(),passed=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/token_geometry_teacher_hold.json',dict(time=time.time(),error=repr(error)));raise
