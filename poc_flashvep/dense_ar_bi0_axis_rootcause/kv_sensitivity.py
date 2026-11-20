"""Qualify KV16 before a small matched-configuration sensitivity test."""
import hashlib
import json
import time
from .common import ROOT,write,save
from .campaign import cell
from .run import execute
from ..dense_ar_bi0_shortanswer_pd.qualification import configuration,records


def teacher():
    name='KV16_TokenGeometryTeacher8_N8_v1'
    cfg=configuration(name,'Native','qualification8.json',reference_single=True,gpus=[1],
        execution_family='MATCHED_NO_COMPILE',async_scheduling=True,numerical_probe=True,
        numerical_probe_sync=True,token_teacher_cut=8,KV_block_tokens=16,restore_official_UCX_init=True)
    assert execute(save(cfg))
    import torch
    torch.set_num_threads(1)
    actual='qual_Token_Last8_ShortAnswer_N8_BI0_kv16_v1'
    reference=json.loads((ROOT/'raw'/name/'prompt_state_single.json').read_text())['requests']
    dest=json.loads((ROOT/'raw'/actual/'axis_events_decode.json').read_text())['states']
    expected={r['sample_id']:r['output_ids'] for r in records(ROOT/'raw'/name/'requests.jsonl')}
    output={r['sample_id']:r['output_ids'] for r in records(ROOT/'raw'/actual/'requests.jsonl')}
    tq=json.loads((ROOT/'raw'/name/'actual_queries_single.json').read_text())['rows']
    aq=[q for role in ('prefill','decode') for q in json.loads((ROOT/'raw'/actual/f'actual_queries_{role}.json').read_text())['rows']]
    lengths=json.loads((ROOT/'data/qualification8.json').read_text());rows=[]
    for i,row in enumerate(lengths):
        rid=f'q{i}';length=row['input_tokens']
        hidden=torch.load(ROOT/'raw'/name/f'numerical_single_{rid}.pt',weights_only=True)['hidden_last']
        value=hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        spans=lambda values:[(q['start'],min(q['count'],length-q['start'])) for q in values if q['request_id']==rid and q['start']<length]
        rows.append(dict(request_id=rid,full_KV_exact=reference[rid]['KV']==dest[rid]['KV'],
            final_hidden_exact=value==dest[rid]['last_hidden_SHA256'],strict_output_exact=expected[i]==output[i],
            query_spans_exact=spans(tq)==spans(aq)))
    passed=all(all(r[k] for k in ('full_KV_exact','final_hidden_exact','strict_output_exact','query_spans_exact')) for r in rows)
    write(ROOT/'analysis/BI0_Token_same_geometry_gate_kv16_v1.json',dict(results=[dict(cut=8,passed=passed,rows=rows)],evidence='CORRECTNESS_ONLY'))
    return passed


def main():
    declaration=ROOT/'configs/frozen_KV16_sensitivity_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),cut=8,N=200,QPS=2.35,seed=2026112089,blocks=[16,128],
            actual_batch=[1,1,1],source_credit=4,hidden_source_slots=4,BI0=True,B=8192,S=1024,
            gate='Fresh KV16 N8 state/greedy gates, same-geometry Token teacher and N32 controlled-lifetime tests precede performance.',
            scope='One start per axis/block. No KV16 outcome selects or replaces the primary KV128 baseline. This is a transport/admission sensitivity test, not best-vs-best tuning.'))
    while not (ROOT/'state/credit_sweep_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    from ..dense_ar_bi0_shortanswer_pd.axis_qualification import main as qualify
    qualify(tag='kv16_v1',cuts=(8,),block=16,restore=True,asynchronous=True)
    path=ROOT/'analysis/BI0_axis_state_qualification_kv16_v1.json'
    results=json.loads(path.read_text())['results'] if path.exists() else []
    if len(results)!=2 or not all(r['passed'] for r in results) or not teacher():
        write(ROOT/'state/KV16_controlled_qualification_v1.json',dict(time=time.time(),passed=False,status='BLOCKED_CORRECTNESS'));return
    burst=[]
    for axis in ('Layer','Token'):
        name=f'KV16_Burst_{axis}8_N32_v1'
        row=cell(name,axis,8,dataset='qualification32.json',count=32,rate=10,seed=2026112089,
            KV_block_tokens=16,qualified_controlled_performance=False,qualified_async_runtime=False,
            correctness_only=True,async_qualification_only=True,phase='KV16_CONTROLLED_LIFETIME_QUALIFICATION')
        assert row['status']!='FAILED'
        folder=ROOT/'raw'/name;passed=all(json.loads(p.read_text())['passed'] for p in folder.glob('*lifetime*.json'))
        assert passed;burst.append(dict(name=name,passed=passed))
    write(ROOT/'state/KV16_controlled_qualification_v1.json',dict(time=time.time(),passed=True,N8_state_and_teacher=True,N32_burst=burst))
    rows=[];budget=json.loads((ROOT/'state/budget.json').read_text())
    for axis,block in (('Layer',16),('Token',128),('Token',16),('Layer',128)):
        name=f'KVBlock_{axis}8_b{block}_N200_q2.35_s2026112089_v1'
        if time.time()>budget['last_large_dispatch_epoch']-1800:
            row=dict(name=name,status='NOT_RUN',reason='Reserve main confirmation and closeout budget.')
        else:
            extra={} if block==128 else dict(qualified_KV16_controlled=True,qualified_controlled_performance=False,
                qualified_async_runtime=False,runtime_qualification_tag='kv16_v1',token_teacher_gate_tag='kv16_v1')
            row=cell(name,axis,8,seed=2026112089,KV_block_tokens=block,phase='KV_BLOCK_ONE_FACTOR_SENSITIVITY',**extra)
        rows.append(dict(block=block,**row));write(ROOT/'analysis/KV_block_sensitivity_v1.json',dict(results=rows,evidence='MEASURED_ONLINE'))
        assert row['status']!='FAILED',name
    write(ROOT/'state/KV_block_sensitivity_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/KV_block_sensitivity_hold.json',dict(time=time.time(),error=repr(error)));raise
