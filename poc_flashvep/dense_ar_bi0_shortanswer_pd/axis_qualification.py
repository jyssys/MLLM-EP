"""Test new BI0 continuations before any online serving claim."""
import json
import hashlib
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute
from .quality import quality_gate


def main(tag='v1',cuts=(1,4,12),block=16,restore=False,asynchronous=False):
    declaration=ROOT/'configs'/f'frozen_BI0_axis_state_qualification_{tag}.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),dataset='qualification8.json',cuts=list(cuts),
            actual_batch='Serial request admission. Real variable input can require multiple B8192 chunks.',
            B=8192,S=1024,KV_block=block,KV_GiB=42,BI=False,restore_official_UCX_init=restore,
            execution=f'Compile0, full decode CUDA graph, async={asynchronous}, same matched reference.',
            scope='Correctness only. No existing decode is interrupted. No online offload or throughput qualification.',
            token_scope='Completed-prefix Full36 KV then Full36 causal suffix. Prefix16 alignment can cause recorded row mismatch.'))
    reference=f'qual_Standard_NoCompile_ShortAnswer_N8_BI0_{tag}'
    extra=dict(KV_block_tokens=block,restore_official_UCX_init=restore,
        async_qualification_only=asynchronous,numerical_probe_sync=asynchronous) if tag!='v1' else {}
    cfg=configuration(reference,'Native','qualification8.json',reference_single=True,gpus=[1],
        execution_family='MATCHED_NO_COMPILE',async_scheduling=asynchronous,numerical_probe=True,**extra)
    assert execute(save(cfg)),reference
    teacher=json.loads((ROOT/'raw'/reference/'prompt_state_single.json').read_text())['requests']
    import torch
    teacher_hidden={f'q{i}':hashlib.sha256(torch.load(ROOT/'raw'/reference/f'numerical_single_q{i}.pt',weights_only=True)
        ['hidden_last'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for i in range(8)}
    teacher_outputs={r['sample_id']:r['output_ids'] for r in
        [json.loads(line) for line in (ROOT/'raw'/reference/'requests.jsonl').read_text().splitlines()]}
    results=[]
    for cut in cuts:
        for axis in ('Layer','Token'):
            name=f'qual_{axis}_Last{cut}_ShortAnswer_N8_BI0_{tag}'
            cfg=configuration(name,axis,'qualification8.json',state_capture=False,
                numerical_probe=True,axis_kind=axis,axis_cut=cut,
                execution_family='MATCHED_NO_COMPILE',async_scheduling=asynchronous,**extra)
            if not execute(save(cfg)):
                write(ROOT/'state'/f'axis_qualification_{tag}_failed.json',dict(time=time.time(),name=name,
                    scope='Preserve raw failure. Do not dispatch online performance for this arm.'))
                return
            p=json.loads((ROOT/'raw'/name/'axis_events_prefill.json').read_text())['states']
            d=json.loads((ROOT/'raw'/name/'axis_events_decode.json').read_text())['states']
            rows=[]
            for index in range(8):
                rid=f'q{index}';source=p[rid];dest=d[rid];imported=d['import_'+rid]
                rows.append(dict(request_id=rid,imported_KV_exact=source['KV']==imported['KV'],
                    full_KV_equals_full_query_reference=dest['KV']==teacher[rid]['KV'],
                    final_hidden_equals_full_query_reference=dest['last_hidden_SHA256']==teacher_hidden[rid]))
            quality=quality_gate(name)
            output_rows=[json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
            strict_matched_reference=sum(r['output_ids']==teacher_outputs[r['sample_id']] for r in output_rows)
            lifetime={p.name:json.loads(p.read_text())['passed'] for p in (ROOT/'raw'/name).glob('*lifetime*.json')}
            passed=all(r['imported_KV_exact'] for r in rows) and bool(lifetime) and all(lifetime.values())
            if axis=='Layer':passed=passed and all(r['full_KV_equals_full_query_reference'] and r['final_hidden_equals_full_query_reference'] for r in rows)
            if axis=='Layer':passed=passed and strict_matched_reference==8
            result=dict(name=name,cut=cut,axis=axis,passed=passed,rows=rows,quality=quality,lifetime=lifetime,
                strict_matched_NoCompile_reference=strict_matched_reference,
                evidence='CORRECTNESS_ONLY',scope='Layer uses same query geometry. Token changes query geometry, so full-reference strict equality is reported separately from exact prefix transfer and task quality. '
                'N8 task quality is disclosed but cannot qualify or reject full200 task noninferiority by itself.')
            results.append(result);write(ROOT/'analysis'/f'BI0_axis_state_qualification_{tag}.json',dict(results=results))
            if not passed:
                write(ROOT/'state'/f'axis_state_mismatch_{tag}.json',dict(time=time.time(),name=name,status='BLOCKED_CORRECTNESS'))
                return
    flag='axis_qualification_complete.json' if tag=='v1' else f'axis_qualification_{tag}_complete.json'
    write(ROOT/'state'/flag,dict(time=time.time(),complete=True,qualified_cuts=list(cuts),KV_block=block))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/axis_qualification_hold.json',dict(time=time.time(),error=repr(error)));raise
