"""Separate a buffer-capacity failure from unsupported mixed-layer execution."""
import json
import time
from .common import ROOT,write
from .campaign import cell


def prefix_check(name,count):
    folder=ROOT/'raw'/name
    source=json.loads((folder/'axis_events_prefill.json').read_text())['states']
    destination=json.loads((folder/'axis_events_decode.json').read_text())['states']
    rows=[dict(request_id=f'q{i}',imported_prefix_KV_exact=source[f'q{i}']['KV']==destination[f'import_q{i}']['KV']) for i in range(count)]
    return dict(rows=rows,passed=all(r['imported_prefix_KV_exact'] for r in rows),
        scope='Exact source/import prefix bytes. This does not replace a full C3 same-geometry hidden/greedy teacher.')


def main():
    path=ROOT/'configs/frozen_C3_bridge_followup_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),cut=8,KV_block=128,B=8192,S=1024,
            reason='Initial C3 Layer N8 failed because the destination hidden pool still had the C0 one-slot capacity.',
            intervention='Increase the destination pool to8 slots and retain the source16-slot pool. Keep KV42GiB and the normal continuous scheduler.',
            rule='Correctness-only. Never remove the mixed-prefill/decode assertion. If it fails, stop Layer bridge expansion. '
            'A successful N8 process only qualifies its observed shapes. N32 removes intrusive state hashing to exercise natural mixed scheduling.',
            scope='No failed C3 timing enters the main performance comparison. C3 Token N200 remains correctness-only until a full C3 teacher gate exists.'))
    initial='C3Qualification_Layer8_N8_q8_BI0_v1'
    error=(ROOT/'raw'/initial/'decode.log').read_text()
    assert 'HIDDEN_CREDIT_EXHAUSTED' in error
    rows=[]
    name='C3Qualification_Layer8_N8_q8_dst8_BI0_v2'
    row=cell(name,'Layer',8,dataset='qualification8.json',count=8,rate=8,seed=2026112088,
        controlled_pipeline=False,correctness_only=True,continuous_qualification_only=True,
        qualified_controlled_performance=False,axis_state_capture=True,hidden_source_slots=16,hidden_destination_slots=8,
        phase='C3_DESTINATION_BUFFER_CAPACITY_CORRECTION')
    rows.append(row)
    if row['status']!='FAILED':
        row['prefix_transfer_gate']=prefix_check(name,8);assert row['prefix_transfer_gate']['passed']
        name='C3Qualification_Layer8_N32_q8_dst8_BI0_v1'
        rows.append(cell(name,'Layer',8,dataset='qualification32.json',count=32,rate=8,seed=2026112093,
            controlled_pipeline=False,correctness_only=True,continuous_qualification_only=True,
            qualified_controlled_performance=False,axis_state_capture=False,hidden_source_slots=32,hidden_destination_slots=8,
            phase='C3_UNPERTURBED_MIXED_SHAPE_CORRECTNESS_ONLY'))
    write(ROOT/'analysis/C3_layer_bridge_followup_v1.json',dict(results=rows,evidence='CORRECTNESS_ONLY',
        scope='Buffer correction is separate from mixed-row model support. A process failure is not a layer-axis impossibility theorem.'))
    token='C3Qualification_Token8_N8_q8_BI0_v1'
    gate=prefix_check(token,8);assert gate['passed']
    name='C3Qualification_Token8_N32_q8_BI0_v1'
    result=cell(name,'Token',8,dataset='qualification32.json',count=32,rate=8,seed=2026112093,
        controlled_pipeline=False,correctness_only=True,continuous_qualification_only=True,
        qualified_controlled_performance=False,axis_state_capture=False,phase='C3_TOKEN_STRUCTURAL_LIFETIME_CORRECTNESS_ONLY')
    token_rows=[result]
    if result['status']!='FAILED':
        name='C3Qualification_Token8_N200_q2.35_s2026112071_BI0_v1'
        token_rows.append(cell(name,'Token',8,seed=2026112071,controlled_pipeline=False,
            correctness_only=True,continuous_qualification_only=True,qualified_controlled_performance=False,
            axis_state_capture=False,phase='C3_TOKEN_TASK_QUALITY_CORRECTNESS_ONLY'))
    write(ROOT/'analysis/C3_token_bridge_followup_v1.json',dict(N8_import_gate=gate,results=token_rows,
        evidence='CORRECTNESS_ONLY',scope='Normal continuous full36 math with completed-prefix KV continuation. '
        'Full C3 same-geometry teacher validation remains NOT_RUN. Timings are diagnostic, not a qualified C3 method-performance victory.'))
    write(ROOT/'state/C3_bridge_followup_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/C3_bridge_followup_hold.json',dict(time=time.time(),error=repr(error)));raise
