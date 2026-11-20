"""Check the frozen context boundary without selecting on performance."""
import hashlib
import json
import time
from .common import ROOT,digest,save,write
from .qualification import configuration,records
from .run import execute


def main():
    full=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    chosen=[min(full,key=lambda r:r['input_tokens']),
        min(full,key=lambda r:abs(r['input_tokens']-8192)),max(full,key=lambda r:r['input_tokens'])]
    dataset='context_boundary3.json';path=ROOT/'data'/dataset
    if path.exists():assert json.loads(path.read_text())==chosen
    else:write(path,chosen)
    design=ROOT/'configs/frozen_context_boundary_state_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),dataset=dataset,SHA256=digest(path),N=3,cut=12,
            input_lengths=[r['input_tokens'] for r in chosen],selection='Shortest, nearest8192, longest frozen prompt. No output/performance filter.',
            execution='BI0 Compile0, async on, full decode graph, KV128, B8192/S1024.',
            scope='Additional state and lifetime boundary check. Not a performance sample or held-out workload.'))
    while not (ROOT/'state/axis_qualification_fast128_v3_complete.json').exists():
        for marker in ('STOP_REQUESTED.json','async_qualification_state_hold.json','axis_state_mismatch_fast128_v3.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    jobs={}
    for label in ('Standard','Layer','Token','TokenTeacher'):
        name=f'qual_ContextBoundary_{label}_Last12_N3_BI0_v1';jobs[label]=name
        single=label in ('Standard','TokenTeacher');axis=label if not single else None
        extra=dict(axis_kind=axis,axis_cut=12,axis_state_capture=True,async_qualification_only=True) if axis else {}
        if label=='TokenTeacher':extra['token_teacher_cut']=12
        cfg=configuration(name,axis or 'Native',dataset,count=3,
            gpus=[1] if single else [0,1],reference_single=single,state_capture=single,
            numerical_probe=True,numerical_probe_sync=True,execution_family='MATCHED_NO_COMPILE',
            async_scheduling=True,KV_block_tokens=128,restore_official_UCX_init=True,
            phase='CONTEXT_BOUNDARY_CORRECTNESS',**extra)
        assert execute(save(cfg)),name
    import torch
    torch.set_num_threads(1)
    results=[]
    for axis,label in (('Layer','Standard'),('Token','TokenTeacher')):
        teacher_folder=ROOT/'raw'/jobs[label];actual_folder=ROOT/'raw'/jobs[axis]
        teacher=json.loads((teacher_folder/'prompt_state_single.json').read_text())['requests']
        states={role:json.loads((actual_folder/f'axis_events_{role}.json').read_text())['states'] for role in ('prefill','decode')}
        expected={r['sample_id']:r['output_ids'] for r in records(teacher_folder/'requests.jsonl')}
        actual={r['sample_id']:r['output_ids'] for r in records(actual_folder/'requests.jsonl')}
        lifetime={p.name:json.loads(p.read_text())['passed'] for p in actual_folder.glob('*lifetime*.json')}
        for i,row in enumerate(chosen):
            rid=f'q{i}';hidden=torch.load(teacher_folder/f'numerical_single_{rid}.pt',weights_only=True)['hidden_last']
            hidden_sha=hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            results.append(dict(axis=axis,input_tokens=row['input_tokens'],
                full_KV_exact=teacher[rid]['KV']==states['decode'][rid]['KV'],
                hidden_exact=hidden_sha==states['decode'][rid]['last_hidden_SHA256'],
                valid_import_exact=states['prefill'][rid]['KV']==states['decode']['import_'+rid]['KV'],
                greedy_exact=expected[i]==actual[i],lifetime_passed=bool(lifetime) and all(lifetime.values())))
    passed=all(all(r[k] for k in ('full_KV_exact','hidden_exact','valid_import_exact','greedy_exact','lifetime_passed')) for r in results)
    write(ROOT/'analysis/BI0_context_boundary_state_v1.json',dict(passed=passed,results=results,jobs=jobs,evidence='CORRECTNESS_ONLY'))
    assert passed
    write(ROOT/'state/context_boundary_state_complete.json',dict(time=time.time(),passed=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/context_boundary_state_hold.json',dict(time=time.time(),error=repr(error)));raise
