"""Test a legal three-request source without weakening Native deliberately."""
import json
from .common import *


def main():
    initialize()
    for split in ('calibration','discovery'):
        values=json.loads((ROOT/'data'/f'natural_{split}_huginn_L3900.json').read_text())
        rows=[dict(r,length=3500,input_ids=dict(huginn=r['input_ids']['huginn'][:3500])) for r in values]
        path=ROOT/'data'/f'natural_{split}_huginn_L3500.json'
        if path.exists():assert json.loads(path.read_text())==rows
        else:write(path,rows)
    source=json.loads((ROOT/'configs/reserve_huginn_L3900_native_reference_state8.json').read_text())
    jobs=[];reference=None
    for slots in (2,3):
        name=f'native_slots{slots}_huginn_L3500_state8';cfg=dict(source)
        cfg.update(name=name,length=3500,state_capacity_tokens=3500,chunk=3500,
            P_slots=slots,kv_blocks=23296,token_budget=slots*3500,D_token_budget=3500,
            P_prefill_graph_rows=slots*3500,D_prefill_graph_rows=3500,
            dataset='natural_calibration_huginn_L3500.json',
            phase='NATIVE_LEGAL_THREE_REQUEST_GEOMETRY_STATE_CHECK',
            qualification_fresh_native=reference is None,required_qualification=reference,
            comparison_reference=None,state_hash_defer_until_close=False,state_pinned_slots=6)
        for k in ('campaign_state_reference','qualification_output_reference','qualification_state_subset_reference'):cfg.pop(k,None)
        if reference:cfg['campaign_state_reference']=reference
        save(cfg);jobs.append(name)
        if slots==2:reference=name
    cfg=json.loads((ROOT/'configs/edge_capacity_huginn_L4080_B12_start1.json').read_text())
    name='native_slots3_capacity_huginn_L3500_B123_start1'
    cfg.update(name=name,length=3500,state_capacity_tokens=3500,chunk=3500,capacity=3,slots=3,
        P_slots=3,token_budget=10500,D_token_budget=3500,P_prefill_graph_rows=10500,
        D_prefill_graph_rows=3500,dataset='natural_calibration_huginn_L3500.json',
        required_qualification=jobs[-1],jobs=[dict(batch=b,cap=16,repeats=3) for b in (1,2,3)],
        phase='NATIVE_THREE_REQUEST_MATCHED_GRAPH_CAPACITY')
    save(cfg);jobs.append(name)
    write(ROOT/'state/native_three_slots_plan.json',dict(jobs=jobs,capacity=name,state2=reference,state3=jobs[1],
        scope='3500-token input permits three full source allocations within 23296 blocks. Three active source requests may improve Native overlap and Decode batching. Preserve the HBM guard. Reject this configuration if it weakens the paired two-slot Native baseline. No Loop migration occurs.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
