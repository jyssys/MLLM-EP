"""Measure complete exact prefix/suffix shapes without token tiling."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import records


def build(batch):
    pool=json.loads((ROOT/'data/variable_qualification.json').read_text())
    grouped={t:[r for r in pool if r['length']==t] for t in (2048,4096,8192,12288)}
    data=[]
    for repeat in range(3):
        for length in grouped:
            for layers in (0,2,4,8,12):
                data.extend(grouped[length][(repeat*batch+j)%len(grouped[length])] for j in range(batch))
    name=f'length_cost_B{batch}_all_cuts_contexts_N{len(data)}_v1';dataset=name+'.json'
    path=ROOT/'data'/dataset
    if path.exists():assert json.loads(path.read_text())==data
    else:write(path,data)
    # The12k Source frontier uses the legal row budget. Actual frame sizes,
    # not the submission count, determine calibration buckets.
    return base(name,'FORCED_QUALIFICATION',count=len(data),cap=16,request_caps=[2]*len(data),
        dataset=dataset,length=12288,chunk=12288,token_budget=32000,variable_input=True,
        arrival_offsets=[0.]*len(data),arrival_rate=None,forced_layers=[0,2,4,8,12],
        P_group_size=batch,group_equal_input_lengths=True,warmup_batch_sizes=[1,2],
        calibration_serial=True,calibration_serial_group_size=batch,
        D_admission_batch=16,D_token_budget=128000,
        verified_ByteLevel_priming=True,direct_phase_norm=True,nixl_descriptor_sort=True,
        execution_events=True,verify_layer_execution=False,hidden_verify_hash=False,
        full_prompt_state_witness=False,clean_timing=False,
        phase='ISOLATED_CUDA_EVENT_ENCLOSURE_AND_HOST_COST_NOT_CLEAN_SERVING')


def main():
    assert json.loads((ROOT/'analysis/generic_runtime_full_state_gate.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/variable_input_full_state_gate.json').read_text())['passed']
    references={r['prompt_id']:r for r in records(ROOT/'raw/qual_variable_Full_reference_DP2_N32_v1/requests.jsonl')}
    try:
        for batch in (1,2,4):
            cfg=build(batch);name=save(cfg)
            if not execute(name,restore=False):raise RuntimeError('Length cost calibration failed')
            rows=records(ROOT/'raw'/name/'requests.jsonl')
            mismatches=[r['sample_id'] for r in rows if r['input_ids']!=references[r['prompt_id']]['input_ids']
                or r['output_ids']!=references[r['prompt_id']]['output_ids'][:len(r['output_ids'])]]
            gate=dict(name=name,count=len(rows),expected=cfg['count'],mismatches=mismatches,
                passed=len(rows)==cfg['count'] and not mismatches,evidence='CORRECTNESS_ONLY',
                scope='Independent full-model tokens at each real input length. Event calibration is not a Serving result.')
            write(ROOT/'analysis'/(name+'_output_gate.json'),gate);assert gate['passed'],gate
            print('LENGTH_COST_COMPLETE',name,flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
