"""Separate KV population effects with a real long/short prompt mixture."""
import copy,json,random
from .common import *


def dataset(kind):
    source=ROOT/'data'/f'natural_{kind}_huginn_L3900.json'
    rows=copy.deepcopy(json.loads(source.read_text()));rng=random.Random(2026111251)
    assignment=[3900,1024]*(len(rows)//2)
    if len(rows)%2:assignment.append(3900)
    # Reserve a balanced first calibration cohort. Shuffle the discovery mix.
    if kind=='discovery':rng.shuffle(assignment)
    for row,length in zip(rows,assignment):
        row['input_ids']['huginn']=row['input_ids']['huginn'][:length]
        row['discovery_mixture']=dict(maximum_input=3900,actual_input=length,
            original_prefix_source_SHA256=digest(source),assignment_seed=2026111251,
            scope='Original document prefix. No filler, extension, duplicated prompt ID, or model change.')
    name=f'mixed_{kind}_huginn_L3900_L1024_p50.json';path=ROOT/'data'/name
    if path.exists():assert json.loads(path.read_text())==rows
    else:write(path,rows)
    return name,rows


def main():
    initialize();calibration,cal=dataset('calibration');discovery,data=dataset('discovery')
    assert len({r['prompt_id'] for r in data})==len(data)==128
    source='capacity_huginn_L3900_O16_B124_start1'
    cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text());name='mixed_capacity_huginn_L3900_L1024_p50_B124_start1'
    cfg.update(name=name,dataset=calibration,prompt_ids=[r['prompt_id'] for r in cal[:4]],
        prompt_token_lengths=[len(r['input_ids']['huginn']) for r in cal[:4]],kv_blocks=24064,
        token_budget=9848,P_prefill_graph_rows=9848,
        phase='MIXED_LENGTH_FINITE_NATIVE_CAPACITY',
        selection_reason='L3900 homogeneous Native has frequent D KV refusals and no growing audited P backlog. A real length mixture changes reservation geometry and service variability. This is a new workload, not recovery of the frozen anchor.')
    save(cfg);capacity=name
    cfg=json.loads((ROOT/'configs'/'reserve_huginn_L3900_native_reference_state8.json').read_text())
    name='mixed_huginn_L3900_L1024_p50_native_reference_state8'
    cfg.update(name=name,dataset=discovery,prompt_ids=[r['prompt_id'] for r in data[:8]],
        prompt_token_lengths=[len(r['input_ids']['huginn']) for r in data[:8]],
        qualification_fresh_native=True,capacity_reference=capacity,
        phase='MIXED_REAL_PREFIX_NATIVE_FULL_STATE_REFERENCE')
    cfg.pop('qualification_output_reference',None);save(cfg)
    write(ROOT/'state/mixed_native_discovery_plan.json',dict(capacity=capacity,state8=name,
        serving_prepared_after='Calibrate the mixture rate before fixing new Poisson arrivals.',
        distinct_prompt_ids=128,input_lengths=[1024,3900],Output=16,long_fraction=.5,
        scope='Discovery only. Native R32, legal KV pools, frozen SLO, and all mathematical work remain unchanged.'))
    print(capacity,name)


if __name__=='__main__':main()
