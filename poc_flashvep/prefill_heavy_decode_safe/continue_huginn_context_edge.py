"""Select the final legal input candidate from measured capacity, not idle guesses."""
import json,random,time,fcntl
from .common import *
from .continue_measurement_alternatives import sequence
from .capacity_analysis import analyze as capacity
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/huginn_resumed_physical_confirmation_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    # Dataset tokenization must not contend with a clean measurement.
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources();assert not owner.processes();owner.restore(list(range(8)))
        subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-m',
            'poc_flashvep.prefill_heavy_decode_safe.prepare_huginn_context_edge'],cwd=REPO,check=True)
    plan=json.loads((ROOT/'state/huginn_context_edge_plan.json').read_text())
    if sequence(plan['jobs'],'huginn_context_edge_qualification'):
        best=capacity(plan['capacity'])['strongest_finite_P_reference'];assert best
        cfg=json.loads((ROOT/'configs/reserve_huginn_L3900_O16_P2_mu115_N64_a2026111230.json').read_text())
        pool=json.loads((ROOT/'data/natural_calibration_huginn_L4080.json').read_text())
        values=[pool[i%24] for i in range(64)];data=ROOT/'data/edge_huginn_L4080_calibration24_repeated64.json'
        if data.exists():assert json.loads(data.read_text())==values
        else:write(data,values)
        rate=1.2*best['P_requests_per_s']['p50'];seed=2026111310;rng=random.Random(seed);offsets=[0.]
        for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
        name=f'edge_huginn_L4080_O16_P2_mu120_pilot64_a{seed}'
        cfg.update(name=name,length=4080,state_capacity_tokens=4080,chunk=4080,
            token_budget=8160,D_token_budget=4080,P_prefill_graph_rows=8160,D_prefill_graph_rows=4080,kv_blocks=23296,
            dataset=data.name,prompt_ids=[r['prompt_id'] for r in values],arrival_rate=rate,
            arrival_seed=seed,arrival_offsets=offsets,capacity_reference=plan['capacity'],
            required_qualification=plan['state8'],comparison_reference=None,
            phase='NATURAL_CONTEXT_EDGE_CAPACITY_GUIDED_NATIVE_PILOT',
            selection_reason='Longer legal input raises Prefill work relative to sixteen Decode outputs. This tests whether the fragmented 3072-token gaps are workload-specific. No Decode-SLO relaxation.',
            data_scope=dict(requests=64,distinct_prompt_ids=24,prefix_caching=False,
                scope='Repeated calibration pilot only. The separate 128 discovery documents remain untested at this geometry.'))
        save(cfg);write(ROOT/'state/huginn_context_edge_pilot_plan.json',dict(jobs=[name],rate=rate))
        if sequence([name],'huginn_context_edge_pilot'):screen(name)
    write(ROOT/'state/huginn_context_edge_controller_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
