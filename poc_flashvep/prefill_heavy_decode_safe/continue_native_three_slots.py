"""Accept a source-slot change only after a matched marker-free Native control."""
import json,random,time
from .common import *
from .continue_measurement_alternatives import sequence
from .capacity_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/ouro_18k_confirmation_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    subprocess.run([__import__('sys').executable,'-m','poc_flashvep.prefill_heavy_decode_safe.prepare_native_three_slots'],cwd=REPO,check=True)
    plan=json.loads((ROOT/'state/native_three_slots_plan.json').read_text())
    if sequence(plan['jobs'],'native_three_slot_qualification'):
        value=analyze(plan['capacity']);best=value['strongest_finite_P_reference'];assert best
        pool=json.loads((ROOT/'data/natural_calibration_huginn_L3500.json').read_text());rows=[pool[i%24] for i in range(64)]
        path=ROOT/'data/native_slots_huginn_L3500_calibration24_repeated64.json'
        if path.exists():assert json.loads(path.read_text())==rows
        else:write(path,rows)
        rate=1.15*best['P_requests_per_s']['p50'];seed=2026111320;rng=random.Random(seed);offsets=[0.]
        for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
        jobs=[]
        for slots in (2,3):
            cfg=json.loads((ROOT/'configs/reserve_huginn_L3900_O16_P2_mu115_N64_a2026111230.json').read_text())
            name=f'native_slots{slots}_huginn_L3500_O16_pilot64_a{seed}'
            cfg.update(name=name,length=3500,state_capacity_tokens=3500,chunk=3500,P_slots=slots,kv_blocks=23296,
                token_budget=slots*3500,D_token_budget=3500,P_prefill_graph_rows=slots*3500,D_prefill_graph_rows=3500,
                dataset=path.name,prompt_ids=[r['prompt_id'] for r in rows],arrival_seed=seed,arrival_rate=rate,
                arrival_offsets=offsets,required_qualification=plan[f'state{slots}'],capacity_reference=plan['capacity'],
                comparison_reference=None,phase='PAIRED_NATIVE_TWO_VS_THREE_SOURCE_SLOTS',
                data_scope=dict(requests=64,distinct_prompt_ids=24,prefix_caching=False,
                    scope='Calibration-only matched pilot. Not an independent-document generalization test.'))
            save(cfg);jobs.append(name)
        write(ROOT/'state/native_three_slots_pilot_plan.json',dict(jobs=jobs,rate=rate))
        if sequence(jobs,'native_three_slot_paired_pilot'):
            results=[screen(n,reference=jobs[0]) for n in jobs]
            a,b=[r['metrics'] for r in results]
            decision=dict(completed_QPS_ratio=b['completed_QPS']/a['completed_QPS'],
                TTFT_p95_ratio=b['TTFT_p95_ms']/a['TTFT_p95_ms'],
                TPOT_p95_ratio=b['TPOT_p95_ms']/a['TPOT_p95_ms'],ITL_p99_ratio=b['ITL_p99_ms']/a['ITL_p99_ms'],
                scope='One marker-free Native pair. No Loop or physical-idle result. Reject performance regressions instead of calling them a new compute bottleneck.')
            write(ROOT/'analysis/native_three_slot_pilot.json',dict(results=results,comparison=decision))
    write(ROOT/'state/native_three_slots_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
