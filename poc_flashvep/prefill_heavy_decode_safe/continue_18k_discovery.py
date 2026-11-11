"""Run bounded native discovery after the current profiler controls."""
import json,random,time
from .common import *
from .continue_measurement_alternatives import sequence
from .capacity_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/bounded_quiescent_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(10)
    recovery=json.loads((ROOT/'state/recovered_loop_primitive_plan.json').read_text())['jobs'][:2]
    sequence(recovery,'recovered_16k_loop_primitive_controller')
    plan=json.loads((ROOT/'state/extended_18k_plan.json').read_text())
    if sequence(plan['jobs'],'extended_18k_native_qualification_controller'):
        best=analyze(plan['capacity'])['strongest_finite_P_reference'];assert best
        rate=1.3*best['P_requests_per_s']['p50'];seed=2026111272;rng=random.Random(seed);offsets=[0.]
        for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
        original='extended_ouro_L24576_O16_P2_mu130_pilot64_a2026111272'
        cfg=json.loads((ROOT/'configs'/(original+'.json')).read_text())
        data=json.loads((ROOT/'data'/'extended_natural_calibration_ouro_L18432.json').read_text())
        pool=[data[i%len(data)] for i in range(64)];path=ROOT/'data'/'extended_ouro_L18432_calibration24_repeated_pilot64.json'
        if path.exists():assert json.loads(path.read_text())==pool
        else:write(path,pool)
        name='extended_ouro_L18432_O16_P2_mu130_pilot64_a2026111272'
        cfg.update(name=name,length=18432,state_capacity_tokens=18432,chunk=18432,
            token_budget=36864,D_token_budget=18432,P_prefill_graph_rows=36864,D_prefill_graph_rows=18432,
            dataset=path.name,prompt_ids=[r['prompt_id'] for r in pool],arrival_rate=rate,arrival_offsets=offsets,
            capacity_reference=plan['capacity'],required_qualification=plan['state8'],
            phase='NATURAL_18K_MARKER_FREE_NATIVE_DISCOVERY',
            selection_reason='The longer 24K Native arm failed frozen TPOT. Test the intermediate length under the same 20480-block pools and unmodified SLO.')
        save(cfg);write(ROOT/'state/extended_18k_pilot_plan.json',dict(jobs=[name],capacity=best,rate=rate))
        if sequence([name],'extended_18k_native_pilot_controller'):screen(name)
    write(ROOT/'state/extended_18k_controller_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
