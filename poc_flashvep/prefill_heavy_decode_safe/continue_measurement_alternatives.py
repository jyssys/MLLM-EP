"""Run justified Native measurement alternatives in one exclusive queue."""
import json,random,sys,time
from .common import *
from .capacity_analysis import analyze as capacity
from .host_screen import screen


def sequence(names,label):
    assert time.time()<LONG_START_CUTOFF
    log=ROOT/'logs'/(label+'.log')
    with log.open('a') as f:
        code=subprocess.run([sys.executable,'-u','-m','poc_flashvep.prefill_heavy_decode_safe.sequence',*names],
            cwd=REPO,stdout=f,stderr=subprocess.STDOUT).returncode
    write(ROOT/'state'/(label+'_receipt.json'),dict(returncode=code,names=names,time=time.time()))
    print(label,code,flush=True);return code==0


def main():
    initialize()
    while not json.loads((ROOT/'state/controller_serialization.json').read_text()).get('closed'):
        assert time.time()<LONG_START_CUTOFF;time.sleep(10)
    if sequence(json.loads((ROOT/'state/quiescent_capture_plan.json').read_text())['jobs'],'quiescent_native_controller'):
        from .quiescent_analysis import main as quiescent
        quiescent()
    if sequence(json.loads((ROOT/'state/late_capture_plan.json').read_text())['jobs'],'late_native_controller'):
        from .profile_controls import analyze
        analyze(ROOT/'state/late_capture_plan.json')
    # Primitive qualification does not constitute Phase 2 Serving.
    costs=json.loads((ROOT/'state/long_loop_cost_plan.json').read_text())['jobs'][:3]
    sequence(costs,'long_loop_primitive_controller')
    dataset_plan=ROOT/'state/extended_24k_plan.json'
    if dataset_plan.exists():
        plan=json.loads(dataset_plan.read_text())
        if sequence(plan['jobs'],'extended_24k_native_qualification_controller'):
            best=capacity(plan['capacity'])['strongest_finite_P_reference'];assert best
            rate=1.3*best['P_requests_per_s']['p50'];seed=2026111272;rng=random.Random(seed);offsets=[0.]
            for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
            cfg=json.loads((ROOT/'configs'/'memory_ouro_L16384_O16_P2_mu115_N64_a2026111230.json').read_text())
            calibration=json.loads((ROOT/'data'/'extended_natural_calibration_ouro_L24576.json').read_text())
            pool=[calibration[i%len(calibration)] for i in range(64)]
            path=ROOT/'data'/'extended_ouro_L24576_calibration24_repeated_pilot64.json'
            if path.exists():assert json.loads(path.read_text())==pool
            else:write(path,pool)
            name='extended_ouro_L24576_O16_P2_mu130_pilot64_a2026111272'
            cfg.update(name=name,length=24576,state_capacity_tokens=24576,chunk=24576,
                kv_blocks=20480,D_kv_blocks=20480,token_budget=49152,D_token_budget=24576,
                P_prefill_graph_rows=49152,D_prefill_graph_rows=24576,
                dataset=path.name,prompt_ids=[r['prompt_id'] for r in pool],
                arrival_seed=seed,arrival_rate=rate,arrival_offsets=offsets,
                capacity_reference=plan['capacity'],required_qualification=plan['state8'],comparison_reference=None,
                phase='NATURAL_24K_MARKER_FREE_NATIVE_DISCOVERY',
                data_scope=dict(distinct_prompt_ids=24,requests=64,prefix_caching=False,
                    scope='Only calibration documents repeat in this pilot. The separate 128-document pool remains unseen at this geometry. No independent-text claim for this pilot.'),
                selection_reason='Native L16384 provides a valid host pressure candidate, but no qualified long full-loop window. Test a legal longer context after fresh exact-state qualification and finite calibration. No SLO change.')
            save(cfg);write(ROOT/'state/extended_24k_pilot_plan.json',dict(jobs=[name],rate=rate,capacity=best))
            if sequence([name],'extended_24k_native_pilot_controller'):screen(name)
    write(ROOT/'state/measurement_alternatives_controller_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()
