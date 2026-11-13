"""Build immutable matched configurations from qualified predecessor recipes."""
import random
from .common import *


def offsets(count,rate,seed):
    rng=random.Random(seed);values=[0.]
    for _ in range(count-1):values.append(values[-1]+rng.expovariate(rate))
    return values


def base(name,arm='Native',count=64,cap=64,rate=4.8,seed=2026111321,dataset='discovery.json',**changes):
    template='heldout_O16_N128_s2026111341_'+('COST' if arm not in ('Native','Colocated') else 'Native')+'_v1'
    c=json.loads((PREVIOUS/'configs'/(template+'.json')).read_text())
    c.update(name=name,count=count,cap=cap,dataset=dataset,request_caps=[cap]*count,
        arrival_offsets=offsets(count,rate,seed),arrival_rate=rate,arrival_seed=seed,
        phase_stable_norm=True,phase='NEW_CAMPAIGN_DISCOVERY',evaluation_arm=arm,
        physical_activity=False,execution_events=False,full_prompt_state_witness=False,numeric_probe=False,
        clean_timing=True,timeout=900,wall_timeout=1400)
    if arm=='Colocated':
        c.update(arm='DP2',module='native',policy=None,reuse_prefill_token=False,
            layer_mode=None,hidden_transport=None,partial_KV_wire=False,separate_suffix_runner=False,
            colocated_cohorts=False,cohort_barrier=False,serialize_D_imports=False,
            atomic_prompt=False,chunk=8000,token_budget=8192)
    elif arm=='Native':c.update(module='native',policy=None)
    else:
        c.update(module='dynamic_native',policy='REVISED' if arm=='Revised' else arm,
            cost_model='calibrated_cost_model_previous.json')
        if arm=='Revised':
            c.update(cost_model='revised_cost_model_v1.json',safety_margin_ms=110.,
                decode_prediction='BATCH_LOOKUP',decode_prediction_margin_ms=2.,
                layer_candidates=[2,4,8,12],queue_benefit_weight=1.)
    c.update(changes);return c


def qualification(name,count):
    template=f'qual_forced_mixed_Last024812_N{count}_v1'
    c=json.loads((PREVIOUS/'configs'/(template+'.json')).read_text())
    c.update(name=name,phase_stable_norm=True,phase='NEW_RUNTIME_EXACT_MIXED_CUT_QUALIFICATION')
    return c


def initial_plan():
    assert (ROOT/'state/continuous_N32_qualified.json').exists()
    jobs=[qualification(f'exact_normfix_mixed_Last024812_N{n}_v1',n) for n in (8,32)]
    # A200-token reference qualifies the shorter prefixes too. Its outputs are
    # a correctness reference, not interchangeable performance observations.
    jobs.append(base('baseline_Native_O200_N64_q4p8_s21_v1',cap=200))
    for budget in (8192,2048,1024,512):
        jobs.append(base(f'baseline_Continuous_budget{budget}_O200_N64_q4p8_s21_v1',arm='Colocated',cap=200,
            token_budget=budget,chunk=min(8000,budget)))
    for c in jobs:save(c)
    write(ROOT/'state/initial_baseline_plan.json',dict(time=time.time(),jobs=[c['name'] for c in jobs],
        scope='Discovery only. Different token budgets do not change the reference output or SLO.'))
    print(' '.join(c['name'] for c in jobs))


if __name__=='__main__':initial_plan()
