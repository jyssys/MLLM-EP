"""Prepare paired Native confirmation only when the new observer qualifies."""
import json,random
from .common import *


def main():
    initialize();parent='graph_bound_ouro_L18432_O16_P2_mu130_pilot64_a2026111272_clean'
    source=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
    pool=json.loads((ROOT/'data/extended_natural_discovery_ouro_L18432.json').read_text())
    assert len(pool)==128 and len({p['prompt_id'] for p in pool})==128
    source.update(dataset='extended_natural_discovery_ouro_L18432.json',count=128,
        prompt_ids=[r['prompt_id'] for r in pool],request_caps=[16]*128,
        data_scope=dict(requests=128,distinct_prompt_ids=128,prefix_caching=False,
            scope='Disjoint natural discovery documents. No prompt expansion or filler.'))
    state='confirm_ouro_L18432_native_reference_state32'
    cfg=json.loads((ROOT/'configs/extended_ouro_L18432_native_reference_state8.json').read_text())
    # The state pool differs from the held-out serving documents by design.
    cfg.update(name=state,count=32,request_caps=[16]*32,arrival_offsets=[0.]*32,
        qualification_state_subset_reference='extended_ouro_L18432_native_reference_state8',
        prompt_ids=cfg['prompt_ids'][:32],state_hash_defer_until_close=False,state_pinned_slots=6,
        required_qualification='extended_ouro_L18432_native_reference_state8',
        phase='OURO_18432_NATIVE_N32_FULL_STATE_REFERENCE')
    # Native reference configs can list only their initial eight IDs.
    state_pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    if len(state_pool)<32:
        cfg.update(dataset=source['dataset'],prompt_ids=source['prompt_ids'][:32])
        cfg.pop('qualification_state_subset_reference',None)
        cfg['qualification_fresh_native']=True
    else:cfg['prompt_ids']=[r['prompt_id'] for r in state_pool[:32]]
    save(cfg);jobs=[state];plans=[]
    for seed in (2026111300,2026111301,2026111302):
        rng=random.Random(seed);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(source['arrival_rate']))
        names=[]
        for mode in ('clean','capture','repeat'):
            name=f'confirm_ouro_L18432_O16_P2_native_N128_a{seed}_{mode}';cfg=dict(source)
            cfg.update(name=name,arrival_seed=seed,arrival_offsets=offsets,
                required_qualification='graph_bound_ouro_L18432_observer_state8' if mode=='capture' else state,
                discovery_activity=mode=='capture',discovery_event_observer=mode=='capture',
                phase='OURO_18432_N128_PAIRED_NATIVE_CONFIRMATION',comparison_reference=None)
            save(cfg);names.append(name);jobs.append(name)
        path=ROOT/'state'/f'confirm_ouro18k_a{seed}_graph_plan.json'
        write(path,dict(clean_reference=names[0],jobs=names[1:]));plans.append(str(path))
    write(ROOT/'state/ouro_18k_confirmation_plan.json',dict(jobs=jobs,plans=plans,
        phase='CONDITIONAL_PREPARED_NOT_YET_RUN'))
    print(' '.join(jobs))


if __name__=='__main__':main()
