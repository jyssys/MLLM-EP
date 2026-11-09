"""Bounded online tests use prior measured costs while new calibration runs."""
from .prepare import *
from .analyze import metrics


def config_for(model,method):
    cfg=configuration(model,f'd0_{model}_{method}_burst8',8,state=False,method=method)
    cfg.update(arrival_offsets=[0.]*8,arrival_rate=None,arrival_seed=None,
        phase='N8_ACTUAL_ONLINE_BUDGET_INTERVENTION_NOT_FINAL_PERFORMANCE',
        arrival_pattern='SIMULTANEOUS_SUBMISSION_STRESS',minimum_P_pressure=2,
        qualification_allow_unpressured=False)
    if method=='native':return cfg
    costs=json.loads((PREVIOUS/'full_prompt_primitive_costs.json').read_text())
    old=next(r for r in costs if r['model']==model and r['remaining_loops']==1)
    one=old['streams']['decode']['maximum_ms'];bound=one+max(1.,.2*one)
    native=metrics(f'a_{model}_native_heavy24_s1')
    cfg.update(D_tile_tokens=cfg['length'],D_loop_quantum=1,
        minimum_quantum_ms=bound,empty_decode_budget_ms=200.,maximum_quantum_ms=200.,
        native_ITL_target_ms=native['frozen_SLO']['ITL_ms'],decode_reserve_ms=native['ITL_p99_ms'],
        decode_guard_ms=1.,guard_ms=max(1.,.2*one),
        remaining_cost_ms={str(n):bound*n for n in range(1,cfg['depth'])},
        require_whole_suffix_budget=True,separate_admission_budget=True,
        admission_window_multiplier=2.,empty_admission_window_multiplier=1.,
        admission_limit_ms=1000.,handoff_upper_ms=250. if model=='huginn' else 35.,
        max_continuations=1,negotiation_ttl_ms=10.,
        calibration_provenance=dict(path=str(PREVIOUS/'full_prompt_primitive_costs.json'),
            SHA256=digest(PREVIOUS/'full_prompt_primitive_costs.json'),maximum_one_loop_enclosure_ms=one,
            scope='Prior measured same-model full-prompt enclosure plus 20% or 1 ms. New campaign calibration remains required.'),
        estimator_scope='Absolute-deadline intervention candidate. Empty-window and future-opportunity assumptions are not safety guarantees. Native-relative tail protection is evaluated, not assumed.')
    return cfg


def main():
    prepare();names=[]
    for model in ('ouro','huginn'):
        for method in ('native','dynamic_loop'):
            cfg=config_for(model,method);save(cfg);names.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(path,queue[:i]+names+queue[i:])


if __name__=='__main__':main()
