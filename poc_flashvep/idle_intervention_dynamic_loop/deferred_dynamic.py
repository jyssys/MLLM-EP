"""Keep heavy hashes after early online decisions. Do not use this run for SLOs."""
from .early_online import *
from .policy import phase_candidates


def cell(model,count=8):
    cfg=config_for(model,'dynamic_loop')
    cfg.update(name=f'n{count}_{model}_actual_deferred_mixed_state{count}',count=count,
        arrival_offsets=[0.]*count,state_probe=True,verify_transfer=False,
        deferred_prompt_state_probe=True,dedicated_handoff_copy_stream=True,
        phase='ACTUAL_ONLINE_STATE_QUALIFICATION_NOT_CLEAN_DECODE_SAFETY',
        timeout=1800,startup_timeout=300,deep_graph_budget=64,P_prefill_graph_shapes=64,
        multiple_quanta_per_observed_deadline=True)
    pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    cfg.update(prompt_ids=[r['prompt_id'] for r in pool[:count]],request_caps=[cfg['cap']]*count)
    cfg['data_scope'].update(distinct_prompt_ids=count)
    if model=='ouro':
        cfg.update(intervention_method='dynamic_tile',phase_aware_costs=True,
            execution_candidates=phase_candidates(model),admission_window_multiplier=4.,
            D_candidate_widths=[32,128,256,4096],D_tile_tokens=128)
        cfg['minimum_quantum_ms']=min(c['cost_ms']+1. for c in cfg['execution_candidates'])
    return cfg


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        cfg=cell(model);save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
