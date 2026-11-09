"""Select empirical execution limits before held-out confirmation."""
from .early_online import *
from .costs import summarize


def guarded(model,width):
    name=f'p_{model}_tile{width}_loops1_serial8'
    value=summarize(name)
    assert value is not None,'Calibration must complete before policy preparation'
    rows=[g for g in value['groups'] if g['role']=='decode' and g['tokens']==width]
    assert rows
    return max(g['p99_plus_guard_ms'] for g in rows)


def common_pilot(model,name,method='dynamic_loop',count=24,seed=2026110801):
    cfg=configuration(model,name,count,seed=seed,method=method)
    if model=='huginn':
        cfg['D_kv_blocks']=23552
        cfg['workspace_reserve_note']='Multi-shape startup exhausted the original 26752-block D pool envelope. Reserve 3200 blocks of HBM for exact graph and Coda workspace. Keep this pool identical across all new compared methods.'
    native=metrics(f'a_{model}_native_heavy24_s1')
    one=guarded(model,cfg['length'])
    cfg.update(deep_graph_budget=64,P_prefill_graph_shapes=64,
        dedicated_handoff_copy_stream=True,minimum_P_pressure=2,
        qualification_allow_unpressured=False,D_loop_quantum=1,
        D_tile_tokens=cfg['length'],empty_decode_budget_ms=200.,empty_quantum_cap_ms=200.,
        maximum_quantum_ms=16.,native_ITL_target_ms=1.05*native['ITL_p99_ms'],
        decode_reserve_ms=native['ITL_p50_ms'],decode_prediction_guard_ms=1.,
        minimum_decode_reserve_ms=native['ITL_p50_ms'],adaptive_decode_reserve=True,
        minimum_decode_history=16,decode_guard_ms=1.,guard_ms=1.,
        feedback_ITL_limit_ms=1.05*native['ITL_p99_ms'],
        remaining_cost_ms={str(n):one*n for n in range(1,cfg['depth'])},
        minimum_quantum_ms=one,require_whole_suffix_budget=True,
        separate_admission_budget=True,admission_window_multiplier=4.,
        empty_admission_window_multiplier=1.,admission_limit_ms=1000.,
        handoff_upper_ms=100. if model=='huginn' else 35.,max_continuations=1,
        negotiation_ttl_ms=10.,
        phase='EMPIRICAL_QUANTUM_SELECTION_NOT_HELD_OUT_CONFIRMATION',
        estimator_scope='No current Decode means no outstanding observed ITL deadline. The 200 ms policy cap is not a prediction of a physical idle gap. Active reserve starts at the measured Native median. Online batch-conditioned p95 and a guard replace it after 16 samples. Actual interventions must validate tail protection. Future arrivals remain unknown.')
    return cfg


def probe(model,width,seed=2026110801):
    cfg=common_pilot(model,f'i_{model}_tile{width}_lastloop_seed{seed}')
    cfg.update(boundary_checkpoints=[cfg['depth']-1],D_tile_tokens=width,
        minimum_quantum_ms=guarded(model,width),require_whole_suffix_budget=False,
        probe_scope='Actual last-loop serving continuation. Not an independent D-local request. Complete prompt execution may span many yielded tile submissions.')
    return cfg


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        # One shared trace first screens all legal widths. Confirmation is separate.
        for arm in ('native','zero'):
            cfg=common_pilot(model,f'i_{model}_{arm}_control24',arm)
            if arm=='native':
                for key in ('native_path','feature_native_wire_v5'):cfg[key]=False
                cfg['atomic_prompt']=True
            save(cfg);jobs.append(cfg['name'])
        for width in (16,32,64,128,256,4096 if model=='ouro' else 2048):
            cfg=probe(model,width);save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';write(p,json.loads(p.read_text())+jobs)


if __name__=='__main__':main()
