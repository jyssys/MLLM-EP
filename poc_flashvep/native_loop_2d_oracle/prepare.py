"""Build immutable matched contracts from the last qualified serving epoch."""
import copy
import random
from .common import *

RATES = {'ouro': dict(low=1.23460875, medium=3.4569045, heavy=6.356),
         'huginn': dict(low=.37903355455, medium=1.06129395274, heavy=1.971)}


def configuration(model, path, count, name, state=False, load='medium', cut=None, seed=2026110402):
    initialize()
    source = PREVIOUS / 'configs' / f'c_{model}_native_timed_state32.json'
    cfg = copy.deepcopy(json.loads(source.read_text()))
    cfg.update(name=name, count=count, state_probe=state, verify_transfer=state,
        observe_service_timing=True, observe_transport=True,
        phase='NATIVE_PATH_EXACT_STATE' if state else 'NATIVE_PATH_FINITE_SERVING_EVENT_MARKED',
        load=load, arrival_rate=RATES[model][load], arrival_seed=seed,
        timeout=1200 if state else 600, startup_timeout=240,
        parent_configuration=dict(path=str(source), SHA256=digest(source), publication=BASE),
        native_path=path != 'native', split_evaluation=False,
        token_suffix_origin_candidate=False, dynamic=False,
        handoff_mode='native_G1', arm='PD_G1', wavefront=False,
        atomic_prompt=cut is None, native_partial_cut=cut, native_path_candidate_version=5,
        feature_native_wire_v5=path != 'native')
    cfg.pop('contiguous_prompt_loop_burst', None)
    cfg.pop('runtime_control', None)
    pool = json.loads((PREVIOUS / 'data' / cfg['dataset']).read_text())
    assert count <= len(pool)
    destination = ROOT / 'data' / cfg['dataset']
    if not destination.exists():
        write(destination, pool)
    cfg.update(request_caps=[cfg['cap']] * count,
               prompt_ids=[row['prompt_id'] for row in pool[:count]])
    rng = random.Random(seed)
    cfg['arrival_offsets'] = [0.]
    for _ in range(count - 1):
        cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
    slo = json.loads((PREVIOUS / 'configs/frozen_SLO.json').read_text())
    destination = ROOT / 'configs/frozen_SLO.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == slo
    else:
        write(destination, slo)
    return cfg


def frontier_configuration(model, c, k, count, name, state=False, load='medium', seed=2026110402):
    cfg = configuration(model, 'partial', count, name, state, load,
                        cut=c if c else (4 if model == 'ouro' else 32), seed=seed)
    cfg.update(native_frontier=[c, k], atomic_prompt=False,
               native_path_candidate_version=4,
               mechanism='TOKEN_PREFIX_SUFFIX' if c == 0 else 'SINGLE_HANDOFF_TOKEN_LOOP_STAIRCASE')
    return cfg


if __name__ == '__main__':
    for model in ('ouro', 'huginn'):
        for path in ('native', 'zero', 'partial'):
            cut = (3 if model == 'ouro' else 31) if path == 'partial' else None
            cfg = configuration(model, path, 8, f'a_{model}_{path}_v2_state8', True, cut=cut)
            save(cfg)
    print('Prepared six independent N8 contracts. No GPU process started.')
