"""Build immutable same-runtime contracts before any cut selection."""
import copy
import random
from .common import *

RATES = {'ouro': dict(low=1.23460875, medium=3.4569045, heavy=6.356),
         'huginn': dict(low=.37903355455, medium=1.06129395274, heavy=1.971)}
ACTIONS = {
    'ouro': {'native': None, 'zero': None, 'token3840': [0, 3840], 'token3072': [0, 3072],
             'token4032': [0, 4032], 'loop3': 3, 'loop2': 2, 'loop1': 1,
             '2d_c3_k3072': [3, 3072], '2d_c2_k2048': [2, 2048], '2d_c3_k3840': [3, 3840]},
    'huginn': {'native': None, 'zero': None, 'token1984': [0, 1984], 'token2016': [0, 2016],
               'token1792': [0, 1792], 'loop31': 31, 'loop28': 28, 'loop24': 24,
               '2d_c31_k1024': [31, 1024], '2d_c28_k1536': [28, 1536], '2d_c31_k1792': [31, 1792]},
}


def permitted_methods(model):
    geometry = ROOT / 'state/equal_candidate_geometry_amendment.json'
    if geometry.exists():
        return json.loads(geometry.read_text())['mandatory_methods'][model]
    marker = ROOT / 'state/equal_candidate_budget_amendment.json'
    if not marker.exists():
        return list(ACTIONS[model])
    budget = json.loads(marker.read_text())['candidates_per_axis']
    result = ['native', 'zero']
    for axis in ('token', 'loop', '2d_'):
        result += [method for method in ACTIONS[model] if method.startswith(axis)][:budget]
    return result


def configuration(model, method, count, name, *, state=False, load='medium', start=1,
                  markers=True, split='discovery', policy='primary_reserved', graph_budget=8,
                  arrival_seed=None, simultaneous=False):
    initialize()
    source = PREVIOUS / 'configs' / f'c_{model}_native_medium_start1_128.json'
    cfg = copy.deepcopy(json.loads(source.read_text()))
    frontier = ACTIONS[model][method]
    for key in ('native_frontier', 'native_partial_cut', 'native_request_cuts'):
        cfg.pop(key, None)
    if isinstance(frontier, list):
        cfg.update(native_frontier=frontier, native_partial_cut=frontier[0] or cfg['depth'])
    elif frontier is not None:
        cfg['native_partial_cut'] = frontier
    cfg.update(name=name, method=method, count=count, state_probe=state, verify_transfer=state,
        native_path=method != 'native', feature_native_wire_v5=method != 'native',
        split_evaluation=False, dynamic=False, wavefront=False, D_local_probe_output_pacing=False,
        observe_service_timing=markers, observe_transport=True, deep_graph_policy=policy,
        deep_graph_budget=graph_budget, P_prefill_graph_shapes=graph_budget,
        deep_serial_primary_warmup=policy == 'primary_reserved',
        deep_complete_D_geometry_warmup=(policy == 'primary_reserved' and graph_budget >= 8 and not state),
        deep_fast_verification=True,
        independent_start=start, load=load, data_split=split,
        arrival_seed=arrival_seed if arrival_seed is not None else 2026110500 + start,
        timeout=900 if state else 600, startup_timeout=300,
        atomic_prompt=method in ('native', 'zero'), phase='DEEP_EXACT_QUALIFICATION' if state else 'DEEP_FINITE_SERVING',
        parent_configuration=dict(path=str(source), SHA256=digest(source), publication=BASE))
    if split == 'discovery':
        old = PREVIOUS / 'data' / cfg['dataset']
        dataset = ROOT / 'data' / cfg['dataset']
        if not dataset.exists():
            write(dataset, json.loads(old.read_text()))
    else:
        cfg['dataset'] = f'{split}_{model}_L{cfg["length"]}.json'
        dataset = ROOT / 'data' / cfg['dataset']
    pool = json.loads(dataset.read_text())
    assert count <= len(pool), 'No implicit prompt cycling'
    cfg.update(request_caps=[cfg['cap']] * count, prompt_ids=[row['prompt_id'] for row in pool[:count]],
               arrival_offsets=[0.] * count)
    if not simultaneous:
        rng = random.Random(cfg['arrival_seed'])
        for index in range(1, count):
            cfg['arrival_offsets'][index] = cfg['arrival_offsets'][index - 1] + rng.expovariate(RATES[model][load])
        cfg['arrival_rate'] = RATES[model][load]
        cfg['arrival_pattern'] = 'OPEN_LOOP_POISSON'
    else:
        cfg.pop('arrival_rate', None)
        cfg['arrival_pattern'] = 'SIMULTANEOUS_SUBMISSION_QUALIFICATION'
    frozen = ROOT / 'configs/frozen_SLO.json'
    value = json.loads((PREVIOUS / 'configs/frozen_SLO.json').read_text())
    if frozen.exists():
        assert json.loads(frozen.read_text()) == value
    else:
        write(frozen, value)
    return cfg


if __name__ == '__main__':
    for method in ('native', 'token1984'):
        cfg = configuration('huginn', method, 8, f'a_huginn_{method}_scheduler_recovery8',
                            state=True, policy=None, simultaneous=True)
        save(cfg)
    print('Prepared isolated Huginn scheduler recovery. No GPU job started.')
