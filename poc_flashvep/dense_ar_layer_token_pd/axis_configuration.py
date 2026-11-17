"""Common runtime and explicit best-vs-best or matched-geometry controls."""
import json
from .common import ROOT, SLO_PRIOR, write
from ..duetserve_agg_decode.configuration import offsets


def base(name, method='Native', count=64, cap=16, rate=4.8, seed=2026111710,
         layers=8, suffix=2048, matched=False, **changes):
    cfg = json.loads((SLO_PRIOR / 'configs/heldout_backoff_Dynamic_O16_q4.4_s2026111516_W600_v1.json').read_text())
    for key in ('fixed_measurement', 'prompt_reuse_scope', 'selection_SHA256'):
        cfg.pop(key, None)
    cfg.update(name=name, count=count, cap=cap, request_caps=[cap] * count,
        arrival_offsets=offsets(count, rate, seed), arrival_rate=rate, arrival_seed=seed,
        dataset='discovery.json', evaluation_arm=method, phase='LAYER_TOKEN_HEADROOM_DISCOVERY',
        policy='NATIVE' if method in ('Native', 'Token', 'PChunk') else 'STATIC' if method == 'Static' else 'DIRECT',
        static_layers=layers, migration_fraction=1., P_group_size=4, P_slots=4,
        D_admission_batch=4, D_slots=16, D_wait_for_source_macro=True,
        D_combine_ready_macros=False, preplanned_source_groups=matched,
        clean_timing=True, correctness_only=False, runtime_observation=False,
        full_prompt_state_witness=False, imported_native_state_witness=False,
        verify_layer_execution=False, hidden_verify_hash=False, verify_native_prefix_KV=False,
        hidden_destination_slots=16, hidden_source_slots=16,
        warmup_count=5, warmup_batch_sizes=[1, 2, 3, 4], warmup_layer_counts=[0, 2, 4, 8, 12],
        physical_activity=False, execution_events=False, boundary_events=False,
        entry_module='poc_flashvep.dense_ar_layer_token_pd.dynamic_serving',
        timeout=1200, wall_timeout=1600)
    if method == 'Token':
        cfg.update(token_axis=True, token_suffix=suffix, reuse_prefill_token=False,
            layer_mode=None, warmup_layer_counts=[0])
    elif method == 'PChunk':
        cfg.update(chunk=suffix, atomic_prompt=False, warmup_layer_counts=[0])
    cfg.update(changes)
    pool = json.loads((ROOT / 'data' / cfg['dataset']).read_text())
    if count > len(pool):
        path = ROOT / 'data' / (cfg['dataset'].removesuffix('.json') + f'_declared_cycle_N{count}.json')
        expanded = [pool[i % len(pool)] for i in range(count)]
        if path.exists():
            assert json.loads(path.read_text()) == expanded
        else:
            write(path, expanded)
        cfg.update(dataset=path.name, prompt_reuse_scope='Declared source-pool cycle. No prefix caching. Repeated documents are not independent samples.')
    return cfg
