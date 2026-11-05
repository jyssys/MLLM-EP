"""Use recorded query geometries to account for the unchanged numerical patch."""
import math
from .common import *


def structural_attention(cfg):
    t, depth = cfg['length'], cfg['depth']
    frontier = cfg.get('native_frontier')
    c, k = frontier if frontier else (cfg.get('native_partial_cut', depth), 0)
    if cfg.get('native_request_cuts'):
        c = sum(cfg['native_request_cuts']) / len(cfg['native_request_cuts'])
    pairs = lambda length: length * (length + 1) // 2
    d = (depth - c) * (pairs(t) - pairs(k))
    return dict(model=cfg['model'], method=cfg['method'], input_tokens=t, recurrent_depth=depth,
        D_core_recurrent_rows=(depth - c) * (t - k),
        P_core_recurrent_rows=depth * t - (depth - c) * (t - k),
        D_core_causal_pairs_per_core_layer=d,
        P_core_causal_pairs_per_core_layer=depth * pairs(t) - d,
        scope='STRUCTURAL_CAUSAL_PAIR_ACCOUNTING. Includes causal prefix dependence. Excludes Prelude, Coda, head dimensions, numerical padding, and runtime overhead. Not measured FLOPs or a cost prediction.')


def account(name):
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    result = []
    core_layers = 24 if cfg['model'] == 'ouro' else 4
    for path in (ROOT / 'raw' / name).glob('deep_graph_cache_*.json'):
        audit = json.loads(path.read_text())
        for entry in audit['geometries'].values():
            calls = entry['measured']
            q, sequences, max_query, packed, masked = entry['key']
            if not calls:
                continue
            tiles = math.ceil(q / 256)
            result.append(dict(name=name, model=cfg['model'], method=cfg['method'], role=audit['role'],
                actual_query_rows=q, actual_sequences=sequences, maximum_query_length=max_query,
                measured_recurrent_prefill_calls=calls, fixed_projection_tile_rows=256,
                structural_core_down_GEMM_tiles=calls * core_layers * tiles,
                structural_core_down_padded_rows=calls * core_layers * tiles * 256,
                logical_core_down_rows=calls * core_layers * q,
                evidence='MEASURED_QUERY_GEOMETRY_WITH_STRUCTURAL_FIXED_TILE_ACCOUNTING',
                scope='Counts follow the unchanged fixed-256-row Down projection code. Not CUPTI kernel invocations, measured FLOPs, or achieved throughput. Other projections and Huginn adapter are separate.'))
    return result
