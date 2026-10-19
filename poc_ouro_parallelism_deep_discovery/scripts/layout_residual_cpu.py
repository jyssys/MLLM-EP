#!/usr/bin/env python3
"""Check layout and residual algebra on constructed, non-model CPU tensors.

The script prints JSON. It does not read checkpoints or use accelerator libraries.
The software BF16 path is a declared rounding model, not a GPU kernel emulator.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import platform
import struct
import sys


EPS = 1.0e-6


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def add(x, y):
    return [a + b for a, b in zip(x, y)]


def mul(x, y):
    return [a * b for a, b in zip(x, y)]


def scale(s, x):
    return [s * a for a in x]


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def mv(w, x):
    return [dot(row, x) for row in w]


def rms(x, gamma, eps=EPS):
    s = 1.0 / math.sqrt(dot(x, x) / len(x) + eps)
    return scale(s, mul(gamma, x))


def silu(x):
    return x / (1.0 + math.exp(-x))


def softmax(x):
    m = max(x)
    exp = [math.exp(a - m) for a in x]
    return scale(1.0 / sum(exp), exp)


def maxerr(x, y):
    return max(abs(a - b) for a, b in zip(x, y))


def rows_error(x, y):
    return max(maxerr(a, b) for a, b in zip(x, y))


def slices(width):
    assert width % 2 == 0
    return [list(range(width // 2)), list(range(width // 2, width))]


def row_tp_parts(w, x):
    """Return full-output partial sums from two disjoint input partitions."""
    return [[sum(row[j] * x[j] for j in idx) for row in w]
            for idx in slices(len(x))]


def reduce_parts(parts):
    return [sum(p[j] for p in parts) for j in range(len(parts[0]))]


def norm_layout(x, gamma, feature_sharded):
    if not feature_sharded:
        return rms(x, gamma)
    # This central CPU representation emulates disjoint feature statistics.
    local_squares = [sum(x[j] * x[j] for j in idx) for idx in slices(len(x))]
    s = 1.0 / math.sqrt(sum(local_squares) / len(x) + EPS)
    return [s * gamma[j] * x[j] for j in range(len(x))]


def rope2(x, position):
    """Rotate a two-coordinate head by the explicit toy angle position/4."""
    angle = position / 4.0
    c, s = math.cos(angle), math.sin(angle)
    return [x[0] * c - x[1] * s, x[1] * c + x[0] * s]


def attention_heads(xs, layer):
    qs = [mv(layer['q'], x) for x in xs]
    ks = [mv(layer['k'], x) for x in xs]
    vs = [mv(layer['v'], x) for x in xs]
    outputs = []
    for token in range(len(xs)):
        out = []
        for head in range(2):
            begin = 2 * head
            q = rope2(qs[token][begin:begin + 2], token)
            keys = [rope2(ks[j][begin:begin + 2], j) for j in range(token + 1)]
            probs = softmax([dot(q, key) / math.sqrt(2.0) for key in keys])
            out.extend(sum(probs[j] * vs[j][begin + i] for j in range(token + 1))
                       for i in range(2))
        outputs.append(out)
    return outputs


# These small arrays are constructed algebra inputs, not sampled model weights.
LAYERS = [
    {
        'q': [[1, .25, 0, -.5], [0, 1, .5, 0], [.25, 0, 1, .25], [.5, -.25, 0, 1]],
        'k': [[.5, 0, .25, 0], [0, .75, 0, -.25], [.25, 0, .5, 0], [0, .25, 0, 1]],
        'v': [[1, 0, .5, 0], [0, .5, 0, .25], [.25, 0, .75, 0], [0, -.5, 0, 1]],
        'o': [[.5, .25, 0, -.25], [0, .75, .25, 0], [.25, 0, .5, .25], [-.5, .25, 0, .5]],
        'gate': [[1, .25, 0, 0], [0, .5, -.25, 0], [.25, 0, .75, 0],
                 [0, -.25, 0, 1], [.5, 0, .25, 0], [0, .25, 0, -.5]],
        'up': [[.5, 0, 0, .25], [0, .75, .25, 0], [.25, 0, .5, 0],
               [0, .25, 0, .5], [.5, 0, -.25, 0], [0, -.25, 0, .75]],
        'down': [[.5, 0, .25, 0, -.25, 0], [0, .5, 0, .25, 0, -.25],
                 [.25, 0, .5, 0, .25, 0], [0, -.25, 0, .5, 0, .25]],
        'norms': [[1, .75, 1.25, .5], [.75, 1, .5, 1.25],
                  [1.25, .5, 1, .75], [.5, 1.25, .75, 1]],
    },
    {
        'q': [[.75, 0, .25, 0], [.25, .5, 0, -.25], [0, .25, 1, 0], [0, -.25, .25, .5]],
        'k': [[1, -.25, 0, 0], [0, .5, .25, 0], [.25, 0, .75, 0], [0, .25, 0, .5]],
        'v': [[.5, .25, 0, 0], [0, 1, -.25, 0], [.25, 0, .5, .25], [0, -.25, 0, .75]],
        'o': [[.75, 0, .25, 0], [0, .5, 0, -.25], [.25, 0, .75, 0], [0, .25, 0, .5]],
        'gate': [[.5, 0, .25, 0], [.25, 1, 0, 0], [0, .25, .5, 0],
                 [0, -.25, 0, .75], [.75, 0, -.25, 0], [0, .5, 0, .25]],
        'up': [[1, 0, -.25, 0], [0, .5, .25, 0], [.25, 0, .75, 0],
               [0, .25, 0, 1], [.25, 0, .5, 0], [0, -.25, 0, .5]],
        'down': [[.25, 0, .5, 0, .25, 0], [0, .75, 0, -.25, 0, .25],
                 [.5, 0, .25, 0, -.25, 0], [0, .25, 0, .75, 0, -.25]],
        'norms': [[.75, 1.25, .5, 1], [1, .5, 1.25, .75],
                  [.5, 1, .75, 1.25], [1.25, .75, 1, .5]],
    },
]


def layer_forward(xs, layer, layout):
    sharded = layout == 'feature_sharded'
    n1 = [norm_layout(x, layer['norms'][0], sharded) for x in xs]
    heads = attention_heads(n1, layer)
    if layout == 'reference':
        attn = [mv(layer['o'], h) for h in heads]
    else:
        attn = [reduce_parts(row_tp_parts(layer['o'], h)) for h in heads]
    n2 = [norm_layout(a, layer['norms'][1], sharded) for a in attn]
    residual = [add(x, a) for x, a in zip(xs, n2)]
    n3 = [norm_layout(x, layer['norms'][2], sharded) for x in residual]
    mlp = []
    for x in n3:
        gate = mv(layer['gate'], x)
        up = mv(layer['up'], x)
        swiglu = [silu(g) * u for g, u in zip(gate, up)]
        if layout == 'reference':
            mlp.append(mv(layer['down'], swiglu))
        else:
            mlp.append(reduce_parts(row_tp_parts(layer['down'], swiglu)))
    n4 = [norm_layout(x, layer['norms'][3], sharded) for x in mlp]
    return [add(x, y) for x, y in zip(residual, n4)]


def two_layer_two_loop(layout):
    xs = [[.25, -.5, 1.25, 2], [1, .25, -.75, .5]]
    gamma_loop = [1, .75, 1.25, .5]
    states = []
    for recurrent_depth in range(2):
        for layer_index, layer in enumerate(LAYERS):
            xs = layer_forward(xs, layer, layout)
            states.append({'r': recurrent_depth, 'l': layer_index,
                           'boundary': 'block_output', 'hidden': xs})
        xs = [norm_layout(x, gamma_loop, layout == 'feature_sharded') for x in xs]
        states.append({'r': recurrent_depth, 'l': None,
                       'boundary': 'loop_norm', 'hidden': xs})
    return states


def compound_norm_moments(x, a, gamma_a, gamma_next):
    """Derive norm->residual->norm from four exact feature moments."""
    d = len(x)
    y = mul(gamma_a, a)
    moments = [dot(a, a), dot(x, x), dot(x, y), dot(y, y)]
    s_a = 1.0 / math.sqrt(moments[0] / d + EPS)
    norm_u_squared = moments[1] + 2 * s_a * moments[2] + s_a * s_a * moments[3]
    s_next = 1.0 / math.sqrt(norm_u_squared / d + EPS)
    u = add(x, scale(s_a, y))
    return scale(s_next, mul(gamma_next, u)), moments


def loop_norm_moments(x, gamma_loop, gamma_next):
    d = len(x)
    s_loop = 1.0 / math.sqrt(dot(x, x) / d + EPS)
    gx = mul(gamma_loop, x)
    s_next = 1.0 / math.sqrt(s_loop * s_loop * dot(gx, gx) / d + EPS)
    return scale(s_loop * s_next, mul(gamma_next, gx))


def f32(x):
    return struct.unpack('!f', struct.pack('!f', float(x)))[0]


def bf16(x):
    """Round an FP32 value to BF16 with round-to-nearest, ties-to-even."""
    bits = struct.unpack('!I', struct.pack('!f', f32(x)))[0]
    exponent = bits & 0x7f800000
    if exponent == 0x7f800000:
        return f32(x)
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('!f', struct.pack('!I', bits))[0]


def bf16_add(x, y):
    return bf16(f32(bf16(x) + bf16(y)))


def bf16_mv(w, x):
    outputs = []
    for row in w:
        acc = 0.0
        for a, b in zip(row, x):
            acc = f32(acc + f32(bf16(a) * bf16(b)))
        outputs.append(bf16(acc))
    return outputs


def bf16_rms(x, gamma):
    x = [bf16(a) for a in x]
    acc = 0.0
    for a in x:
        acc = f32(acc + f32(a * a))
    mean = f32(acc / len(x))
    inv = f32(1.0 / math.sqrt(f32(mean + f32(EPS))))
    normalized = [bf16(f32(a * inv)) for a in x]
    return [bf16(bf16(g) * a) for g, a in zip(gamma, normalized)]


def run_checks():
    assert sys.float_info.mant_dig == 53
    assert bf16(257.0) == 256.0
    assert bf16(259.0) == 260.0
    assert bf16(-257.0) == -256.0
    checks = {}
    x, a = [1.0, 0.0], [0.0, 1.0]
    correct = rms(add(x, a), [1, 1])
    wrong = add(rms(x, [1, 1]), rms(a, [1, 1]))
    checks['nonlinear_partial_norm'] = {
        'correct': correct, 'sum_of_separate_norms': wrong,
        'max_abs_error': maxerr(correct, wrong),
    }
    x, a = [1.0, 2.0], [3.0, -1.0]
    checks['missing_cross_rank_norm_moment'] = {
        'square_of_sum': dot(add(x, a), add(x, a)),
        'sum_of_local_squares': dot(x, x) + dot(a, a),
        'missing_cross_term': 2 * dot(x, a),
    }
    merged = silu(1.0 - 1.0) * (2.0 + 3.0)
    separate = silu(1.0) * 2.0 + silu(-1.0) * 3.0
    checks['nonlinear_partial_swiglu'] = {
        'after_preactivation_reduce': merged,
        'sum_of_local_swiglu': separate,
        'abs_error': abs(merged - separate),
    }
    merged = softmax(add([2.0, 0.0], [0.0, 1.0]))
    separate = scale(.5, add(softmax([2.0, 0.0]), softmax([0.0, 1.0])))
    checks['nonlinear_partial_softmax'] = {
        'softmax_after_score_reduce': merged,
        'mean_of_local_softmax': separate,
        'max_abs_error': maxerr(merged, separate),
    }

    reference = two_layer_two_loop('reference')
    replicated = two_layer_two_loop('replicated_tp')
    feature = two_layer_two_loop('feature_sharded')
    errors = []
    for ref, rep, shard in zip(reference, replicated, feature):
        errors.append({'r': ref['r'], 'l': ref['l'], 'boundary': ref['boundary'],
                       'replicated_TP_max_abs_error': rows_error(ref['hidden'], rep['hidden']),
                       'feature_layout_max_abs_error': rows_error(ref['hidden'], shard['hidden'])})
    checks['two_layer_two_loop_float64'] = {
        'shape': {'tokens': 2, 'hidden': 4, 'heads': 2, 'head_width': 2,
                  'mlp_width': 6, 'physical_layers': 2, 'loops': 2, 'ranks_simulated': 2},
        'physical_layer_order': [0, 1, 0, 1],
        'norm_calls_per_token': 18,
        'reference_final_hidden': reference[-1]['hidden'],
        'boundary_errors': errors,
        'max_abs_error': max(max(e['replicated_TP_max_abs_error'], e['feature_layout_max_abs_error'])
                             for e in errors),
        'scope': 'Central CPU arithmetic representation. No distributed runtime or resource schedule ran.',
        'stage': 'Two-token causal prefill. No cached decode is simulated.',
        'KV_scope': 'Each logical layer/depth computes separate K/V from its matching input states.',
    }

    x, a = [.25, -.5, 1.25, 2.0], [.5, .25, -.75, .125]
    gamma = [1, .75, 1.25, .5]
    w = [[1, 0, .5, -.25], [.25, -.5, 1, .75], [-.5, 1, .25, 0]]
    s_x = 1 / math.sqrt(dot(x, x) / len(x) + EPS)
    y = add(x, a)
    s_y = 1 / math.sqrt(dot(y, y) / len(y) + EPS)
    direct = mv(w, rms(y, gamma))
    base = mv(w, rms(x, gamma))
    delta_projection = mv(w, mul(gamma, a))
    cached_formula = add(scale(s_y / s_x, base), scale(s_y, delta_projection))
    two_projection_formula = scale(s_y, add(mv(w, mul(gamma, x)), delta_projection))
    checks['float64_normalized_base_plus_delta'] = {
        'x': x, 'a': a, 'gamma': gamma, 'W': w,
        's_x': s_x, 's_y': s_y,
        'direct': direct, 'cached_base_formula': cached_formula,
        'two_projection_formula': two_projection_formula,
        'cached_max_abs_error': maxerr(direct, cached_formula),
        'two_projection_max_abs_error': maxerr(direct, two_projection_formula),
        'norm_square_via_cross_term_error': abs(dot(y, y) - (dot(x, x) + 2 * dot(x, a) + dot(a, a))),
    }
    gamma_next = [.75, 1.25, .5, 1]
    chain_direct = rms(add(x, rms(a, gamma)), gamma_next)
    chain_moments, moments = compound_norm_moments(x, a, gamma, gamma_next)
    loop_direct = rms(rms(y, gamma), gamma_next)
    checks['float64_norm_moment_fusion'] = {
        'compound_chain_max_abs_error': maxerr(chain_direct, chain_moments),
        'compound_chain_four_moments': moments,
        'loop_norm_chain_max_abs_error': maxerr(loop_direct, loop_norm_moments(y, gamma, gamma_next)),
        'scope': 'Real-arithmetic identities checked with float64. No native BF16 equivalence follows.',
    }

    checks['bf16_addition_reordering'] = {
        'inputs': [256.0, 1.0, -256.0],
        'left_associated': bf16_add(bf16_add(256, 1), -256),
        'right_associated': bf16_add(256, bf16_add(1, -256)),
    }
    x_b, a_b = [256.0, 1.0], [-256.0, 0.0]
    y_b = [bf16_add(u, v) for u, v in zip(x_b, a_b)]
    direct_b = bf16_mv([[1, 1]], bf16_rms(y_b, [1, 1]))
    base_b = bf16_mv([[1, 1]], x_b)
    delta_b = bf16_mv([[1, 1]], a_b)
    sy_b = f32(1 / math.sqrt(f32(dot(y_b, y_b) / len(y_b) + EPS)))
    split_b = [bf16(sy_b * bf16_add(u, v)) for u, v in zip(base_b, delta_b)]
    sx_b = f32(1 / math.sqrt(f32(dot(x_b, x_b) / len(x_b) + EPS)))
    cached_base_b = bf16_mv([[1, 1]], bf16_rms(x_b, [1, 1]))
    cached_split_b = [bf16_add(bf16(f32(sy_b / sx_b) * u), bf16(sy_b * v))
                      for u, v in zip(cached_base_b, delta_b)]
    checks['bf16_normalized_two_projection_reordering'] = {
        'x': x_b, 'a': a_b, 'W': [[1, 1]], 'gamma': [1, 1],
        'direct_software_source_order': direct_b,
        'two_rounded_projections_then_norm_scalar': split_b,
        'abs_error': maxerr(direct_b, split_b),
        'cached_normalized_base': cached_base_b,
        'cached_base_plus_scaled_delta': cached_split_b,
        'cached_formula_abs_error': maxerr(direct_b, cached_split_b),
    }

    candidates = [
        ([.125, -.5, 2.0, 1.0], [.2, 1.3, -.7, .04], [1, .75, 1.25, .5], [.9, 1.1, .7, 1.3]),
        ([.25, -.5, 1.25, 2], [.5, .25, -.75, .125], [1, .75, 1.25, .5], [.75, 1.25, .5, 1]),
        ([1, 2, 3, 4], [4, 3, 2, 1], [.75, 1.25, .5, 1], [1.25, .5, 1, .75]),
    ]
    counterexamples = []
    for x_c, a_c, g_c, gn_c in candidates:
        x_c, a_c = [bf16(z) for z in x_c], [bf16(z) for z in a_c]
        g_c, gn_c = [bf16(z) for z in g_c], [bf16(z) for z in gn_c]
        native_u = [bf16_add(u, v) for u, v in zip(x_c, bf16_rms(a_c, g_c))]
        native = bf16_rms(native_u, gn_c)
        fused, _ = compound_norm_moments(x_c, a_c, g_c, gn_c)
        fused = [bf16(z) for z in fused]
        native_loop = bf16_rms(bf16_rms(x_c, g_c), gn_c)
        fused_loop = [bf16(z) for z in loop_norm_moments(x_c, g_c, gn_c)]
        counterexamples.append({
            'x': x_c, 'a': a_c, 'gamma_a': g_c, 'gamma_next': gn_c,
            'native_compound': native, 'real_moment_fusion_then_bf16': fused,
            'compound_max_abs_error': maxerr(native, fused),
            'native_loop_chain': native_loop, 'real_loop_fusion_then_bf16': fused_loop,
            'loop_chain_max_abs_error': maxerr(native_loop, fused_loop),
        })
    checks['bf16_norm_fusion_counterexamples'] = counterexamples

    assert checks['nonlinear_partial_norm']['max_abs_error'] > .1
    assert checks['nonlinear_partial_swiglu']['abs_error'] > .1
    assert checks['nonlinear_partial_softmax']['max_abs_error'] > .1
    assert checks['two_layer_two_loop_float64']['max_abs_error'] < 1e-12
    assert checks['float64_normalized_base_plus_delta']['cached_max_abs_error'] < 1e-12
    assert checks['float64_normalized_base_plus_delta']['two_projection_max_abs_error'] < 1e-12
    assert checks['float64_norm_moment_fusion']['compound_chain_max_abs_error'] < 1e-12
    assert checks['float64_norm_moment_fusion']['loop_norm_chain_max_abs_error'] < 1e-12
    assert checks['bf16_addition_reordering']['left_associated'] != checks['bf16_addition_reordering']['right_associated']
    assert checks['bf16_normalized_two_projection_reordering']['abs_error'] > 1
    assert checks['bf16_normalized_two_projection_reordering']['cached_formula_abs_error'] > 0
    assert any(c['compound_max_abs_error'] > 0 for c in counterexamples)
    assert any(c['loop_chain_max_abs_error'] > 0 for c in counterexamples)
    return checks


def main():
    start = utc_now()
    checks = run_checks()
    result = {
        'schema_version': 1,
        'status': 'PASS_CPU_ALGEBRA_CONTROLS',
        'evidence_class': 'ANALYTIC_BOUND',
        'evidence_subtype': 'EXECUTED_CONSTRUCTED_CPU_ALGEBRA',
        'segment_start_utc': '2026-10-19T10:12:07Z',
        'execution_start_utc': start,
        'execution_end_utc': utc_now(),
        'environment': {'python': platform.python_version(), 'implementation': platform.python_implementation(),
                        'float_mantissa_bits': sys.float_info.mant_dig,
                        'float_max_exponent': sys.float_info.max_exp},
        'precision_contract': {
            'float64': 'Python binary64 operations with the explicitly coded evaluation order.',
            'bf16': 'Software FP32-to-BF16 round-to-nearest, ties-to-even. FP32 products/accumulations use struct rounding.',
            'normalization': 'The software path follows FP32 norm statistics, BF16 normalized-value cast, then BF16 gamma multiplication.',
            'not_gpu_equivalence': 'No H100, CUDA, PyTorch kernel, NCCL reduction, or backend-equivalence claim.',
        },
        'scope': {'checkpoint_loaded': False, 'gpu_used': False, 'quality_measurement': False,
                  'performance_measurement': False, 'random_weights_used': False,
                  'target_model_shape_emulated': False},
        'sources': {
            'model_revision': '574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
            'model_source': 'https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py',
            'functions': ['OuroMLP.forward', 'OuroAttention.forward', 'OuroRMSNorm.forward', 'OuroDecoderLayer.forward', 'OuroModel.forward'],
            'generic_controls': [
                {'title': 'Megatron-LM', 'version': 'arXiv:1909.08053v4',
                 'url': 'https://arxiv.org/html/1909.08053v4', 'locator': 'Section 3'},
                {'title': 'PyTorch TP', 'version': '2.14 documentation',
                 'url': 'https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html',
                 'locator': 'ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput'},
                {'title': 'FlashInfer communication', 'version': '0.7.2 documentation',
                 'url': 'https://docs.flashinfer.ai/api/comm.html',
                 'locator': 'Unified AllReduce Fusion API'},
                {'title': 'NCCL collectives', 'version': '2.32.3 documentation',
                 'url': 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html',
                 'locator': 'AllReduce, ReduceScatter, AllGather'},
            ],
        },
        'checks': checks,
        'projection_costs': [
            {'operator': name, 'input_width': d, 'output_width': m,
             'direct_dense_flop_per_token': 2 * m * d,
             'uncached_base_plus_delta_dense_flop_per_token': 4 * m * d,
             'cached_base_incremental_dense_flop_per_token': 2 * m * d,
             'extra_scalar_and_vector_work_excluded_from_flop_columns': True}
            for name, d, m in [('QKV_stack', 2048, 6144), ('gate_up_stack', 2048, 11264),
                               ('attention_output', 2048, 2048), ('MLP_down', 5632, 2048)]
        ],
        'layout_collective_counts_for_two_layers_two_loops': {
            'replicated_1D_TP': {'vector_allreduces': 8, 'scalar_allreduces': 0},
            'feature_layout_unfused': {'vector_reduce_scatters': 8, 'vector_allgathers': 8,
                                       'scalar_stat_allreduces': 18},
            'count_scope': 'One illustrative layout. No minimum-communication theorem or timing estimate.',
        },
        'layout_contract': {
            'replicated': ['N1 local', 'head-local attention', 'O row-partials', 'vector allreduce',
                           'N2 local', 'residual', 'N3 local', 'gate/up column shards',
                           'local SwiGLU', 'down row-partials', 'vector allreduce', 'N4 local', 'residual'],
            'feature_sharded': ['N1 statistic allreduce', 'normalized-input allgather', 'head-local attention',
                                'O row-partials', 'vector reduce-scatter', 'N2 statistic allreduce',
                                'feature-local residual', 'N3 statistic allreduce', 'normalized-input allgather',
                                'gate/up column shards', 'local SwiGLU', 'down row-partials',
                                'vector reduce-scatter', 'N4 statistic allreduce', 'feature-local residual'],
            'loop_boundary': 'Shared physical weights retain their layout. Feature-sharded loop RMSNorm still needs global statistics.',
        },
        'conclusions': [
            'Local nonlinear evaluation of unreduced row-TP partials is not the original graph.',
            'Disjoint feature shards can propagate with correct global norm statistics and explicit projection communication.',
            'Real-arithmetic moment fusion is a positive control, not native BF16 graph equivalence.',
            'Dense base-plus-delta projection does not reduce dense projection work without extra structure.',
            'These failures do not reject all communication-avoiding algorithms.',
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
