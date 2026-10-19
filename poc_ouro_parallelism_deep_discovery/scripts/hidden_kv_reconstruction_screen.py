"""Exact toy counterexamples and target-shaped accounting. No model or GPU run."""
import argparse
from fractions import Fraction
import math
import struct

from poc_ouro_parallelism_deep_discovery.contract import ROOT, MODELS, write


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def transpose_mv(matrix, vector):
    return [dot(column, vector) for column in zip(*matrix)]


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    assert word & 0x7f800000 != 0x7f800000
    word = (word + 0x7fff + ((word >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', word))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    # Exact rational arithmetic. One key rotates90 degrees, the other does not.
    weight = [[Fraction(1), Fraction(2), Fraction(0), Fraction(1)],
              [Fraction(0), Fraction(1), Fraction(1), Fraction(0)]]
    query = [Fraction(1), Fraction(0)]
    hidden = [[Fraction(1), Fraction(0), Fraction(0), Fraction(0)],
              [Fraction(0), Fraction(1), Fraction(0), Fraction(0)]]
    rotations = [[[1, 0], [0, 1]], [[0, -1], [1, 0]]]
    native, position_aware, constant = [], [], []
    for state, rotation in zip(hidden, rotations):
        key = [dot(row, state) for row in weight]
        rotated = [dot(row, key) for row in rotation]
        native.append(dot(query, rotated))
        absorbed = transpose_mv(weight, transpose_mv(rotation, query))
        position_aware.append(dot(absorbed, state))
        constant.append(dot(transpose_mv(weight, query), state))
    assert native == position_aware == [1, -1] and constant == [1, 2]
    # Preserve the source projection rounding even when no rotation is involved.
    projected = bf16(1 + 1 / 256)
    assert projected == 1 and bf16(1 / 256) == 1 / 256
    counterexample = None
    for integer in range(1, 513):
        q = bf16(integer / 8)
        score_native = q * projected / math.sqrt(128)
        score_reordered = q * (1 + 1 / 256) / math.sqrt(128)
        probability_native = 1 / (1 + math.exp(-score_native))
        probability_reordered = 1 / (1 + math.exp(-score_reordered))
        if bf16(probability_native) != bf16(probability_reordered):
            counterexample = dict(head_dimension=128, query_nonzero=q,
                hidden_nonzero=[1, 1 / 256], key_weight_nonzero=[1, 1],
                native_BF16_projected_key=projected, native_score=score_native,
                reordered_score=score_reordered, native_probability=probability_native,
                reordered_probability=probability_reordered,
                native_BF16_output=bf16(probability_native),
                reordered_BF16_output=bf16(probability_reordered),
                scope='Toy softmax over two keys with one nonzero value. Not an actual Ouro attention result.')
            break
    assert counterexample is not None
    shapes = []
    for size, model in MODELS.items():
        layers, hidden_size = model['layers'], 2048
        for batch in [1, 4, 16, 32, 128]:
            shapes.append(dict(size=size, layers=layers, rows=batch,
                per_layer_normalized_hidden_BF16_bytes=batch * hidden_size * 2,
                per_layer_KV_BF16_bytes=batch * hidden_size * 4,
                all_layers_hidden_BF16_bytes=layers * batch * hidden_size * 2,
                all_layers_KV_BF16_bytes=layers * batch * hidden_size * 4,
                KV_weight_payload_bytes=layers * 2 * hidden_size ** 2 * 2,
                KV_projection_FLOPs=layers * batch * 4 * hidden_size ** 2,
                native_full_QKV_weight_payload_bytes=layers * 3 * hidden_size ** 2 * 2,
                native_full_QKV_projection_FLOPs=layers * batch * 6 * hidden_size ** 2,
                source='Loaded model geometry. Structural bytes/FLOPs, not HBM transactions or physical latency.'))
    write(output, dict(status='SCOPED_NEGATIVE_SCREEN', model_or_GPU_execution='NOT_RUN',
        exact_real_rope_control=dict(native=list(map(int, native)),
            correct_position_dependent_pullback=list(map(int, position_aware)),
            invalid_position_independent_pullback=list(map(int, constant))),
        BF16_projection_counterexample=counterexample, structural_shapes=shapes,
        interpretation='Reconstruction needs original per-layer QKV inputs, native projection/RoPE boundaries, and a numerical check. KV-only projection accounting is optimistic and does not establish fused-QKV word parity. Absorption changes operation order.',
        candidate='NONE. Refines existing KV-rematerialization and attention-summary screens.',
        limits='No pretrained output, quality, traffic, service time, or E2E speedup is measured. No universal rejection of all latent-cache methods.'))
    print('HIDDEN_KV_CPU_SCREEN', len(shapes), 'structural rows; two counterexamples and a positive real-arithmetic control')


if __name__ == '__main__':
    main()
