"""Check scalar algebra and dependency examples. No model or tensor backend."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import isqrt
from pathlib import Path
import struct


def utc():
    return datetime.now(timezone.utc).isoformat()


def compose_affine(after, before):
    a, b = after
    c, d = before
    return a * c, a * d + b


def apply_affine(pair, x):
    a, b = pair
    return a * x + b


def bf16_rne(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def run_checks():
    rows = []

    # Exact positive control. All affine coefficients exist before the scan.
    pairs = [(Fraction(3, 2), Fraction(r, 8)) for r in range(1, 5)]
    x0 = Fraction(2, 3)
    sequential = []
    x = x0
    for pair in pairs:
        x = apply_affine(pair, x)
        sequential.append(x)
    p01 = compose_affine(pairs[1], pairs[0])
    p23 = compose_affine(pairs[3], pairs[2])
    prefixes = [pairs[0], p01, compose_affine(pairs[2], p01),
                compose_affine(p23, p01)]
    scanned = [apply_affine(pair, x0) for pair in prefixes]
    assert scanned == sequential
    rows.append({"id": "known_affine_positive_control", "status": "PASS",
                 "states": [str(x) for x in sequential],
                 "claim": "Known affine summaries compose associatively. This checks algebra, not physical parallel execution."})

    # Balanced composition objects still evaluate a black-box chain in order.
    calls = []

    def leaf(x):
        calls.append(x)
        return x * x + 1

    def compose(after, before):
        return lambda x: after(before(x))

    left = compose(leaf, leaf)
    right = compose(leaf, leaf)
    balanced = compose(right, left)
    answer = balanced(1)
    assert answer == 677 and calls == [1, 2, 5, 26]
    rows.append({"id": "balanced_composition_evaluation", "status": "PASS",
                 "composition_tree_height": 2, "dependent_leaf_calls": len(calls),
                 "leaf_inputs": calls, "output": answer,
                 "claim": "A balanced composition syntax does not itself shorten opaque function evaluation."})

    # Infinite polynomial closure differs from a finite-horizon exact expression.
    degrees = [1]
    for _ in range(4):
        degrees.append(2 * degrees[-1])
    assert degrees == [1, 2, 4, 8, 16]
    for degree in range(1, 17):
        assert 2 * degree > degree
    rows.append({"id": "polynomial_lift_scope", "status": "PASS",
                 "four_step_powers_for_F_x_squared": degrees,
                 "claim": "A finite polynomial basis with a positive largest degree is not invariant under x->x^2. A four-step expression remains finite. Neither fact decides cheap nonpolynomial transforms."})

    # A reversible nonlinear coupling is not a forward dependency removal.
    def forward_pair(pair):
        x1, x2 = pair
        y1 = x1 + x2 * x2
        y2 = x2 + y1 * y1
        return y1, y2

    def inverse_pair(pair):
        y1, y2 = pair
        x2 = y2 - y1 * y1
        x1 = y1 - x2 * x2
        return x1, x2

    for x in [(1, 2), (-3, 1), (0, 0)]:
        assert inverse_pair(forward_pair(x)) == x
    rows.append({"id": "reversible_coupling_positive_control", "status": "PASS",
                 "cases": 3,
                 "claim": "The inverse reconstructs inputs from available outputs. Forward y2 still consumes actual y1."})

    # BF16 RNE has exact collisions even for a real-arithmetic translation.
    x_a, x_b = 0.0, 2.0 ** -8
    assert bf16_rne(x_a) == x_a and bf16_rne(x_b) == x_b
    y_a, y_b = bf16_rne(1.0 + x_a), bf16_rne(1.0 + x_b)
    assert x_a != x_b and y_a == y_b == 1.0
    rows.append({"id": "bf16_translation_collision", "status": "PASS",
                 "inputs": [x_a, x_b], "outputs": [y_a, y_b],
                 "claim": "A real inverse need not invert a rounded implementation. This is a BF16 RNE toy, not an Ouro collision observation."})

    # Positive epsilon does not automatically make real RMSNorm noninjective.
    x = [Fraction(3), Fraction(4)]
    gain = [Fraction(2), Fraction(3)]
    eps = Fraction(7, 2)
    mean_square = sum(v * v for v in x) / len(x)
    scale_square = mean_square + eps
    scale = Fraction(isqrt(scale_square.numerator), isqrt(scale_square.denominator))
    assert scale * scale == scale_square
    y = [g * v / scale for g, v in zip(gain, x)]
    u = [v / g for v, g in zip(y, gain)]
    a = sum(v * v for v in u) / len(u)
    inverse_scale_square = eps / (1 - a)
    assert inverse_scale_square == scale_square and a < 1
    recovered = [v * scale for v in u]
    assert recovered == x
    rows.append({"id": "epsilon_rmsnorm_inverse_positive_control", "status": "PASS",
                 "input": [str(v) for v in x], "output": [str(v) for v in y],
                 "claim": "Real RMSNorm with positive epsilon and nonzero gains can be invertible on its range. This does not establish full-core or BF16 invertibility."})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = utc()
    rows = run_checks()
    result = {"schema_version": 1, "started_utc": started, "ended_utc": utc(),
              "evidence_class": "CPU_ALGEBRA_AND_SCOPE_CHECKS",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_execution": False, "GPU_execution": False,
              "checks": rows, "check_count": len(rows), "status": "PASS"}
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
