"""Check tiny arithmetic controls. Do not execute a model or use a GPU."""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_word(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def finite_parts(word: int) -> tuple[int, int, int]:
    sign = -1 if word & 0x8000 else 1
    exponent = (word >> 7) & 0xFF
    fraction = word & 0x7F
    if exponent == 0xFF:
        raise ValueError("The rational control accepts finite BF16 words only.")
    significand = fraction + (128 if exponent else 0)
    power = exponent - 127 - 7 if exponent else -126 - 7
    return sign, significand, power


def power_two(power: int) -> Fraction:
    return Fraction(2**power) if power >= 0 else Fraction(1, 2 ** (-power))


def exact_value(word: int) -> Fraction:
    sign, significand, power = finite_parts(word)
    return sign * significand * power_two(power)


def controls() -> dict:
    started = now()
    table = array("H", (a * b for a in range(256) for b in range(256)))
    assert table.itemsize == 2
    assert len(table) == 65536
    assert all(table[256 * a + b] == a * b for a in range(256) for b in range(256))

    # A deterministic finite-word sample includes normal and subnormal values.
    words = [0, 1, 0x7F, 0x80, 0x100, 0x1A00, 0x3F00, 0x3F80, 0x7F7F,
             0x8000, 0x8001, 0x8080, 0xBF80, 0xFF7F]
    pairs = []
    for index in range(1024):
        left = words[index % len(words)]
        right = words[(index * 5 + 3) % len(words)]
        sa, ma, ea = finite_parts(left)
        sb, mb, eb = finite_parts(right)
        reconstructed = sa * sb * table[ma * 256 + mb] * power_two(ea + eb)
        assert reconstructed == exact_value(left) * exact_value(right)
        if index < 8:
            pairs.append({"left_word": left, "right_word": right,
                          "exact_product": str(reconstructed)})

    # Single rounding differs from storing a rounded product at underflow.
    operand = 2.0**-75
    accumulator = 2.0**-149
    rounded_product = f32(operand * operand)
    two_roundings = f32(rounded_product + accumulator)
    # Python binary64 represents this constructed exact sum without loss.
    one_rounding = f32(operand * operand + accumulator)
    assert bf16_word(operand) == 0x1A00
    assert rounded_product == 0.0
    assert two_roundings == accumulator
    assert one_rounding == 2.0**-148

    # Repeated coefficients expose a genuine exact-arithmetic shared sum.
    toy_weights = [[2, 2, 3, 4], [5, 5, 6, 7]]
    toy_input = [1, 2, 3, 4]
    direct = [sum(Fraction(w) * x for w, x in zip(row, toy_input))
              for row in toy_weights]
    shared = Fraction(toy_input[0] + toy_input[1])
    grouped = [row[0] * shared + row[2] * toy_input[2] + row[3] * toy_input[3]
               for row in toy_weights]
    assert direct == grouped == [Fraction(31), Fraction(61)]

    # All operands are BF16-representable. This is not a Tensor Core emulator.
    reorder_input = [2.0**24, 0.5, -(2.0**24)]
    reorder_weights = [1.0, 2.0, 1.0]
    direct_f32 = 0.0
    for weight, value in zip(reorder_weights, reorder_input):
        direct_f32 = f32(direct_f32 + f32(weight * value))
    grouped_f32 = f32(f32(reorder_input[0] + reorder_input[2])
                       + f32(2.0 * reorder_input[1]))
    real_direct = sum(Fraction(w) * Fraction(x)
                      for w, x in zip(reorder_weights, reorder_input))
    assert real_direct == 1
    assert direct_f32 == 0.0 and grouped_f32 == 1.0
    assert bf16_word(direct_f32) == 0 and bf16_word(grouped_f32) == 0x3F80

    # Structural counts for one D-to-I projection, not measured traffic.
    inputs, outputs, alphabet = 2048, 5632, 65536
    global_scalar_bytes = alphabet**2 * 4
    per_output_two_input_bytes = outputs * (inputs // 2) * alphabet**2 * 4
    remote_product_wire_bytes = inputs * outputs * (2 + 2 + 4)
    complete_projection_wire_bytes = (inputs + outputs) * 2
    assert global_scalar_bytes == 16 * 2**30
    assert per_output_two_input_bytes == 88 * 2**50
    assert remote_product_wire_bytes == 88 * 2**20
    assert complete_projection_wire_bytes == 15360

    return {
        "schema_version": 1,
        "start_utc": started,
        "end_utc": now(),
        "evidence_class": "DETERMINISTIC_CPU_ARITHMETIC_AND_STRUCTURAL_CONTROLS",
        "GPU_execution": False,
        "model_execution": False,
        "model_weights_used": False,
        "Tensor_Core_emulation": False,
        "Python": platform.python_version(),
        "host_byteorder": sys.byteorder,
        "significand_table": {"entries_checked": 65536, "bytes": len(table) * table.itemsize,
                              "sha256_native_bytes": hashlib.sha256(table.tobytes()).hexdigest(),
                              "rational_product_pairs_checked": 1024,
                              "sample_pairs": pairs,
                              "scope": "Exact finite rational products. Does not preserve signed zero, exceptions, native FMA, or a GEMM reduction."},
        "product_rounding_counterexample": {"BF16_operand_word": 0x1A00,
                                            "operand": operand, "FP32_accumulator": accumulator,
                                            "rounded_product": rounded_product,
                                            "stored_product_then_add": two_roundings,
                                            "one_rounding_exact_sum_control": one_rounding},
        "shared_subset_positive_control": {"weights": toy_weights, "input": toy_input,
                                           "direct": [str(x) for x in direct],
                                           "grouped": [str(x) for x in grouped],
                                           "direct_multiplies": 8, "grouped_multiplies": 6,
                                           "direct_additions": 6, "grouped_additions_including_shared_sum": 5},
        "reordering_counterexample": {"weights": reorder_weights, "input": reorder_input,
                                       "real_exact": str(real_direct), "direct_FP32": direct_f32,
                                       "grouped_FP32": grouped_f32,
                                       "direct_BF16_word": bf16_word(direct_f32),
                                       "grouped_BF16_word": bf16_word(grouped_f32)},
        "structural_counts": {"input_width": inputs, "output_width": outputs,
                               "global_full_BF16_product_table_FP32_bytes": global_scalar_bytes,
                               "unshared_two_input_partial_dot_table_FP32_bytes": per_output_two_input_bytes,
                               "remote_per_product_query_plus_return_bytes": remote_product_wire_bytes,
                               "complete_projection_input_plus_output_BF16_bytes": complete_projection_wire_bytes,
                               "wire_limit": "Chosen uncompressed protocol, not a lower bound for every implementation."},
        "assertion_groups": 6,
        "result": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Create one new JSON artifact. Refuse an existing path.")
    args = parser.parse_args()
    result = controls()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps({"result": result["result"], "assertion_groups": result["assertion_groups"],
                      "output": str(args.output), "GPU_execution": False}))


if __name__ == "__main__":
    main()
