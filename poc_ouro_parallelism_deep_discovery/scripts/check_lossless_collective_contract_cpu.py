"""Check bounded BF16 identities on CPU. This is not a transport codec."""

import json

import torch


def bf16(value):
    return torch.tensor(value, dtype=torch.bfloat16, device="cpu")


def word(value):
    return int(bf16(value).view(torch.int16).item()) & 0xFFFF


def from_word(value):
    signed = value if value < 0x8000 else value - 0x10000
    return torch.tensor(signed, dtype=torch.int16, device="cpu").view(torch.bfloat16)


def add(left, right):
    return bf16(left) + bf16(right)


def number(value):
    return float(value.item())


def main():
    results = []
    base, current = word(1.0), word(1.5)
    difference = current ^ base
    assert difference ^ base == current
    wrong_decode = difference ^ word(0.5)
    assert wrong_decode == word(0.75)
    results.append({
        "case": "base_version_identity",
        "base_word": hex(base),
        "payload_word": hex(current),
        "xor_word": hex(difference),
        "wrong_base_decoded_word": hex(wrong_decode),
        "wrong_base_decoded_value": number(from_word(wrong_decode)),
    })

    words = [0x8000, 0x7FC1, 0x7F80, 0x0001, 0x3F80]
    for source in words:
        assert ((source ^ base) ^ base) == source
        assert int(from_word(source).view(torch.int16).item()) & 0xFFFF == source
    results.append({"case": "special_payload_words", "checked_words": [hex(x) for x in words]})

    codes = [word(1.5) ^ base, word(2.0) ^ base]
    integer_code_sum = sum(codes) & 0xFFFF
    decoded_integer_sum = integer_code_sum ^ base
    desired_word = word(number(add(1.5, 2.0)))
    assert decoded_integer_sum == word(3.0)
    assert decoded_integer_sum != desired_word
    code_as_bf16_sum = add(number(from_word(codes[0])), number(from_word(codes[1])))
    decoded_float_sum = word(number(code_as_bf16_sum)) ^ base
    assert decoded_float_sum == word(2.0)
    assert decoded_float_sum != desired_word
    results.append({
        "case": "encoded_words_are_not_sum_operands",
        "operand_words": [hex(word(1.5)), hex(word(2.0))],
        "xor_words": [hex(x) for x in codes],
        "integer_sum_then_xor_value": number(from_word(decoded_integer_sum)),
        "bf16_sum_of_reinterpreted_codes_then_xor_value": number(from_word(decoded_float_sum)),
        "desired_bf16_sum": number(from_word(desired_word)),
    })

    delta = bf16(1.0) - bf16(1024.0)
    reconstructed = bf16(1024.0) + delta
    assert number(delta) == -1024.0
    assert number(reconstructed) == 0.0
    results.append({
        "case": "bf16_subtraction_is_not_lossless",
        "base": 1024.0,
        "current": 1.0,
        "rounded_delta": number(delta),
        "reconstructed": number(reconstructed),
    })

    prior = [bf16(256.0), bf16(1.0)]
    current_partials = [bf16(256.0), bf16(2.0)]
    deltas = [current_partials[i] - prior[i] for i in range(2)]
    for i in range(2):
        assert torch.equal(prior[i] + deltas[i], current_partials[i])
    prior_reduced = prior[0] + prior[1]
    delta_reduced = deltas[0] + deltas[1]
    reused_global = prior_reduced + delta_reduced
    direct = current_partials[0] + current_partials[1]
    assert number(reused_global) == 256.0
    assert number(direct) == 258.0
    results.append({
        "case": "exact_local_deltas_do_not_restore_a_rounded_global_sum",
        "prior_partials": [number(x) for x in prior],
        "current_partials": [number(x) for x in current_partials],
        "local_deltas": [number(x) for x in deltas],
        "prior_reduced": number(prior_reduced),
        "reused_global": number(reused_global),
        "direct_two_operand_bf16_sum": number(direct),
    })

    left_fold = add(number(add(number(add(256.0, 1.0)), -256.0)), 0.0)
    regrouped = add(number(add(256.0, -256.0)), number(add(1.0, 0.0)))
    assert number(left_fold) == 0.0
    assert number(regrouped) == 1.0
    results.append({
        "case": "reduction_grouping_can_change_bf16_bits",
        "operands": [256.0, 1.0, -256.0, 0.0],
        "left_fold": number(left_fold),
        "regrouped": number(regrouped),
        "scope": "synthetic BF16 grouping, not an observed native GPU algorithm",
    })

    common_base = 1.0
    partials = [1.5, 2.0]
    real_deltas = [x - common_base for x in partials]
    assert common_base + sum(real_deltas) == 2.5
    assert len(partials) * common_base + sum(real_deltas) == sum(partials) == 3.5
    results.append({
        "case": "common_base_needs_rank_multiplicity",
        "rank_count": len(partials),
        "one_base_plus_delta_sum": common_base + sum(real_deltas),
        "rank_count_times_base_plus_delta_sum": len(partials) * common_base + sum(real_deltas),
        "direct_sum": sum(partials),
    })

    print(json.dumps({
        "status": "PASS",
        "device": "cpu",
        "torch_version": torch.__version__,
        "case_count": len(results),
        "cases": results,
        "limitations": [
            "No GPU kernel or collective executed.",
            "No entropy codec, transport, or model workload executed.",
            "The examples do not establish installed native accumulation order or precision.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
