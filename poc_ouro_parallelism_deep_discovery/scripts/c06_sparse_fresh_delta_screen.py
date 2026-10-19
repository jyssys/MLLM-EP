"""CPU-only adjacent-pass sparsity screen. This script does not execute projections."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import random

import torch


ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (8, 16, 32, 128)
PINS = [('1.4B', 'capture_base14_v2/1.4B', '574fa66cb8bf5abdc979642d01cf2b79b16bfab1', 24),
        ('2.6B', 'capture_base26_v1/2.6B', '1ed04250da1a9936042725d302e81c8fa2ab5abd', 48)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_fp32_delta_mask(a, b, delta):
    """Check mathematical BF16 differences without assuming FP32 exactness."""
    # A BF16 value is a signed integer of at most 8 bits, times a power of two.
    # At exponent gaps <=16, the integer difference has at most 24 bits.
    # Zero operands are exact. BF16's minimum exponent also fits FP32 subnormals.
    aa, bb = a.view(torch.int16).to(torch.int32) & 65535, b.view(torch.int16).to(torch.int32) & 65535
    ea, eb = ((aa >> 7) & 255), ((bb >> 7) & 255)
    ea = torch.where(ea == 0, -133, ea - 134)
    eb = torch.where(eb == 0, -133, eb - 134)
    good = torch.isfinite(delta)
    rare = (a != 0) & (b != 0) & ((ea - eb).abs() > 16)
    for index in torch.nonzero(rare, as_tuple=False).flatten().tolist():
        wa, wb = int(aa[index]), int(bb[index])
        ma = ((wa & 127) | (128 if wa & 32640 else 0)) * (-1 if wa & 32768 else 1)
        mb = ((wb & 127) | (128 if wb & 32640 else 0)) * (-1 if wb & 32768 else 1)
        low = min(int(ea[index]), int(eb[index]))
        integer = abs((mb << (int(eb[index]) - low)) - (ma << (int(ea[index]) - low)))
        precision = (integer // (integer & -integer)).bit_length() if integer else 0
        good[index] = bool(good[index]) and precision <= 24
    return good


def measure(a, b):
    if a.device.type != 'cpu' or b.device.type != 'cpu':
        raise ValueError('Only CPU tensors are permitted.')
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or a.ndim != 1 or a.shape != b.shape:
        raise ValueError('Use equal one-dimensional BF16 rows.')
    if not a.numel() or any(a.numel() % width for width in WIDTHS):
        raise ValueError('The row width must divide every configured tile width.')
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Nonfinite inputs are not permitted.')
    numeric_changed = a != b
    word_changed = a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)
    delta = b.float() - a.float()
    exact_delta = exact_fp32_delta_mask(a, b, delta)
    reconstructed = a.float().clone()
    reconstructed[numeric_changed] += delta[numeric_changed]
    corrections = reconstructed.to(torch.bfloat16).view(torch.int16) != b.contiguous().view(torch.int16)
    word_reconstruction = reconstructed.to(torch.bfloat16)
    word_reconstruction[corrections] = b[corrections]
    if not torch.equal(word_reconstruction.view(torch.int16), b.contiguous().view(torch.int16)):
        raise AssertionError('The CPU word-correction model did not reconstruct the target.')
    k, kw, correction_count = int(numeric_changed.sum()), int(word_changed.sum()), int(corrections.sum())
    index_bits = math.ceil(math.log2(a.numel()))
    pack = lambda values, bits: (values * bits + 7) // 8
    tiles = {str(width): dict(total=a.numel() // width,
              numeric_unchanged=int((~numeric_changed).reshape(-1, width).all(dim=1).sum()),
              word_unchanged=int((~word_changed).reshape(-1, width).all(dim=1).sum())) for width in WIDTHS}
    payloads = dict(dense_bf16_bytes=2 * a.numel(),
        replacement_bf16_word_bitpacked_index_bytes=pack(kw, index_bits + 16),
        replacement_bf16_word_uint16_index_bytes=kw * 4,
        fp32_delta_bitpacked_index_bytes=pack(k, index_bits + 32),
        fp32_delta_uint16_index_bytes=k * 6,
        fp32_delta_word_correction_bitpacked_index_bytes=pack(correction_count, index_bits + 16),
        fp32_delta_word_correction_uint16_index_bytes=correction_count * 4)
    payloads['fp32_delta_with_word_corrections_bitpacked_index_bytes'] = (
        payloads['fp32_delta_bitpacked_index_bytes'] + payloads['fp32_delta_word_correction_bitpacked_index_bytes'])
    payloads['fp32_delta_with_word_corrections_uint16_index_bytes'] = (
        payloads['fp32_delta_uint16_index_bytes'] + payloads['fp32_delta_word_correction_uint16_index_bytes'])
    return dict(elements=a.numel(), numeric_equal=a.numel() - k, numeric_changed=k,
                word_equal=a.numel() - kw, word_changed=kw,
                signed_zero_only_word_changes=int((word_changed & ~numeric_changed).sum()),
                exact_fp32_delta_elements=int(exact_delta.sum()),
                nonexact_fp32_delta_elements=int((~exact_delta).sum()),
                exact_bf16_delta_elements=int((exact_delta & (delta.to(torch.bfloat16).float() == delta)).sum()),
                fp32_delta_reconstruction_word_corrections=correction_count,
                fp32_delta_payload_status=('EXACT_MATHEMATICAL_DELTAS' if bool(exact_delta.all())
                                           else 'NOT_AN_EXACT_DELTA_ENCODING'),
                fp32_delta_with_word_corrections_reconstructs_target_words=True,
                index_bits=index_bits, tiles=tiles, payloads=payloads)


def controls():
    base = torch.ones(2048, dtype=torch.bfloat16)
    one = base.clone(); one[17] = 2
    tile = base.clone(); tile[:32] = 2
    zero = torch.zeros_like(base); signed = zero.clone(); signed[17] = -0.0
    dense = base * 2
    rows = {name: measure(a, b) for name, a, b in [('identical', base, base),
             ('one_changed_coordinate', base, one), ('one_changed_tile32', base, tile),
             ('signed_zero_only', zero, signed), ('all_coordinates_changed', base, dense)]}
    assert rows['identical']['numeric_changed'] == 0
    assert rows['one_changed_coordinate']['numeric_changed'] == 1
    assert rows['one_changed_coordinate']['tiles']['128']['word_unchanged'] == 15
    assert rows['one_changed_tile32']['numeric_changed'] == 32
    assert rows['one_changed_tile32']['tiles']['8']['word_unchanged'] == 252
    assert rows['signed_zero_only']['numeric_changed'] == 0
    assert rows['signed_zero_only']['signed_zero_only_word_changes'] == 1
    assert rows['signed_zero_only']['fp32_delta_reconstruction_word_corrections'] == 1
    assert rows['all_coordinates_changed']['tiles']['8']['numeric_unchanged'] == 0
    assert rows['all_coordinates_changed']['payloads']['replacement_bf16_word_bitpacked_index_bytes'] == 6912
    tiny = base.clone(); tiny[17] = 2 ** -10
    bf16_counter = measure(base, tiny)
    assert bf16_counter['exact_fp32_delta_elements'] == 2048
    assert bf16_counter['exact_bf16_delta_elements'] == 2047
    assert float((base[17] + (tiny[17] - base[17]))) != float(tiny[17])
    rare = base.clone(); rare[17] = 2 ** -30
    fp32_counter = measure(base, rare)
    assert fp32_counter['nonexact_fp32_delta_elements'] == 1
    for value in [float('nan'), float('inf'), -float('inf')]:
        invalid = base.clone(); invalid[0] = value
        try:
            measure(base, invalid)
        except ValueError as error:
            assert str(error) == 'Nonfinite inputs are not permitted.'
        else:
            raise AssertionError('A nonfinite input passed.')
    rng = random.Random(606)
    words = lambda: [rng.choice(range(0, 32640)) | (rng.randrange(2) << 15) for _ in range(2048)]
    a = torch.tensor(words(), dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    b = torch.tensor(words(), dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    delta = b.float() - a.float()
    exact = exact_fp32_delta_mask(a, b, delta)
    for index, (av, bv, dv) in enumerate(zip(a.tolist(), b.tolist(), delta.tolist(), strict=True)):
        reference = math.isfinite(dv) and Fraction(dv) == Fraction(bv) - Fraction(av)
        assert bool(exact[index]) == reference, (index, av, bv, dv)
    return dict(status='PASS', sparse_controls=rows, bf16_subtraction_counterexample=bf16_counter,
                fp32_subtraction_counterexample=fp32_counter, nonfinite_rejection_cases=3,
                fp32_exactness_fraction_oracle_pairs=2048, fraction_oracle_seed=606)


def summaries(records):
    groups = defaultdict(list)
    for row in records:
        key = f"{row['size']}|{row['mode']}|{row['phase']}"
        groups[key].append(row)
    result = {}
    for key, rows in groups.items():
        sums = {name: sum(row[name] for row in rows) for name in ['elements', 'numeric_equal', 'numeric_changed',
                'word_equal', 'word_changed', 'signed_zero_only_word_changes', 'exact_fp32_delta_elements',
                'nonexact_fp32_delta_elements', 'exact_bf16_delta_elements', 'fp32_delta_reconstruction_word_corrections']}
        sums.update(row_pairs=len(rows), numeric_equal_fraction=sums['numeric_equal'] / sums['elements'],
                    word_equal_fraction=sums['word_equal'] / sums['elements'],
                    changed_coordinates_min=min(row['numeric_changed'] for row in rows),
                    changed_coordinates_max=max(row['numeric_changed'] for row in rows),
                    unchanged_numeric_rows=sum(row['numeric_changed'] == 0 for row in rows),
                    unchanged_word_rows=sum(row['word_changed'] == 0 for row in rows),
                    rows_with_exact_fp32_mathematical_deltas=sum(row['nonexact_fp32_delta_elements'] == 0 for row in rows),
                    fp32_delta_payload_status=('EXACT_MATHEMATICAL_DELTAS' if sums['nonexact_fp32_delta_elements'] == 0
                                               else 'NOT_AN_EXACT_DELTA_ENCODING'))
        sums['tiles'] = {str(width): {field: sum(row['tiles'][str(width)][field] for row in rows)
                          for field in ['total', 'numeric_unchanged', 'word_unchanged']} for width in WIDTHS}
        sums['payloads'] = {field: sum(row['payloads'][field] for row in rows) for field in rows[0]['payloads']}
        sums['payload_to_dense_ratios'] = {field: value / sums['payloads']['dense_bf16_bytes']
                                           for field, value in sums['payloads'].items()}
        result[key] = sums
    return result


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    control = controls()
    if args.self_test and args.output is None:
        print(json.dumps(dict(status=control['status'], constructed_sparse_controls=5,
                              subtraction_counterexamples=2, nonfinite_rejection_cases=3)))
        return
    if args.output is None:
        parser.error('Use --output or --self-test.')
    records, provenance = [], []
    pattern = re.compile(r'^(fixed|adaptive)_([01])_layer([0-9]+)_call([0-9]+)$')
    for size, suffix, revision, layers in PINS:
        directory = ROOT / 'outputs' / suffix
        capture, sanity_path = directory / 'actual_mlp_inputs.pt', directory / 'sanity.json'
        before = sha(capture), sha(sanity_path)
        tensors = torch.load(capture, map_location='cpu', weights_only=True)
        sanity = json.loads(sanity_path.read_text())
        if sanity['revision'] != revision or sanity['size'] != size:
            raise ValueError('The capture checkpoint does not match its pin.')
        selected_layers = [0, layers // 2, layers - 1]
        expected = {f'{mode}_{request}_layer{layer}_call{call}' for mode in ['fixed', 'adaptive']
                    for request in range(2) for layer in selected_layers for call in range(1, 21)}
        if set(tensors) != expected:
            raise ValueError('The retained call inventory does not match the bounded capture.')
        runs = {mode: [r for r in sanity['runs'] if r['mode'] == mode] for mode in ['fixed', 'adaptive']}
        if any(len(rows) != 2 for rows in runs.values()):
            raise ValueError('The source must contain exactly two runs per mode.')
        for key, tensor in tensors.items():
            if pattern.fullmatch(key) is None or tensor.dtype != torch.bfloat16 or tensor.device.type != 'cpu':
                raise ValueError('A capture key, dtype, or device differs.')
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Nonfinite capture input: {key}')
            call = int(key.rsplit('_call', 1)[1])
            if list(tensor.shape) != ([32, 2048] if call <= 4 else [1, 2048]):
                raise ValueError('The retained capture shape differs.')
            if sanity['activation_shapes'][key] != list(tensor.shape):
                raise ValueError('The raw shape manifest differs.')
        for mode in ['fixed', 'adaptive']:
            for request, run in enumerate(runs[mode]):
                if run['stats']['prefill_batches'] != 1 or len(run['result']) != 1:
                    raise ValueError('The source must have one unchunked prefill and one request.')
                native_request = run['result'][0]['request_id']
                if mode == 'fixed' and (set(run['depths'][native_request]) != {4}
                                        or run['stats']['recurrent_batches'] != 60):
                    raise ValueError('The fixed call-to-depth premise does not hold.')
                n = run['prompt_tokens']
                for layer in selected_layers:
                    for call in range(2, 21):
                        if call <= 4:
                            phase, depth = 'prefill', call - 1
                            positions = list(range(n - 32, n))
                        elif mode == 'fixed' and (call - 5) % 4 != 0:
                            phase, depth = 'fixed_decode', (call - 5) % 4
                            positions = [n + (call - 5) // 4]
                        else:
                            continue
                        previous = f'{mode}_{request}_layer{layer}_call{call - 1}'
                        current = f'{mode}_{request}_layer{layer}_call{call}'
                        for row, position in enumerate(positions):
                            measured = measure(tensors[previous][row], tensors[current][row])
                            records.append(dict(size=size, mode=mode, phase=phase, dataset_id=run['id'],
                                request_index=request, native_request_id=native_request,
                                request_scope=f'{size}:{mode}:{request}:{run["id"]}:{native_request}',
                                physical_layer=layer, absolute_position=position,
                                previous_depth=depth - 1, current_depth=depth, tensor_row=row,
                                previous_capture_key=previous, current_capture_key=current, **measured))
        if before != (sha(capture), sha(sanity_path)):
            raise ValueError('A raw capture or sanity file changed during the screen.')
        provenance.append(dict(size=size, revision=revision, layers=selected_layers,
            capture=str(capture.relative_to(ROOT)), capture_sha256=before[0],
            sanity=str(sanity_path.relative_to(ROOT)), sanity_sha256=before[1], raw_unchanged=True,
            captured_tensors=len(tensors), excluded_adaptive_decode_tensors=96,
            fixed_adaptive_prefill_word_identical=all(torch.equal(
                tensors[f'fixed_{request}_layer{layer}_call{call}'].view(torch.int16),
                tensors[f'adaptive_{request}_layer{layer}_call{call}'].view(torch.int16))
                for request in range(2) for layer in selected_layers for call in range(1, 5))))
    source_paths = [Path(__file__), ROOT / 'scripts/capture_and_sanity.py', ROOT / 'runtime.py',
        Path('/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py')]
    result = dict(schema_version=1, created_UTC=datetime.now(timezone.utc).isoformat(),
        evidence_class='CPU_STRUCTURAL_SCREEN', candidate='C06', new_card=False,
        premise='An exact sparse fresh input delta could reduce the dense residual projection.',
        scope='Two continuity prompts, three physical layers, retained prefill tails and first four fixed decode positions.',
        exclusions=['Adaptive decode', 'Cross-token pairs', 'Nonadjacent passes', 'Held-out evidence',
                    'Independent-token confidence intervals', 'Projection execution', 'Numerical equivalence',
                    'Quality evidence', 'GPU latency or speedup'],
        definitions=dict(numeric_equal='BF16 values compare equal. Opposite signed zeros compare equal.',
            word_equal='All 16 BF16 storage bits compare equal.',
            tile='Aligned, non-overlapping coordinates within one H=2048 row.',
            index='Optimistic 11-bit packed coordinate index, or a separate uint16 index control.',
            payload='New values or deltas plus coordinate indices only. Row counts and headers are omitted.',
            dense='One fresh H=2048 BF16 input row, 4096 bytes.',
            positions='Source-backed reconstruction from capture calls and prompt length, not a retained position tensor.',
            fp32_delta='Mathematical BF16 difference must fit FP32 exactly. Signed-zero word corrections are separate.'),
        optimistic_omitted_costs=['Base activation and base projection storage/availability',
            'Detecting changed coordinates', 'Index construction', 'Sparse gathers and kernel startup',
            'Weight reads and sparse contraction', 'Merge/addition and native BF16 rounding boundaries',
            'Normalization and nonlinear operations', 'Messages, alignment, counts, headers, and transport'],
        cdb_revision='abdfae0adbdd12be8c6123ccb34eee98a38afb39',
        source_sha256={str(path): sha(path) for path in source_paths},
        torch_version=torch.__version__, cpu_threads=torch.get_num_threads(), positive_controls=control, raw_provenance=provenance,
        summaries=summaries(records), row_pair_count=len(records), records=records)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write('\n')
    print(json.dumps(dict(output=str(args.output), row_pairs=len(records),
                         summaries=result['summaries']), indent=2))


if __name__ == '__main__':
    main()
