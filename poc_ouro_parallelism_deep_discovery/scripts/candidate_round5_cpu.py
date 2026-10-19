"""Bounded CPU controls. This script does not execute a GPU or test quality."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import struct


ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', f32(value)))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def controls():
    tiny = 2.0 ** -133
    rounded_activation = bf16(f32(tiny * 0.5))
    staged = bf16(f32(rounded_activation * (2.0 ** 16)))
    fused = bf16(f32(f32(tiny * 0.5) * (2.0 ** 16)))
    assert staged == 0 and fused != 0
    values = [float(2 ** 25), 3.0, -float(2 ** 25)]
    sequential = 0.0
    for value in values:
        sequential = f32(sequential + value)
    assert sequential == 4.0 and sum(values) == 3.0
    return {
        'evidence_class': 'CONSTRUCTED_CPU_ALGEBRA_CONTROL',
        'no_model_quality_or_GPU_parity': True,
        'premature_rounding': {
            'gate': tiny, 'up': 2 ** 16,
            'BF16_activation_then_product': staged,
            'FP32_activation_and_product_then_BF16': fused,
            'limit': 'IEEE-style FP32 subnormals. Not a GPU FTZ or exp implementation test.'},
        'exact_dot_vs_FP32_order': {
            'BF16_input_x': [4096, 3, 4096],
            'BF16_input_w': [8192, 1, -8192],
            'real_dot': 3, 'declared_FP32_serial_dot': sequential,
            'limit': 'This serial order is not an identified cuBLAS or Tensor Core reduction order.'},
        'shared_scale_small_counterexample': {
            'values': [1.0, 2 ** -16], 'common_quantum': 2 ** -16,
            'integer_magnitudes': [2 ** 16, 1],
            'required_magnitude_bits': 17,
            'base128_signed_digit_slices': math.ceil(17 / 7),
            'limit': 'Rejects one shared-scale INT8 representation, not all integer algorithms.'},
        'zero_times_nonfinite': {
            'zero_times_infinity_is_NaN': math.isnan(0.0 * float('inf')),
            'limit': 'A zero-gate certificate also needs a finite unused up result or a defined exceptional path.'}
    }


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_screen():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    torch.set_num_threads(4)
    tz = torch.tensor([8] + [(n & -n).bit_length() - 1 for n in range(1, 256)], dtype=torch.int32)
    high = torch.tensor([0] + [n.bit_length() - 1 for n in range(1, 256)], dtype=torch.int32)

    def row_spans(tensor):
        assert tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16
        raw = tensor.contiguous().view(torch.int16).to(torch.int32) & 0x7fff
        exponent = raw >> 7
        fraction = raw & 127
        assert not (exponent == 255).any()
        mantissa = torch.where(exponent == 0, fraction, fraction + 128)
        base = torch.where(exponent == 0, -133, exponent - 134)
        active = mantissa != 0
        low = torch.where(active, base + tz[mantissa], 10000).amin(dim=-1)
        top = torch.where(active, base + high[mantissa], -10000).amax(dim=-1)
        span = torch.clamp(top - low + 1, min=0)
        return span, (span + 6) // 7

    unit_span, unit_slices = row_spans(torch.tensor([[1.0, 1.5]], dtype=torch.bfloat16))
    wide_span, wide_slices = row_spans(torch.tensor([[1.0, 2 ** -16]], dtype=torch.bfloat16))
    assert (int(unit_span.item()), int(unit_slices.item())) == (2, 1)
    assert (int(wide_span.item()), int(wide_slices.item())) == (17, 3)

    configs = [
        ('1.4B', '574fa66cb8bf5abdc979642d01cf2b79b16bfab1', 'capture_base14_v2/1.4B'),
        ('2.6B', '1ed04250da1a9936042725d302e81c8fa2ab5abd', 'capture_base26_v1/2.6B')]
    result = []
    for size, revision, suffix in configs:
        capture = ROOT / 'outputs' / suffix / 'actual_mlp_inputs.pt'
        weights = Path('/home/esjung/.cache/huggingface/hub') / (
            'models--ByteDance--Ouro-' + size) / 'snapshots' / revision / 'model.safetensors'
        tensors = torch.load(capture, map_location='cpu', weights_only=True)
        records = []
        with safe_open(weights, framework='pt', device='cpu') as handle:
            cache = {}
            for key in sorted(tensors):
                match = re.fullmatch(r'fixed_(\d+)_layer(\d+)_call(\d+)', key)
                if not match or int(match[3]) not in range(5, 9):
                    continue
                layer = int(match[2])
                x = tensors[key]
                assert x.shape == (1, 2048) and x.device.type == 'cpu'
                if layer not in cache:
                    weight = handle.get_tensor(f'model.layers.{layer}.mlp.gate_proj.weight')
                    span, slices = row_spans(weight)
                    cache[layer] = (weight.double(), {
                        'rows': weight.shape[0], 'span_min': int(span.min()),
                        'span_max': int(span.max()), 'slice_min': int(slices.min()),
                        'slice_max': int(slices.max()),
                        'slice_median': float(slices.double().median())})
                weight64, weight_stats = cache[layer]
                gate = F.linear(x.double(), weight64).to(torch.bfloat16)
                activated = F.silu(gate.float()).to(torch.bfloat16)
                assert torch.isfinite(gate).all()
                xspan, xslices = row_spans(x)
                records.append({
                    'capture_key': key, 'layer': layer, 'rows': 1,
                    'gate_channels': gate.numel(),
                    'zero_gate_values': int((gate == 0).sum()),
                    'zero_rounded_SiLU_values': int((activated == 0).sum()),
                    'complete_zero_tiles_width32': int((activated.reshape(-1, 32) == 0).all(-1).sum()),
                    'gate_min': float(gate.min()), 'gate_max': float(gate.max()),
                    'input_shared_scale_span': int(xspan.item()),
                    'input_base128_slices': int(xslices.item()),
                    'gate_weight_row_representation': weight_stats,
                    'minimum_complete_pair_products_for_best_weight_row': int(xslices.item()) * weight_stats['slice_min']})
        result.append({
            'size': size, 'revision': revision, 'capture': str(capture.relative_to(ROOT)),
            'capture_sha256': sha(capture), 'records': records,
            'row_denominator': len(records),
            'channel_denominator': sum(r['gate_channels'] for r in records),
            'zero_gate_values': sum(r['zero_gate_values'] for r in records),
            'zero_rounded_SiLU_values': sum(r['zero_rounded_SiLU_values'] for r in records),
            'complete_zero_tiles_width32': sum(r['complete_zero_tiles_width32'] for r in records)})
    return {
        'evidence_class': 'BOUNDED_CPU_CHECKPOINT_COMPONENT_SCREEN',
        'representation_unit_controls': 'PASS. Narrow row needs one slice. Wide row needs three slices.',
        'method': 'FP64 CPU gate projection, BF16 gate rounding, FP32 CPU SiLU, BF16 activation rounding.',
        'sample': 'Earlier two-prompt captures. Three physical layers. First four fixed-depth decode calls per layer/request.',
        'limits': [
            'Not the captured native GPU gate or activation output.',
            'No GPU kernel equivalence or model-quality claim.',
            'Fixed-R4 rows only. No population-level rate or confidence interval.',
            'Shared-scale base128 digit accounting is one representation, not every exact algorithm.',
            'The minimum pair count assumes complete pair expansion, not a performance lower bound.'],
        'models': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-screen', action='store_true')
    args = parser.parse_args()
    begin = dt.datetime.now(dt.timezone.utc).isoformat()
    output = {'schema': 'candidate_round5_cpu_v1', 'begin_utc': begin, 'controls': controls()}
    if args.checkpoint_screen:
        output['checkpoint_screen'] = checkpoint_screen()
    output['end_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
