"""Read retained C07 evidence on CPU. Do not execute a model or change raw files."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import torch

from poc_ouro_parallelism_deep_discovery.c07_fixture import compare_bf16_words
from poc_ouro_parallelism_deep_discovery.fixture_archive import read_manifest, check_stream


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(value):
    if torch.is_tensor(value):
        result = dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
        if value.numel() <= 100:
            result['values'] = value.tolist()
        return result
    return value


def equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left, right))
    return left == right


def difference(name, actual, reference):
    result = compare_bf16_words(name, actual, reference, torch)
    a, r = actual.to(torch.float64), reference.to(torch.float64)
    d = a - r
    reference_l2 = float(torch.linalg.vector_norm(r))
    difference_l2 = float(torch.linalg.vector_norm(d))
    result.update(actual_finite=bool(torch.isfinite(a).all()),
                  reference_finite=bool(torch.isfinite(r).all()),
                  max_absolute_difference=float(d.abs().max()),
                  mean_absolute_difference=float(d.abs().mean()),
                  difference_l2=difference_l2, reference_l2=reference_l2,
                  relative_l2=difference_l2 / reference_l2 if reference_l2 else None,
                  metric_arithmetic='CPU float64 from retained BF16 values')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assert-bitwise', action='store_true')
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    if not job.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Use one explicit campaign job directory.')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'analysis'):
        raise ValueError('Write derived evidence only in the campaign analysis directory.')
    paths = [job / name for name in ['capture_finish.json', 'service.json', 'pre_run_contract.json',
              'preparation.json', 'rank0/manifest.json', 'rank0/fixture.archive-v1/manifest.json']]
    before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    path, archive = read_manifest(paths[-1])
    stream = io.BytesIO()
    archive_check = check_stream(path.parent, archive, stream)
    stream.seek(0)
    bundle = torch.load(stream, map_location='cpu', weights_only=True)
    if any(value.device.type != 'cpu' for value in bundle.values() if torch.is_tensor(value)):
        raise ValueError('The diagnostic requires CPU tensors.')
    manifest = json.loads(paths[-2].read_text())
    service = json.loads((job / 'service.json').read_text())
    selected = manifest['selected']
    if selected != [[2, 2]]:
        raise ValueError('This diagnostic is bounded to the observed row 2, depth 2 failure.')
    row, depth = selected[0]
    first = next(iter(bundle['native_metadata'].values()))
    fixture = bundle['fixture_metadata']
    metadata_delta = {name: dict(native=describe(value), fixture=describe(fixture[name]))
                      for name, value in first.items() if not equal(value, fixture[name])}
    checks = [difference('normalized_pass_output', bundle['actual_output'], bundle['fixture_reference'])]
    current = []
    for layer, (actual, reference) in enumerate(zip(bundle['actual_current'][0], bundle['current'][0], strict=True)):
        current.append(dict(layer=layer, combined=difference(f'current_KV.layer{layer}', actual, reference),
                            key=difference(f'current_K.layer{layer}', actual[..., :128], reference[..., :128]),
                            value=difference(f'current_V.layer{layer}', actual[..., 128:], reference[..., 128:])))
    checks.extend([
        difference('selected_input_identity', bundle['fixture_input'], bundle['inputs'][depth][row:row + 1]),
        difference('selected_reference_identity', bundle['fixture_reference'], bundle['outputs'][depth][row:row + 1]),
    ])
    checks.extend(difference(f'recurrent_boundary.{i}_to_{i+1}', bundle['outputs'][i], bundle['inputs'][i + 1])
                  for i in range(3))
    raw_checks = service['ranks'][0]['parity']['checks']
    raw_histories = [r for r in raw_checks if r['name'].startswith('history_KV.')]
    raw_sources = [r for r in raw_checks if r['name'].startswith('original_history_KV.')]
    source_pins = json.loads((job / 'pre_run_contract.json').read_text())
    source_hashes = {path: dict(recorded=expected, current=sha(path), matches=sha(path) == expected)
                     for path, expected in source_pins['native_sources_sha256'].items()}
    for field, relative in [('helper', 'c07_fixture.py'), ('worker', 'c07_fixture_probe.py'),
                            ('runner', 'scripts/c07_one_pass_fixture.py')]:
        current_hash = sha(ROOT / relative)
        source_hashes[relative] = dict(recorded=source_pins[f'{field}_sha256'], current=current_hash,
                                       matches=current_hash == source_pins[f'{field}_sha256'])
    native_root = Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages')
    for relative in ['vllm/model_executor/layers/linear.py', 'vllm/model_executor/layers/utils.py',
                     'vllm/model_executor/layers/attention/attention.py', 'vllm/model_executor/layers/layernorm.py',
                     'vllm/vllm_flash_attn/flash_attn_interface.py']:
        source_hashes[str(native_root / relative)] = dict(current=sha(native_root / relative),
                                                         recorded=None, matches=None)
    protected = set(manifest['preparation']['protected_pages'])
    pages = manifest['preparation']['pages'][0]
    after = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    if before != after:
        raise ValueError('A raw JSON input changed during the diagnostic.')
    result = dict(schema_version=1, created_UTC=datetime.now(timezone.utc).isoformat(),
                  scope='Retained CPU tensor diagnosis only. No model execution or new GPU evidence.',
                  job=str(job.relative_to(ROOT)), archive_check=archive_check,
                  fixture_original_sha256_matches_rank_manifest=archive['original_sha256'] == manifest['fixture_sha256'],
                  checks=checks, current_kv_by_layer=current, metadata_delta=metadata_delta,
                  unchanged_metadata={name: describe(value) for name, value in first.items()
                                      if equal(value, fixture[name])},
                  native_metadata_entries=len(bundle['native_metadata']),
                  native_metadata_all_equal=all(all(equal(value, item[name]) for name, value in first.items())
                                               for item in bundle['native_metadata'].values()),
                  positions=bundle['positions'].tolist(),
                  native_padding=bundle['native_forward_context']['is_padding'].tolist(),
                  source_pages_selected=bundle['source_table'][row, :65].tolist(),
                  fixture_pages=pages, fixture_pages_disjoint_from_all_sources=not (set(pages) & protected),
                  fixture_pages_exclude_null=0 not in pages,
                  fixture_slot_mapping_matches_pages=fixture['slot_mapping'].tolist() == [pages[64] * 16],
                  fixture_table_matches_lease=fixture['block_table'][0, :65].tolist() == pages,
                  raw_gpu_check_summary=dict(destination_history_checks=len(raw_histories),
                    destination_history_all_equal=all(r['equal'] for r in raw_histories),
                    destination_history_words=sum(r['total_words'] for r in raw_histories),
                    source_history_checks=len(raw_sources), source_history_all_equal=all(r['equal'] for r in raw_sources),
                    source_history_words=sum(r['total_words'] for r in raw_sources),
                    service_trials=len(service['ranks'][0]['service'])),
                  raw_json_sha256=before, raw_json_unchanged=before == after,
                  source_sha256=source_hashes, torch_version=torch.__version__,
                  script_sha256=sha(__file__))
    with output.open('x') as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write('\n')
    print(json.dumps(dict(output=str(output), output_comparison=checks[0],
                         source_hashes_match=all(row['matches'] is not False for row in source_hashes.values()),
                         raw_gpu_check_summary=result['raw_gpu_check_summary']), indent=2))
    if args.assert_bitwise:
        assert checks[0]['equal'], 'RETAINED_FIXTURE_BITWISE_FAILURE'


if __name__ == '__main__':
    main()
