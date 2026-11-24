"""Publish closed low-load kernels and compiled correctness, not live C32."""
import gzip
import hashlib
import json
from pathlib import Path
import time

from .common import ROOT, REPO, SOURCE, BASE, BRANCH, digest, preserve, write

TAGS = ('ec_LOW_CUPTI_C25', 'prefix_QUANTA_C26', 'ec_ROI_CUPTI_C27',
    'token_NUMERICAL_C28', 'token_NUMERICAL_C29', 'token_CORRECTNESS_C30', 'token_BATCH_C31')
DERIVED = TAGS + ('low_load_C25', 'short_capture_C27', 'prefix_quanta_E_D_C26',
    'compiled_token_C30', 'compiled_token_correctness_checkpoint', 'batch_token_C31')
REPORTS = ('mllm_epd_low_load_kernel_idle_validation_v1.md',
    'mllm_epd_high_load_short_capture_diagnostic_v1.md',
    'mllm_epd_encoder_decoder_prefix_quantum_cost_v1.md',
    'mllm_epd_compiled_token_correctness_v1.md',
    'mllm_epd_batched_token_prefix_correctness_v1.md')


def compress(source, folder):
    sha = digest(source)
    target = folder / (str(source.relative_to(ROOT)) + '.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        with gzip.GzipFile(filename='', fileobj=dst, mode='wb', mtime=0, compresslevel=6) as archive:
            while block := src.read(8 * 1024 ** 2):
                archive.write(block)
    check = hashlib.sha256()
    size = 0
    with gzip.open(target, 'rb') as stream:
        while block := stream.read(8 * 1024 ** 2):
            check.update(block)
            size += len(block)
    assert sha == check.hexdigest() == digest(source) and size == source.stat().st_size
    return dict(source=str(source.relative_to(REPO)), source_SHA256=sha, source_bytes=size,
        compressed_path=str(target.relative_to(REPO)), compressed_SHA256=digest(target),
        compressed_bytes=target.stat().st_size, lossless_verification='PASS')


def main():
    preserve()
    test = ROOT / 'logs/cpu_regression_v26.log'
    assert 'Ran 124 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    closed = []
    for path in (ROOT / 'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAGS):
            continue
        row = json.loads(path.read_text())
        assert row['finished'] and not Path('/proc', str(row['guardian'])).exists(), path
        closed.append(path)
    assert closed
    source_paths = {p for p in (ROOT / 'raw').iterdir()
        if p.is_file() and p.suffix in ('.json', '.jsonl') and
        p.name.startswith(TAGS + tuple('proxy_' + tag for tag in TAGS))}
    for profile in ['ec_LOW_CUPTI_C25_CPC_q0.5_v0', 'ec_ROI_CUPTI_C27_CPC_q3_v0']:
        folder = ROOT / 'raw/profiles' / profile
        assert folder.is_dir()
        source_paths.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.csv', '.json', '.jsonl'))
    folder = ROOT / 'archives/public_checkpoint_v4'
    manifest = ROOT / 'archives/checkpoint_raw_records_v4.json'
    assert not manifest.exists() and not folder.exists()
    folder.mkdir()
    compressed = [compress(p, folder) for p in sorted(source_paths)]
    write(manifest, dict(time=time.time(), passed=True, files=compressed,
        scope='Closed raw responses and two original kernel/copy traces. Gzip round-trip hashes match. '
            'Tensor payloads remain local with published hashes. No live C32 or shared append-only role log is included.'))
    readme = ROOT / 'README_checkpoint_v4.md'
    assert not readme.exists()
    readme.write_text('\n'.join([
        '# MLLM EPD checkpoint v4', '',
        'Status: ACTIVE and INCOMPLETE. Main100 remains active. Main400 remains unchanged.',
        'The low-load QPS0.5 CUPTI clean/profile/clean test passed all five frozen5% observer checks.',
        'Central-window E/P/D kernel busy was12.7326%/14.0682%/14.3716%.',
        'Simultaneous E/D physical kernel idle covered74.2376% of that window.',
        'This was low load, not a demonstrated prefill-heavy regime or safe extra-prefill budget.',
        'The shortened QPS3 capture failed observer checks. It remains DIAGNOSTIC.',
        'E/D prefix quantum cost controls also failed their whole-arm observer gates.',
        'Those costs cannot establish qualified safe quanta or an online serving improvement.',
        'The128-token probe imported a complete2691-row image embedding,19289088 bytes.',
        'That observation concerns image-level transport granularity. It is not visual-token compute complexity.',
        'A new common compiled BI0 policy passed15 serial correctness cases in each of two independent initializations.',
        'The original stock compiled strict failures remain unchanged.',
        'The new policy controls initial reduction candidates and disables incompatible fusion benchmarking.',
        'Original BF16 weights, model forward, attention, vision and processor remain unchanged.',
        'The separate concurrent N8 control passed all28 imported-prefix KV hashes and query MRoPE checks.',
        'Actual P and D batches reached3, not8. Each request generated only one first output.',
        'This does not establish full suffix cross-batch equivalence, task quality or active-decode safety.',
        'Task quality remains PENDING. Token C3 serving speedup and safe headroom remain unproved.',
        'CPU regression:124PASS. The earlier failing CPU control log is preserved.',
        'Common-policy Native C32 is live and excluded from these evidence records.',
        'No final goal completion or GPU closeout is claimed. Approved burners cover unused GPUs.', '']))
    paths = list(SOURCE.glob('*.py')) + [test, manifest, readme] + closed
    paths += [ROOT.parent / report for report in REPORTS]
    for directory in ['analysis', 'configs', 'state', 'logs', 'figures']:
        for path in (ROOT / directory).iterdir():
            name = path.name.removeprefix('server_')
            if path.is_file() and (name.startswith(DERIVED) or
                directory == 'logs' and name.startswith(('reduction_initial_configs_', 'deterministic_combo_', 'cpu_regression_v24'))):
                if path.stat().st_size < 90 * 1024 ** 2:
                    paths.append(path)
    for path in (ROOT / 'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith(TAGS):
            paths.append(path)
    paths += list((ROOT / 'raw/batch_prefix').glob('token_BATCH_C31*/manifest.json'))
    paths += [p for p in (ROOT / 'archives').glob('*.source.zip') if p.name.startswith(TAGS)]
    paths += [REPO / row['compressed_path'] for row in compressed]
    paths = sorted(set(paths))
    assert all(path.is_file() and path.stat().st_size < 90 * 1024 ** 2 for path in paths)
    selection = ROOT / 'state/checkpoint_publication_selection_v4.json'
    write(selection, dict(time=time.time(), status='INCOMPLETE_RESEARCH_CHECKPOINT',
        files=len(paths), total_bytes=sum(p.stat().st_size for p in paths),
        closed_server_leases=len(closed), excluded_live_jobs=['ec_COMMON_C32'], final_goal_complete=False,
        user_checkout_and_index='Preserved through a private Git index'))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT, helper.REPO, helper.SOURCE = ROOT, REPO, SOURCE
    helper.BASE, helper.BRANCH = BASE, BRANCH
    helper.preserve, helper.write, helper.digest = preserve, write, digest
    result = helper.commit_paths(paths,
        'WIP MLLM EPD: qualified low-load kernels, compiled exact token cases and concurrent KV handoffs',
        'publication_commit_checkpoint_v4.json')
    assert result['passed'] and result['remote_verified']
    preserve()
    print('CHECKPOINT_V4_REMOTE_VERIFIED', result['remote_SHA'], flush=True)


if __name__ == '__main__':
    main()
