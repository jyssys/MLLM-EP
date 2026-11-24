"""Publish closed Native/CPU evidence while the separate baseline job continues."""
import gzip
import hashlib
import json
from pathlib import Path
import time

from .common import ROOT, REPO, SOURCE, BASE, BRANCH, digest, preserve, write


def closed_raw_archive():
    folder = ROOT / 'archives/public_checkpoint_v1'
    folder.mkdir(exist_ok=True)
    target_manifest = ROOT / 'archives/checkpoint_raw_records_v1.json'
    assert not target_manifest.exists()
    files = []
    for source in sorted((ROOT / 'raw').glob('*.jsonl')):
        if not source.name.startswith(('ec_', 'proxy_ec_', 'NVML_ec_')):
            continue
        if source.name.startswith('ec_cuda_ipc_'):
            continue  # Shared live log: later E/PD starts will append to it.
        before = digest(source)
        target = folder / (source.name + '.gz')
        with source.open('rb') as src, target.open('xb') as stream:
            with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0, compresslevel=6) as dst:
                while block := src.read(8 * 1024**2):
                    dst.write(block)
        sha = hashlib.sha256()
        size = 0
        with gzip.open(target, 'rb') as stream:
            while block := stream.read(8 * 1024**2):
                sha.update(block)
                size += len(block)
        assert sha.hexdigest() == before == digest(source) and size == source.stat().st_size
        files.append(dict(source=str(source.relative_to(REPO)), source_SHA256=before, source_bytes=size,
            compressed_path=str(target.relative_to(REPO)), compressed_SHA256=digest(target),
            compressed_bytes=target.stat().st_size, lossless_verification='PASS'))
    write(target_manifest, dict(time=time.time(), files=files, passed=True,
        scope='Closed ec_* Native controls and independent starts only. Live C14 and shared role logs are excluded. '
              'Nested GPU traces/tensors remain local. This is not the final evidence archive.'))
    return files, target_manifest


def main():
    preserve()
    test_log = ROOT / 'logs/cpu_regression_token_v10.log'
    text = test_log.read_text()
    assert 'Ran 71 tests' in text and text.rstrip().endswith('OK')
    native = ROOT / 'analysis/native_inline_independent_N3_v0.json'
    assert json.loads(native.read_text())['independent_starts'] == 3
    for start in [1, 2, 3]:
        assert json.loads((ROOT / 'analysis' / f'ec_NATIVE_C11_S{start}.json').read_text())['status'] == 'COMPLETE'
    raw, manifest = closed_raw_archive()
    from .source_snapshot import snapshot
    source = snapshot('publication_checkpoint_v1')
    proof = ROOT / 'state/checkpoint_cpu_verification_v1.json'
    write(proof, dict(time=time.time(), status='PASS', tests=71, log_SHA256=digest(test_log),
        scope='CPU regressions. No Generic Token Split GPU correctness or online success is established.'))
    readme = ROOT / 'README_checkpoint_v1.md'
    assert not readme.exists()
    readme.write_text('\n'.join([
        '# MLLM EPD checkpoint v1', '',
        'Status: ACTIVE and INCOMPLETE. Earlier checkpoint documents remain unchanged.',
        'Main100 is the active workload. Main400 and original prompts are preserved.',
        'Native EPD completed three independent starts at QPS2, with Natural EOS and equal-output replay kept separate.',
        'Natural start-mean TTFTp95 is4880.240ms, ITLp99 is16.267ms, TPOTp95 is6.827ms, and Joint SLO is91.667%.',
        'These values do not establish a token-split speedup or complete task quality.',
        'The official MMLongBench-Doc extraction pipeline remains unavailable without an authorized external API.',
        'Native CPU/IPC and publication/ACK interventions preserve their negative results and scopes.',
        'C11S2 and C13 GPU profiles failed the frozen5% observer gate. Their kernel/idle results stay diagnostic.',
        'C12 failed during initialization because identical source archives shared an extraction directory.',
        'The failure was preserved. A red-capable regression test now covers the per-boot identity fix.',
        'Future server processes execute verified extracted source snapshots.',
        'Generic token ownership, exact remote prefix counting, original multimodal metadata and independent image-consumer plans have CPU tests.',
        'Generic token-split GPU correctness, active-decode insertion, and online serving gains remain NOT_RUN.',
        'The new stock Agg DP3/E+PD DP2/Native three-GPU Natural sweep is running separately asC14.',
        'No Source Credit, D Cap or original model forward was weakened for that comparison.', '',
        '## Evidence', '',
        '- Main100 manifest SHA256:910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1.',
        '- Model/processor revision:cc594898137f460bfe9f0759e9844b3ce807cfb5.',
        '- Native table:analysis/native_inline_independent_N3_v0.csv and matchingJSON.',
        '- CPU tests:71PASS. GPU model correctness and task scoring are separate gates.',
        '- archives/checkpoint_raw_records_v1.json verifies the closed Native raw subset losslessly.',
        '- Nested GPU traces,tensors,media and active baseline logs remain local at this checkpoint.',
        '- No final resource closeout is claimed. Approved burners remain on unused GPUs.', '',
    ]))
    paths = list(SOURCE.glob('*.py')) + [proof, test_log, native, manifest, readme]
    reports = ['mllm_epd_native_runtime_confirmation_v1.md', 'mllm_epd_ec_transport_and_ack_diagnostic_v1.md',
               'mllm_epd_token_prefix_contract_v0.md']
    paths.extend(ROOT.parent / name for name in reports)
    for directory, patterns in [
        ('analysis', ['ec_*.json', 'ec_*.csv', 'native_EC_*v1.json', 'native_EC_*v1.csv', 'native_IPC_*v0.json',
                      'native_inline_independent_N3_v0.csv']),
        ('configs', ['ec_*.json', 'native_IPC_*.json']),
        ('state', ['server_ec_*.json', 'ec_*correctness*.json', 'ec_*failure*.json', 'ec_*ownership*.json']),
        ('figures', ['native_EC_*v1.png', 'native_EC_*v1.pdf'])]:
        for pattern in patterns:
            paths.extend((ROOT / directory).glob(pattern))
    paths.extend(REPO / row['compressed_path'] for row in raw)
    paths.extend((ROOT / 'archives').glob('*.source.zip'))
    paths = sorted(set(paths))
    local = [dict(path=str(p.relative_to(REPO)), bytes=p.stat().st_size, SHA256=digest(p))
             for p in paths if p.stat().st_size >= 90 * 1024**2]
    assert all('/analysis/' in row['path'] for row in local)
    paths = [p for p in paths if p.stat().st_size < 90 * 1024**2]
    selection = ROOT / 'state/checkpoint_publication_selection_v1.json'
    write(selection, dict(time=time.time(), status='INCOMPLETE_RESEARCH_CHECKPOINT', files=len(paths),
        total_bytes=sum(p.stat().st_size for p in paths), local_only_derived=local,
        excluded_live_job='C14 baseline confirmation', final_goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT, helper.REPO, helper.SOURCE = ROOT, REPO, SOURCE
    helper.BASE, helper.BRANCH = BASE, BRANCH
    helper.preserve, helper.write, helper.digest = preserve, write, digest
    result = helper.commit_paths(paths,
        'WIP MLLM EPD Main100: independent Native N3, exact token CPU contracts and preserved profiler failures',
        'publication_commit_checkpoint_v1.json')
    assert result['passed'] and result['remote_verified']
    preserve()
    print('CHECKPOINT_V1_REMOTE_VERIFIED', result['remote_SHA'])


if __name__ == '__main__':
    main()
