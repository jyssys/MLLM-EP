"""Validate captured physical facts; optional read-only local handoff checks."""
import argparse
import gzip
import hashlib
import subprocess
from collections import Counter
from datetime import datetime, timezone
from poc_dllm_attn_moe_pipeline.contract import *


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', action='store_true')
    args = parser.parse_args()
    result = ROOT/'results/main'
    env = read(result/'environment.json')
    assert sha(ROOT/'references/authoritative_spec.md') == env['spec_sha256']
    frozen = read(result/'design_freeze.json')['rows']
    assert len(frozen) == len({r['problem_id'] for r in frozen}) == 32
    assert Counter(r['split'] for r in frozen) == {'calibration': 8, 'heldout': 24}
    selected = [r for r in frozen if r['early_kill']]
    assert Counter(r['split'] for r in selected) == {'calibration': 4, 'heldout': 8}
    manifest = read(ROOT/'evidence/integrity.json')
    for row in manifest['files']:
        data = gzip.decompress((ROOT/row['packed']).read_bytes())
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        assert sha(ROOT/row['raw']) == row['sha256']
    ownership = [read(ROOT/f'raw/route_sanity_v4/ownership.rank{r}.json') for r in range(4)]
    expert_bytes = set()
    for rank, record in enumerate(ownership):
        assert record['physical_gpu'] == GPUS[rank]
        assert record['actual_cuda_uuid'] == UUIDS[rank]
        assert record['mask'] == MASK
        assert len(record['ownership']) == 19
        for layer in record['ownership']:
            assert layer['owned'] == list(range(rank*64, (rank+1)*64))
            assert layer['dispatcher'] == 'DeepEPDispatcher'
            assert all(w['dtype'] == 'torch.bfloat16' for w in layer['weights'].values())
            expert_bytes.add(sum(w['bytes'] for w in layer['weights'].values())//64)
    assert expert_bytes == {6291456}
    records = [read(ROOT/f'raw/route_sanity_v4/coding_00.trace.0.rank{r}.json') for r in range(4)]
    order = [(x['call_id'], x['layer']) for x in records[0]['routes']]
    for record in records:
        assert [(x['call_id'], x['layer']) for x in record['routes']] == order
        assert len(record['routes']) > 0
        for route in record['routes']:
            assert route['owner_local_ids_valid']
            assert len(route['ids']) == len(route['weights'])
            assert all(len(ids) == 8 and all(0 <= x < 256 for x in ids) for ids in route['ids'])
    native_counts = []
    for row in selected:
        rid = row['request_id']
        paths = list((ROOT/'raw/main_v1').glob(f'{rid}.*.rank*.json'))
        assert len(paths) == 16
        baseline = read(ROOT/f'raw/main_v1/{rid}.clean.0.rank0.json')
        for path in paths:
            record = read(path)
            assert record['output_ids'] == baseline['output_ids']
            assert record['nfe'] == baseline['nfe']
        native_counts.append(len(baseline['output_ids']))
    runs = []
    for name in ('main_v1', 'components_v2', 'components_v3', 'route_sanity_v4'):
        directory = ROOT/'raw'/name
        calls = sum(1 for p in directory.glob('*.rank0.json')
                    if any(s in p.name for s in ('.clean.', '.events.', '.trace.')))
        status = read(directory/'status.json') if (directory/'status.json').exists() else None
        runs.append(dict(run=name, full_model_generate_calls=calls,
                         complete=bool(status and status['ok']), failed_log_preserved=(directory/'full.log').exists()))
    assert sum(x['full_model_generate_calls'] for x in runs) == 96
    audit = dict(physical_EP4=True, ownership_layers=19, experts_per_rank=64,
                 loaded_expert_weight_bytes=6291456, actual_route_records_per_rank=len(order),
                 route_collective_order_equal=True, all_rank_native_token_nfe_parity=True,
                 compressed_evidence_lossless=True, packed_files=len(manifest['files']),
                 unique_frozen_requests=32, executed_unique_requests=12, untouched_requests=20,
                 native_requested_gen_length=64, native_returned_token_range=[min(native_counts), max(native_counts)],
                 native_length_rule='Unchanged dInfer rounds prompt+gen_length to block32; returned suffix can exceed64',
                 runs=runs, full_model_generate_calls=96, component_probes_separate=True,
                 candidate_runtime='NOT_RUN', task_accuracy='NOT_RUN')
    write(result/'integrity_audit.json', audit)
    write(result/'timing_ledger.json', dict(runs=runs, full_model_generate_calls=96,
                                           measured_speedup=False, full_runtime_candidate=False))
    if args.workspace:
        command = lambda *argv: subprocess.check_output(argv, text=True).strip()
        main_worktree = '/home/esjung/MLLM-EP-github'
        dirty_hashes = {}
        for scope, flags, expected in (
            ('staged', ['--cached'], '7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9'),
            ('unstaged', [], 'f56c2cbd47c4995364f072907ad7d886c935c3afe55d90e175a65abf0a4d941e')):
            digest = hashlib.sha256(subprocess.check_output(['git', '-C', main_worktree, 'diff', *flags, '--binary'])).hexdigest()
            assert digest == expected, 'Do not overwrite changes in the original worktree'
            dirty_hashes[scope] = digest
        assert sha(DINFER/'python/dinfer/model/modeling_llada2_moe_sglang.py') == env['source_sha256']
        source_diff = hashlib.sha256(subprocess.check_output(['git','-C',str(DINFER),'diff','--binary'])).hexdigest()
        assert source_diff == env['source_diff_sha256']
        apps = command('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader')
        target_apps = [line for line in apps.splitlines() if any(u in line for u in UUIDS)]
        assert not target_apps, f'Assigned GPU processes remain: {target_apps}'
        write(result/'handoff_environment.json', dict(
            checked_utc=datetime.now(timezone.utc).isoformat(),
            unchanged_original_worktree_diff_hashes=dirty_hashes, unchanged_external_source_diff=source_diff,
            gpu_mask=MASK, target_compute_processes=target_apps,
            gpu_state=command('nvidia-smi','--id='+MASK,'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'),
            burner_policy='OFF restored (initialOFF); no new service under authoritative spec section6',
            other_gpu_processes_untouched=True))
    print('EVIDENCE_AUDIT_PASS', len(manifest['files']), 'files / 96 native generate calls')


if __name__ == '__main__':
    main()
