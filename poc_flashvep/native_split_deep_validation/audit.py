"""Freeze previous evidence, source consistency, and data separation checks."""
import csv
import hashlib
import subprocess
from collections import Counter
from .common import *


def main():
    initialize()
    required = ['serving_results.csv', 'paired_start_ratios.json', 'graph_cache_observations.csv',
                'primitive_costs.json', 'state/effective_qualification.json',
                'structural_frontiers_not_gpu_costs.csv', 'figure_manifest.json', 'failures.json']
    records = {}
    for name in required:
        path = PREVIOUS / name
        value = list(csv.DictReader(path.open())) if path.suffix == '.csv' else json.loads(path.read_text())
        records[name] = dict(SHA256=digest(path), records=len(value))
    failures = json.loads((PREVIOUS / 'failures.json').read_text())
    failure_audit = []
    for item in failures:
        path = PREVIOUS / 'logs' / (item['name'] + '.log')
        text = path.read_text(errors='replace')
        errors = [line for line in text.splitlines()
                  if line.startswith(('TypeError:', 'ValueError:', 'AttributeError:', 'RuntimeError:', 'AssertionError:'))]
        failure_audit.append(dict(name=item['name'], log_SHA256=digest(path), errors=errors,
            wavefront_guard='wavefront prefill requires one recurrent depth per scheduling step' in text))
    code = {}
    source = REPO / 'poc_flashvep/native_loop_2d_oracle'
    for path in source.rglob('*.py'):
        relative = str(path.relative_to(REPO))
        blob = subprocess.check_output(['git', 'show', BASE + ':' + relative], cwd=REPO)
        code[relative] = dict(disk_SHA256=digest(path), parent_SHA256=hashlib.sha256(blob).hexdigest(),
                              matches=blob == path.read_bytes())
    assert all(row['matches'] for row in code.values())
    data = {}
    old = json.loads((REPO / 'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover/data/long_source_pool.json').read_text())
    for model, length in (('ouro', 4096), ('huginn', 2048)):
        prior_ids = {row['prompt_id'] for row in old}
        prior_prefixes = {tuple(row['input_ids'][model][:128]) for row in old}
        pools = {split: json.loads((ROOT / 'data' / f'{split}_{model}_L{length}.json').read_text())
                 for split in ('calibration', 'heldout')}
        for split, rows in pools.items():
            assert all(len(row['input_ids'][model]) == length for row in rows)
            assert len({row['prompt_id'] for row in rows}) == len(rows)
            assert not prior_ids & {row['prompt_id'] for row in rows}
            assert not prior_prefixes & {tuple(row['input_ids'][model][:128]) for row in rows}
            data[model + '_' + split] = dict(count=len(rows), length=length,
                unique_prompts=len({row['prompt_id'] for row in rows}),
                source_distribution=dict(Counter(row['source_dataset'] for row in rows)),
                prior_ID_overlap=0, prior_first128_token_overlap=0)
        assert not {row['prompt_id'] for row in pools['calibration']} & {row['prompt_id'] for row in pools['heldout']}
    write(ROOT / 'state/previous_evidence_and_data_audit.json', dict(time=time.time(),
        required_evidence=records, failures=failure_audit, reused_native_source=code, data=data,
        original_index_preserved=True, external_runtime_preserved=True))
    prior.parent.prior.preserve()
    print(json.dumps(dict(previous_failure_logs_read=len(failure_audit), source_files_checked=len(code), data=data), indent=2))


if __name__ == '__main__':
    main()
