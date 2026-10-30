"""New, matched serving epoch after a full output and KV regression passes."""
import hashlib
import json
import time
from pathlib import Path

from .common import ROOT, REPO, GPU_STOP, write
from .jobs import configuration, parity, output_rows
from .repair_priority import enqueue


EPOCH = 'canonical_core_linear_bf16_v1'
VARIANT = 'canonical_all'


def wait(names):
    while time.time() + 600 < GPU_STOP:
        states = []
        for name in names:
            path = ROOT / 'raw' / name / 'status.json'
            states.append(json.loads(path.read_text()).get('status') if path.exists() else 'PENDING')
        if all(state in ('COMPLETE', 'FAILED', 'TIME_BUDGET_SKIPPED') for state in states):
            return states
        write(ROOT / 'state/repair_heartbeat.json', dict(waiting_for=names, time=time.time()))
        time.sleep(5)
    return ['TIME_BUDGET_SKIPPED'] * len(names)


def job(model, arm, count, rate, seed, name, order, diagnostic, gpus=(0, 1), **extra):
    lock_path = ROOT / 'configs/repair_candidate_source_lock.json'
    if lock_path.exists():
        locked = json.loads(lock_path.read_text())['files']
        for filename, expected in locked.items():
            path = REPO / 'poc_flashvep/looped_pd_b_slo_serving' / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Repair source changed after its candidate lock: ' + filename)
    environment = dict(B_REPAIR_VARIANT=VARIANT, B_REPAIR_RUN_DIR=str(ROOT / 'raw' / name),
                       B_REPAIR_EPOCH=EPOCH)
    environment.update(extra.pop('env', {}))
    value = configuration(model, arm, 'diagnostic' if diagnostic else 'W1', count, rate, seed,
                          gpus=gpus, dataset=extra.pop('dataset', 'W1'), cap=extra.pop('cap', 128),
                          capacity=extra.pop('capacity', 8),
                          timeout=2400, diagnostic=diagnostic, execution_epoch=EPOCH,
                          repair_order=order, name=name,
                          python=str(REPO / 'poc_flashvep/looped_pd_b_slo_serving/repair_python.py'),
                          env=environment, **extra)
    enqueue(value)
    return value


def main(checks, variant='canonical_all'):
    global VARIANT, EPOCH
    VARIANT = variant
    EPOCH = 'canonical_core_linear_bf16_v1' if variant == 'canonical_all' else 'canonical_linear_bf16_v1'
    qualified = {}
    source = REPO / 'poc_flashvep/looped_pd_b_slo_serving/repair_hooks.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    for model, rate in (('ouro', 1.), ('huginn', .2)):
        full = next((row for row in checks if row['stage']['model'] == model and row['stage']['count'] == 200), None)
        if not full or not full['result']['passed']:
            qualified[model] = dict(passed=False, reason='Full200-request output regression did not pass')
            continue
        from .analyze import summarize
        full_work = []
        for run_name in full['stage']['runs']:
            value = json.loads((ROOT / 'configs/jobs' / (run_name + '.json')).read_text())
            _, _, measured = summarize(value)
            full_work.append(measured.get('issued_loop_rows_equal_structural_full_depth') is True)
        stop_exact = all(value['stop_equal'] == 200 for value in full['result']['comparisons'])
        if not all(full_work) or not stop_exact:
            qualified[model] = dict(passed=False, reason='Full-depth work or stop-reason check failed',
                                    full_work=full_work, stop_exact=stop_exact)
            continue
        name = f'repair_epoch_KV_{model}_N8_B'
        job(model, 'B', 8, rate, 2026103001, name, 7., True, cap=8,
            env=dict(ABC_VERIFY_KV='1', ABC_REVERSE_FRAGMENTS='1'))
        states = wait([name])
        folder = ROOT / 'raw' / name
        closed = folder / 'closed_stats.json'
        tids = {row['transfer_id'] for row in json.loads(closed.read_text()).get('metrics', [])
                if row['request_id'].startswith('r0_')} if closed.exists() else set()
        fingerprints = [json.loads(line) for path in folder.glob('KV_checks-*.jsonl')
                        for line in path.read_text().splitlines()]
        fingerprints = [row for row in fingerprints if row['tid'] in tids]
        passed = states == ['COMPLETE'] and len(fingerprints) == 8 and all(row['passed'] for row in fingerprints)
        hashes = [json.loads(path.read_text())['source_SHA256']
                  for name in full['stage']['runs'] for path in (ROOT / 'raw' / name).glob('repair_override_*.json')]
        repair_hashes = {name: hashlib.sha256(source.with_name(name).read_bytes()).hexdigest()
                         for name in ('repair_hooks.py', 'repair_coda.py')}
        source_manifests = [json.loads(path.read_text()).get('repair_sources_SHA256')
                            for name in full['stage']['runs'] for path in (ROOT / 'raw' / name).glob('repair_override_*.json')]
        all_sources_match = bool(source_manifests) and all(value == repair_hashes for value in source_manifests)
        passed = passed and bool(hashes) and all(value == source_hash for value in hashes) and all_sources_match
        qualified[model] = dict(passed=passed, full_output_regression=full,
                                KV_run=name, KV_exact=sum(row['passed'] for row in fingerprints),
                                KV_N=len(fingerprints), repair_source_SHA256=source_hash,
                                full_depth_work_checks=full_work, stop_reason_exact=stop_exact,
                                repair_source_manifest=repair_hashes, all_repair_sources_match=all_sources_match,
                                source_hash_matches_qualification=bool(hashes) and all(value == source_hash for value in hashes))
        old_control = f'{model}_W1_PD_G1_N2_rate{rate:g}_seed2026103001'
        qualified[model]['pre_repair_vs_post_repair_control'] = parity(old_control, full['stage']['runs'][0])
        if passed and time.time() + 1200 < GPU_STOP:
            reference_name = f'repair_epoch_reference_{model}_Native_N8'
            job(model, 'Native', 8, 0., 20261029, reference_name, 7.01, True,
                cap=32, gpus=(0,), dataset='correctness', capacity=1)
            wait([reference_name])
            qualified[model]['fresh_native_vs_saved_official_HF'] = parity(reference_name, f'{model}_reference_HF')
            qualified[model]['reference_scope'] = 'Eight-prompt numerical triage only. Not answer accuracy or a resolved original-model-fidelity gate.'
        write(ROOT / 'configs/repair_epoch_qualification.json', dict(epoch=EPOCH, models=qualified,
              original_failure_files='Preserved. Original epoch remains rejected.',
              original_reference='FIDELITY_UNRESOLVED', original_SLO_lock='Unchanged'))
    records = []
    source_prompts = json.loads((ROOT / 'data/W1.json').read_text())
    cycles = [dict(row, source_prompt_id=row['prompt_id'], occurrence=occurrence,
                   prompt_id=f"{row['prompt_id']}::repair_occurrence{occurrence}")
              for occurrence in range(2) for row in source_prompts]
    cycles_path = ROOT / 'data/W1_repair_cycles.json'
    if cycles_path.exists() and json.loads(cycles_path.read_text()) != cycles:
        raise ValueError('The repair-cycle dataset changed')
    if not cycles_path.exists():
        write(cycles_path, cycles)
    # Rates are declared before this epoch sees any serving result. No SLO retuning.
    for model, rates in (('ouro', (1., .5)), ('huginn', (.2, .1))):
        if not qualified.get(model, {}).get('passed'):
            continue
        enabled = True
        for rate in rates:
            for index, seed in enumerate((2026106001, 2026107001)):
                count = 400 if model == 'ouro' and rate == 1. else 200
                estimate = 3 * (count / rate + 90)
                if time.time() + estimate + 900 >= GPU_STOP:
                    records.append(dict(model=model, rate=rate, seed=seed, status='NOT_MEASURED_TIME_BUDGET',
                                        estimated_point_seconds=estimate, estimate_is_not_measured_latency=True))
                    write(ROOT / 'configs/repair_epoch_serving_results.json', records)
                    continue
                if not enabled:
                    break
                arms = ['DP_Mixed', 'PD_G1', 'B']
                shift = (seed // 1000) % 3
                arms = arms[shift:] + arms[:shift]
                names = {}
                for position, arm in enumerate(arms):
                    name = f'repair_epoch_{model}_W1_{arm}_N2_rate{rate:g}_seed{seed}'
                    job(model, arm, count, rate, seed, name, 8. + position * .001,
                        False, gpus=(0, 1) if index == 0 else (2, 3), dataset='W1_repair_cycles')
                    names[arm] = name
                states = wait(list(names.values()))
                matched = parity(names['PD_G1'], names['B'])
                by_arm = dict(zip(names, states))
                paired_ready = by_arm['PD_G1'] == by_arm['B'] == 'COMPLETE'
                exact = (paired_ready and matched['N'] == count
                         and matched['exact'] == count and matched['complete_ids'])
                item = dict(model=model, rate=rate, seed=seed, count=count, names=names,
                            statuses=states, B_control_output=matched, outputs_exact=exact,
                            execution_epoch=EPOCH, accepted_original_model_lossless_speedup=False)
                records.append(item)
                if not exact:
                    enabled = False
                    suffix = '_repair_epoch_parity_failure.json' if paired_ready else '_repair_epoch_runtime_blocked.json'
                    write(ROOT / 'state' / (model + suffix), item)
                write(ROOT / 'configs/repair_epoch_serving_results.json', records)
    write(ROOT / 'state/repair_epoch_complete.json', dict(epoch=EPOCH, time=time.time(), results=records,
          scope='Separate matched-runtime epoch. Original failures, SLOs, and15-hour deadline remain unchanged.'))


if __name__ == '__main__':
    main(json.loads((ROOT / 'configs/b_repair_expansion.json').read_text()))
