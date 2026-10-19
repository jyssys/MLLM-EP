"""Audit hybrid raw records and extend the existing native control table."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

from poc_ouro_parallelism_deep_discovery.contract import MODELS, ROOT
from poc_ouro_parallelism_deep_discovery.scripts.analyze_native_tp_dp_cohorts import (
    Inputs, analyze as analyze_controls, availability, canonical, check_engine, require)
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import aggregate, validate_ownership


SOURCES = {
    '1.4B': 'outputs/native_tp2_dp2_cohort_base14_v4_preserve_environ',
    '2.6B': 'outputs/native_tp2_dp2_cohort_base26_v1_preserve_environ',
}
BOUNDARY = 'Observed parent first RELEASE send through both matching DONE receipts.'


def hybrid_trial(raw, size, case, length=128):
    require((raw['size'], raw['R'], raw['TP'], raw['DP'], raw['C'], raw['context']) ==
            (size, 4, 2, 2, case['C'], case['context']), 'Hybrid trial identity differs')
    require(raw['lane'] == 'CLEAN' and raw['quality'] == 'NOT_RUN', 'Hybrid lane or quality differs')
    require(raw['physical_gpus'] == [4, 5, 6, 7] and raw['physical_groups'] == [[4, 5], [6, 7]],
            'Hybrid physical groups differ')
    require(len(raw['workers']) == 2, 'Hybrid needs two worker records')
    results = {row['worker']: row for row in raw['workers']}
    require(len(results) == 2, 'Hybrid worker identities repeat')
    require(all(row['schedule_trace'] is None for row in results.values()), 'Trace entered clean timing')
    checked = aggregate(case, results, length, raw['release_start_ns'], raw['all_completed_ns'])
    for name in ('global_makespan_ns', 'generated_tokens', 'active_replicas', 'reserved_replicas',
                 'reserved_gpus', 'tensor_parallel_size_per_replica'):
        require(raw[name] == checked[name], f'Hybrid {name} differs')
    for name in ('global_makespan_ms', 'cohort_output_TPS'):
        require(math.isclose(raw[name], checked[name], rel_tol=1e-12), f'Hybrid {name} differs')
    longest = max(row['local_generate_wall_ns'] for row in results.values()) / 1e6
    residual = checked['global_makespan_ms'] - longest
    require(residual >= -1e-9, 'Hybrid charged residual is negative')
    return dict(wall_ms=checked['global_makespan_ms'], output_TPS=checked['cohort_output_TPS'],
        longest_active_local_generate_ms=longest, charged_IPC_controller_other_residual_ms=residual)


def analyze(root):
    root = Path(root)
    old = analyze_controls(root)
    inputs = Inputs(root)
    protocol_path = root / 'artifacts/native_tp2_dp2_cohort_protocol_v1.json'
    protocol = inputs.read(protocol_path)
    require(protocol['base_protocol_sha256'] == old['protocol_sha256'], 'Hybrid base hash differs')
    require(protocol['physical_groups'] == [[4, 5], [6, 7]], 'Hybrid protocol groups differ')
    summaries, trials, jobs = list(old['summaries']), list(old['trials']), list(old['jobs'])
    for size, source in SOURCES.items():
        directory = root / source
        state = availability(directory, inputs)
        jobs.append(dict(size=size, mode='TP2_DP2', source=source, **state))
        if state['status'] != 'AVAILABLE':
            for count in (1, 2, 4, 8):
                for context in (1024, 4096):
                    summaries.append(dict(size=size, mode='TP2_DP2', C=count, context=context,
                        status='NOT_AVAILABLE', repetitions=0, reserved_GPUs=4, active_replicas=None,
                        generated_tokens=count * 128, median_wall_ms=None, median_output_TPS=None,
                        observed_min_wall_ms=None, observed_max_wall_ms=None,
                        median_longest_active_local_generate_ms=None,
                        median_paired_charged_residual_ms=None, boundary=BOUNDARY))
            continue
        plan = inputs.read(directory / 'tp2_dp2_plan.json')
        require(plan['model']['revision'] == MODELS[size]['revision'], 'Hybrid model revision differs')
        require(plan['tp2_dp2_protocol'] == protocol, 'Embedded hybrid protocol differs')
        require(plan['native_protocol_sha256'] == old['protocol_sha256'], 'Embedded native hash differs')
        require(plan['tp2_dp2_protocol_sha256'] == inputs.hashes[str(protocol_path.relative_to(root))],
                'Embedded hybrid hash differs')
        signature = hashlib.sha256(canonical(plan['cases'])).hexdigest()
        require(signature == old['prompt_case_signatures'][size], 'Hybrid prompt IDs or case order differ')
        require(plan['fixed_token_id'] == old['fixed_token_ids'][size], 'Hybrid controlled token differs')
        require(plan['process_title_observability']['SPT_NOENV'] == '1', 'Title option is not recorded')
        require(len(plan['workers']) == 2, 'Hybrid plan needs two replicas')
        loaded = inputs.read(directory / 'loaded_workers.json')
        for worker in (0, 1):
            specification = plan['workers'][worker]
            worker_dir = directory / f'workers/replica{worker}_gpus{4+2*worker}_{5+2*worker}'
            require(inputs.read(worker_dir / 'status.json')['success'] is True, 'Hybrid worker did not stop successfully')
            check_engine(inputs.read(worker_dir / 'engine_config.json'), 2)
            ownership = inputs.read(worker_dir / 'ownership.json')
            validate_ownership(specification['physical_gpus'], MODELS[size]['layers'], ownership)
            require(ownership == loaded[str(worker)]['ownership'], 'Loaded rank ownership differs')
            leases = inputs.read(worker_dir / 'process_leases.json')
            require({row['pid'] for row in leases} == {loaded[str(worker)]['pid']} |
                    {row['pid'] for row in ownership}, 'Saved process leases omit a declared process')
            require(all(row['guard_output_matches'] and row['pgid'] == plan['parent_pgid']
                        and row['sid'] == plan['parent_sid']
                        and row['ancestry_to_frontend'][0] == row['pid']
                        and row['ancestry_to_frontend'][-1] == loaded[str(worker)]['pid'] for row in leases),
                    'Saved process leases differ')
        cases = {(case['C'], case['context']): case for case in plan['cases']}
        require(set(cases) == {(c, n) for c in (1, 2, 4, 8) for n in (1024, 4096)}, 'Hybrid cases differ')
        paths = sorted(directory.glob('C*_N*_repeat*.json'))
        require(len(paths) == 40, 'Hybrid repeated trial files are incomplete')
        seen, local = set(), []
        for path in paths:
            raw = inputs.read(path)
            key = raw['C'], raw['context'], raw['repeat']
            require(key not in seen and key[:2] in cases and key[2] in range(5), 'Hybrid trial key repeats or differs')
            seen.add(key)
            metric = hybrid_trial(raw, size, cases[key[:2]])
            local.append(dict(size=size, mode='TP2_DP2', C=key[0], context=key[1], repeat=key[2],
                              source=str(path.relative_to(root)), **metric))
        require(len(seen) == 40, 'Hybrid trial set is incomplete')
        trials.extend(local)
        for count, context in sorted(cases):
            rows = [row for row in local if (row['C'], row['context']) == (count, context)]
            median = lambda key: statistics.median(row[key] for row in rows)
            summaries.append(dict(size=size, mode='TP2_DP2', C=count, context=context, status='AVAILABLE',
                repetitions=len(rows), reserved_GPUs=4, active_replicas=min(count, 2), generated_tokens=count * 128,
                median_wall_ms=median('wall_ms'), median_output_TPS=median('output_TPS'),
                observed_min_wall_ms=min(row['wall_ms'] for row in rows),
                observed_max_wall_ms=max(row['wall_ms'] for row in rows),
                median_longest_active_local_generate_ms=median('longest_active_local_generate_ms'),
                median_paired_charged_residual_ms=median('charged_IPC_controller_other_residual_ms'), boundary=BOUNDARY))
    comparisons = []
    for size in MODELS:
        for count in (1, 2, 4, 8):
            for context in (1024, 4096):
                lookup = {row['mode']: row for row in summaries if
                          (row['size'], row['C'], row['context'], row['status']) == (size, count, context, 'AVAILABLE')}
                p4 = {name: lookup[name] for name in ('TP4', 'DP4', 'TP2_DP2') if name in lookup}
                best = min(p4, key=lambda name: p4[name]['median_wall_ms']) if len(p4) == 3 else None
                comparisons.append(dict(size=size, C=count, context=context,
                    TP4_API_ms=p4.get('TP4', {}).get('median_wall_ms'),
                    DP4_parent_ms=p4.get('DP4', {}).get('median_wall_ms'),
                    TP2_DP2_parent_ms=p4.get('TP2_DP2', {}).get('median_wall_ms'),
                    lowest_observed_four_GPU_median=best, timing_boundaries_match=False,
                    candidate_speedup='NOT_RUN', interpretation='Descriptive control medians. Different TP and parent boundaries.'))
    return dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), jobs=jobs, trials=trials,
        summaries=summaries, comparisons=comparisons, input_sha256={**old['input_sha256'], **inputs.hashes},
        uncertainty=old['uncertainty'], population=old['population'], quality='NOT_RUN', candidate_speedup='NOT_RUN',
        boundary_rule='TP uses native API wall. DP4 and TP2_DP2 use parent release-to-completion wall. Do not silently equate them.',
        time_rule='Separate jobs execute sequentially. Repeat indices do not create paired randomized architecture trials.',
        memory_rule='Every native rank reserves52GiB KV. TP4, DP4 and TP2_DP2 each reserve208GiB aggregate KV across four ranks. Replica ownership and weight multiplicity differ. TP1/TP2 controls use fewer physical ranks and aggregate reservations.',
        exclusions='Separate trace, warmups, protected CDB factorial, adaptive requests, natural quality, and held-out are excluded.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new output directory')
    result = analyze(args.root)
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    for key in ('summaries', 'trials', 'comparisons'):
        with (args.output / f'{key}.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(result[key])
    for size in MODELS:
        for context in (1024, 4096):
            series = []
            for mode in ('TP1', 'TP2', 'TP4', 'DP4', 'TP2_DP2'):
                points = [(row['C'], row['median_wall_ms']) for row in result['summaries'] if
                          (row['size'], row['mode'], row['context'], row['status']) == (size, mode, context, 'AVAILABLE')]
                if points:
                    series.append(dict(name=mode + (' parent' if mode in ('DP4', 'TP2_DP2') else ' API'), points=points))
            svg(args.output / f'{size}_N{context}_controls.svg', series,
                f'Base{size} fixedR4 controlled cohorts, N={context}', 'Submitted cohort C', 'Observed wall (ms)')
    print(json.dumps(dict(jobs_available=sum(row['status'] == 'AVAILABLE' for row in result['jobs']),
                          clean_trials=len(result['trials']), summary_cells=len(result['summaries']))))


if __name__ == '__main__':
    main()
