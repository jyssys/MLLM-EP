"""Analyze completed fixed-R4 controls. This module makes no CUDA call."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from poc_ouro_parallelism_deep_discovery.contract import MODELS, ROOT
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg


MODES = ('TP1', 'TP2', 'TP4', 'DP4')
BOUNDARIES = {
    'TP': 'One native generate API call, including cohort submission and completion.',
    'DP4': 'Observed parent first RELEASE send through all four matching DONE receipts.',
}
RESIDUAL = ('Parent makespan minus longest observed active local generate interval, per trial. '
            'Includes release skew, transport, controller work, and worker pre/post-call glue. Not pure IPC.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        self.hashes[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)


def availability(directory, inputs):
    statuses, reasons = {}, []
    for name in ('status.json', 'guard_status.json'):
        path = directory / name
        if not path.exists():
            reasons.append(f'{name} is absent')
            continue
        try:
            statuses[name] = inputs.read(path)
        except json.JSONDecodeError:
            reasons.append(f'{name} is not complete JSON')
            continue
        if statuses[name].get('success') is not True:
            reasons.append(f'{name} does not report success=true')
    guard = statuses.get('guard_status.json', {})
    if guard.get('success') is True and guard.get('exit_code') != 0:
        reasons.append('The guard exit code is not zero')
    return dict(status='NOT_AVAILABLE' if reasons else 'AVAILABLE', reasons=reasons,
                runner= statuses.get('status.json'), guard=guard or None)


def check_engine(config, tp):
    expected = dict(dtype='bfloat16', tensor_parallel_size=tp, async_scheduling=True,
                    enable_chunked_prefill=False, enable_prefix_caching=False,
                    enable_expert_parallel=False, enforce_eager=False,
                    kv_cache_memory_bytes=52 * 1024**3, max_num_seqs=8,
                    max_num_batched_tokens=32768)
    require(all(config.get(key) == value for key, value in expected.items()), 'Engine settings differ')
    require(config.get('hf_overrides', {}).get('total_ut_steps') == 4, 'The engine is not fixed R4')


def check_requests(records, count, length):
    require(len(records) == count, 'The request count differs')
    require(all(r['output_tokens'] == length and r['finish_reason'] == 'length' for r in records),
            'A request has the wrong output length or finish reason')


def trial_metrics(raw, mode, size, case, length):
    count, context = case['C'], case['context']
    require((raw['size'], raw['R'], raw['C'], raw['context']) == (size, 4, count, context),
            'Trial identity differs')
    require(raw['generated_tokens'] == count * length and raw['quality'] == 'NOT_RUN',
            'Trial denominator or quality marker differs')
    workers = []
    longest = residual = None
    if mode == 'DP4':
        require((raw['DP'], raw['TP'], raw['lane'], raw['reserved_replicas']) == (4, 1, 'CLEAN', 4),
                'The DP trial is not a clean DP4/TP1 trial')
        require(raw['physical_gpus'] == [4, 5, 6, 7], 'DP GPU ownership differs')
        duration = raw['all_completed_ns'] - raw['release_start_ns']
        require(duration > 0 and duration == raw['global_makespan_ns'], 'Parent timestamps differ')
        wall = duration / 1e6
        require(math.isclose(wall, raw['global_makespan_ms'], abs_tol=1e-9), 'Parent wall differs')
        require(len(raw['workers']) == 4, 'Four DP worker records are required')
        by_worker = {w['worker']: w for w in raw['workers']}
        require(set(by_worker) == set(range(4)), 'DP worker identities differ')
        for worker in range(4):
            value = by_worker[worker]
            slots = list(range(worker, count, 4))
            require(value['assigned_slots'] == slots and value['physical_gpu'] == 4 + worker,
                    'A DP slot or GPU assignment differs')
            require(value['generated_tokens'] == len(slots) * length, 'Worker denominator differs')
            require(value['output_identity'] == 'ALL_FROZEN_CONSTANT_TOKEN_IDS', 'Worker output identity differs')
            require(value['schedule_trace'] is None, 'Trace timing entered the clean lane')
            check_requests(value['request_records'], len(slots), length)
            require([r['cohort_slot'] for r in value['request_records']] == slots, 'Request slot order differs')
            expected_ids = [case['requests'][slot]['source_request_id'] for slot in slots]
            require([r['source_request_id'] for r in value['request_records']] == expected_ids,
                    'DP prompt source identity differs')
            local = value['local_generate_wall_ns']
            require(0 <= local <= duration and (local > 0) == bool(slots), 'Worker wall is invalid')
            require(raw['release_start_ns'] <= value['release_received_ns'] <= raw['all_completed_ns'],
                    'Worker release timestamp is outside the parent interval')
            workers.append(dict(worker=worker, physical_gpu=4 + worker, active=bool(slots),
                assigned_slots=json.dumps(slots), local_generate_wall_ms=local / 1e6,
                generated_tokens=value['generated_tokens']))
        require(raw['active_replicas'] == min(count, 4), 'Active replica count differs')
        longest = max(w['local_generate_wall_ms'] for w in workers if w['active'])
        residual = wall - longest
        require(residual >= -1e-9, 'The charged residual is negative')
    else:
        require(raw['tp'] == int(mode[2:]), 'TP degree differs')
        require(raw['output_identity'] == 'ALL_FROZEN_CONSTANT_TOKEN_IDS', 'TP output identity differs')
        check_requests(raw['request_records'], count, length)
        wall = raw['wall_ms']
    require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0, 'Wall time is invalid')
    tps = count * length * 1000 / wall
    require(math.isclose(tps, raw['cohort_output_TPS'], rel_tol=1e-12), 'TPS denominator differs')
    return dict(wall_ms=wall, output_TPS=tps, longest_active_local_generate_ms=longest,
                charged_IPC_controller_other_residual_ms=residual), workers


def analyze(root):
    root = Path(root)
    inputs = Inputs(root)
    base_path = root / 'artifacts/native_cohort_protocol_v2.json'
    dp_path = root / 'artifacts/native_dp4_cohort_protocol_v1.json'
    protocol, dp_protocol = inputs.read(base_path), inputs.read(dp_path)
    base_sha, dp_sha = inputs.hashes[str(base_path.relative_to(root))], inputs.hashes[str(dp_path.relative_to(root))]
    require(dp_protocol['base_protocol_sha256'] == base_sha, 'The DP protocol base hash differs')
    require((protocol['cohort_sizes'], protocol['contexts'], protocol['generated_tokens'], protocol['repeats']) ==
            ([1, 2, 4, 8], [1024, 4096], 128, 5), 'The frozen cohort shape differs')
    require(protocol['depth_mode'] == 'FIXED_R4_ONLY', 'The protocol is not fixed R4')
    expected_cells = {(c, n) for c in protocol['cohort_sizes'] for n in protocol['contexts']}
    jobs, trials, workers, summaries = [], [], [], []
    reference_cases, reference_tokens = {}, {}
    for size in MODELS:
        short = size.replace('.', '').replace('B', '')
        for mode in MODES:
            relative = (f'outputs/native_dp4_cohort_base{short}_v1' if mode == 'DP4' else
                        f'outputs/native_cohort_base{short}_P{mode[2:]}_v1')
            directory = root / relative
            state = availability(directory, inputs)
            jobs.append(dict(size=size, mode=mode, source=relative, **state))
            if state['status'] != 'AVAILABLE':
                continue
            frozen = inputs.read(directory / ('dp4_plan.json' if mode == 'DP4' else 'protocol.json'))
            require(frozen['model']['revision'] == MODELS[size]['revision'], 'Checkpoint revision differs')
            embedded = frozen['native_protocol'] if mode == 'DP4' else frozen['protocol']
            require(embedded == protocol, 'The embedded native protocol differs')
            require(frozen['native_protocol_sha256' if mode == 'DP4' else 'protocol_sha256'] == base_sha,
                    'The native protocol hash differs')
            if mode == 'DP4':
                require(frozen['dp4_protocol'] == dp_protocol and frozen['dp4_protocol_sha256'] == dp_sha,
                        'The frozen DP protocol differs')
                for worker in range(4):
                    worker_dir = directory / f'workers/replica{worker}_gpu{worker+4}'
                    require(inputs.read(worker_dir / 'status.json')['success'] is True, 'A worker did not stop successfully')
                    check_engine(inputs.read(worker_dir / 'engine_config.json'), 1)
            else:
                check_engine(inputs.read(directory / 'engine_config.json'), int(mode[2:]))
            cases = {(case['C'], case['context']): case for case in frozen['cases']}
            require(set(cases) == expected_cells and len(frozen['cases']) == len(expected_cells), 'Case identities differ')
            for (count, context), case in cases.items():
                require([r['cohort_slot'] for r in case['requests']] == list(range(count)), 'Frozen slots differ')
                require(all(len(r['prompt_token_ids']) == context for r in case['requests']), 'Frozen prompt length differs')
            signature = hashlib.sha256(canonical(frozen['cases'])).hexdigest()
            require(reference_cases.setdefault(size, signature) == signature, 'Prompt tokens or population ordering differ')
            require(reference_tokens.setdefault(size, frozen['fixed_token_id']) == frozen['fixed_token_id'], 'Constant token differs')
            paths = sorted(directory.glob('C*_N*_repeat*.json'))
            require(len(paths) == len(expected_cells) * protocol['repeats'], 'Measured repetition files are missing or extra')
            seen = set()
            for path in paths:
                raw = inputs.read(path)
                key = (raw['C'], raw['context'], raw['repeat'])
                require(key not in seen and key[:2] in expected_cells and key[2] in range(5), 'Repeated or invalid trial key')
                seen.add(key)
                metric, local = trial_metrics(raw, mode, size, cases[key[:2]], 128)
                identity = dict(size=size, mode=mode, C=key[0], context=key[1], repeat=key[2])
                trials.append(dict(**identity, source=str(path.relative_to(root)), **metric))
                workers.extend(dict(**identity, **row) for row in local)
            require(len(seen) == 40, 'Measured trials are incomplete')
    for size in MODELS:
        for mode in MODES:
            for count, context in sorted(expected_cells):
                selected = [r for r in trials if (r['size'], r['mode'], r['C'], r['context']) == (size, mode, count, context)]
                require(len(selected) in (0, 5), 'A cell contains a partial trial set')
                available = bool(selected)
                med = lambda key: statistics.median(r[key] for r in selected) if selected and selected[0][key] is not None else None
                summaries.append(dict(size=size, mode=mode, C=count, context=context,
                    status='AVAILABLE' if available else 'NOT_AVAILABLE', repetitions=len(selected),
                    reserved_GPUs=4 if mode == 'DP4' else int(mode[2:]),
                    active_replicas=min(count, 4) if mode == 'DP4' and available else None,
                    generated_tokens=count*128, median_wall_ms=med('wall_ms'), median_output_TPS=med('output_TPS'),
                    observed_min_wall_ms=min(r['wall_ms'] for r in selected) if available else None,
                    observed_max_wall_ms=max(r['wall_ms'] for r in selected) if available else None,
                    median_longest_active_local_generate_ms=med('longest_active_local_generate_ms'),
                    median_paired_charged_residual_ms=med('charged_IPC_controller_other_residual_ms'),
                    boundary=BOUNDARIES['DP4' if mode == 'DP4' else 'TP']))
    worker_summaries = []
    for size in MODELS:
        for count, context in sorted(expected_cells):
            for worker in range(4):
                selected = [r for r in workers if (r['size'],r['C'],r['context'],r['worker']) == (size,count,context,worker)]
                if selected:
                    worker_summaries.append(dict(size=size,C=count,context=context,worker=worker,
                        physical_gpu=worker+4,active=selected[0]['active'],repetitions=len(selected),
                        assigned_slots=selected[0]['assigned_slots'],
                        median_local_generate_wall_ms=statistics.median(r['local_generate_wall_ms'] for r in selected)))
    comparisons = []
    for size in MODELS:
        for count, context in sorted(expected_cells):
            lookup = {r['mode']:r for r in summaries if (r['size'],r['C'],r['context']) == (size,count,context)}
            tp, dp = lookup['TP4'], lookup['DP4']
            available = tp['status'] == dp['status'] == 'AVAILABLE'
            comparisons.append(dict(size=size,C=count,context=context,status='AVAILABLE' if available else 'NOT_AVAILABLE',
                TP4_API_median_ms=tp['median_wall_ms'],DP4_parent_median_ms=dp['median_wall_ms'],
                TP4_API_over_DP4_parent_wall_ratio=tp['median_wall_ms']/dp['median_wall_ms'] if available else None,
                boundaries_match=False,candidate_gain='NOT_RUN'))
    return dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=jobs, summaries=summaries,
        trials=trials,dp_workers=workers,dp_worker_summaries=worker_summaries,comparisons=comparisons,
        input_sha256=inputs.hashes,protocol_sha256=base_sha,dp4_protocol_sha256=dp_sha,
        prompt_case_signatures=reference_cases,fixed_token_ids=reference_tokens,boundaries=BOUNDARIES,
        charged_residual_definition=RESIDUAL,
        uncertainty='Five repeated controlled cohorts are calibration repeats, not independent natural populations. No population confidence interval.',
        population='C1 uses the first frozen pattern. C2/C4/C8 alternate two repeated patterns. C is not a constant native batch size.',
        exclusions='Warmups and separate trace timing excluded. No candidate gain, CDB factorial, held-out, quality, or steady-state claim.')


def csv_file(path, rows):
    with path.open('x', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new output directory')
    value = analyze(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    for name, key in [('summary','summaries'),('trials','trials'),('dp_workers','dp_workers'),
                      ('dp_worker_summary','dp_worker_summaries'),('matched_comparisons','comparisons')]:
        csv_file(args.output / f'{name}.csv', value[key])
    for size in MODELS:
        for context in (1024,4096):
            for field,label in [('median_wall_ms','Closed-cohort wall (ms)'),('median_output_TPS','Controlled output tokens/s')]:
                series=[]
                for mode in MODES:
                    points=[(r['C'],r[field]) for r in value['summaries'] if
                            (r['size'],r['mode'],r['context'],r['status']) == (size,mode,context,'AVAILABLE')]
                    if points:
                        series.append(dict(name=mode+(' parent wall' if mode=='DP4' else ' API wall'),points=points))
                svg(args.output/f'{size}_N{context}_{field}.svg',series,
                    f'Base {size}, fixed R4 control, N={context}','Submitted cohort C',label)
    print(json.dumps(dict(summary_cells=len(value['summaries']),clean_trials=len(value['trials']),
        completed_jobs=sum(j['status']=='AVAILABLE' for j in value['jobs']),
        unavailable_jobs=[j['source'] for j in value['jobs'] if j['status']!='AVAILABLE'])))


if __name__ == '__main__':
    main()
