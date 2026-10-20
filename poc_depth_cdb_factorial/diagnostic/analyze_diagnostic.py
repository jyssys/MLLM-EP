"""Read archived A1/A2 evidence. Never modify the original experiment."""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'diagnostic' / 'results'
INPUTS = {}
SEED = 20261023
BUCKETS = [1, 2, 4, 8, 12, 16]


def read(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + '.gz')
    INPUTS[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt') as f:
        return json.load(f)


def write(name, data):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvout(name, rows):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    op = gzip.open if target.suffix == '.gz' else open
    with op(target, 'wt', newline='') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})


def cv(a):
    a = np.asarray(a, dtype=float)
    return float(a.std() / a.mean()) if a.size and a.mean() else 0.


def dist(a):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return None
    return dict(N=int(a.size), mean=float(a.mean()), variance=float(a.var()),
                std=float(a.std()), CV=cv(a), min=float(a.min()), max=float(a.max()),
                **{f'P{p}': float(np.percentile(a, p)) for p in [10, 50, 90]})


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return dict(Pearson=float(stats.pearsonr(a, b).statistic),
                Spearman=float(stats.spearmanr(a, b).statistic))


def dataset(r):
    for value in ['gsm8k', 'math500', 'alpaca', 'sharegpt']:
        if value in r['id']:
            return value
    raise ValueError(r['id'])


def depths(r):
    return np.array([e['depth'] for e in r['depths']], dtype=np.int16)


def req_summary(r, cohort):
    d = depths(r)
    assert len(d) == r['generated_tokens'] - 1
    assert set(d.tolist()) <= {2, 3, 4}
    return dict(cohort=cohort, id=r['id'], dataset=dataset(r), GPU=r['gpu'],
                generated_tokens=r['generated_tokens'], decode_tokens=len(d),
                prompt_tokens=r['prompt_tokens'], mean_depth=float(d.mean()),
                depth_variance=float(d.var()), total_loop_work=int(d.sum()),
                R2=int((d == 2).sum()), R3=int((d == 3).sum()), R4=int((d == 4).sum()),
                finish_reason=r['finish_reason'])


def cohort_summary(records, name):
    out = []
    for domain in ['all', 'gsm8k', 'math500', 'alpaca', 'sharegpt']:
        selected = [r for r in records if domain == 'all' or dataset(r) == domain]
        if not selected:
            continue
        ds = np.concatenate([depths(r) for r in selected])
        lengths = [len(depths(r)) for r in selected]
        means = [float(depths(r).mean()) for r in selected]
        work = [int(depths(r).sum()) for r in selected]
        lw, ll, lm = map(np.log, [work, lengths, means])
        # Exact identity, not a causal variance attribution.
        decomposition = dict(log_work_variance=float(lw.var()), log_length_variance=float(ll.var()),
                             log_mean_depth_variance=float(lm.var()),
                             twice_covariance=float(2 * np.mean((ll - ll.mean()) * (lm - lm.mean()))))
        assert np.isclose(decomposition['log_work_variance'], sum(list(decomposition.values())[1:]))
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(work), (2000, len(work)))
        # Resample requests. Do not resample their tokens independently.
        m = np.asarray(work)[indices].sum(axis=1) / np.asarray(lengths)[indices].sum(axis=1)
        out.append(dict(evidence='LIVE_MEASURED', cohort=name, dataset=domain, requests=len(selected),
                        generated_tokens=sum(r['generated_tokens'] for r in selected),
                        recorded_decode_tokens=len(ds), token_depth=dist(ds),
                        token_depth_mean_request_bootstrap_CI95=np.quantile(m, [.025, .975]).tolist(),
                        R2_fraction=float((ds == 2).mean()), R3_fraction=float((ds == 3).mean()),
                        R4_fraction=float((ds == 4).mean()),
                        request_output_length=dist([r['generated_tokens'] for r in selected]),
                        request_mean_depth=dist(means), request_total_work=dist(work),
                        length_vs_work=corr(lengths, work), mean_depth_vs_work=corr(means, work),
                        log_variance_identity=decomposition))
    return out


def case(C, arm, repeat=0, trace=False):
    directory = ROOT / 'outputs' / ('trace_DP4_v1' if trace else f'speed_DP4_r{repeat}_v1')
    name = f'{"trace_" if trace else ""}speed_C{C}_{arm}_r{repeat}'
    parent = read(directory / (name + '.json'))
    ranks = [read(directory / f'rank_{rank}' / (name + '.json')) for rank in range(4)]
    records = [r for raw in ranks for r in raw['outputs']]
    assert len(records) == C
    mapping = {a['id']: a['rank'] for a in parent['assignments']}
    assert all(mapping[r['id']] == r['gpu'] - 4 for r in records)
    return parent, ranks, records


def parity(left, right, label):
    a, b = ({r['id']: r for r in records} for records in (left, right))
    assert a.keys() == b.keys()
    rows = []
    for rid in sorted(a):
        x, y = a[rid], b[rid]
        dx, dy = depths(x).tolist(), depths(y).tolist()
        first = next((i for i, (u, v) in enumerate(zip(x['output_ids'], y['output_ids'])) if u != v),
                     min(x['generated_tokens'], y['generated_tokens']))
        rows.append(dict(label=label, id=rid, token_exact=x['output_ids'] == y['output_ids'],
                         depth_exact=dx == dy, length_exact=x['generated_tokens'] == y['generated_tokens'],
                         work_exact=sum(dx) == sum(dy), reason_exact=x['finish_reason'] == y['finish_reason'],
                         left_tokens=x['generated_tokens'], right_tokens=y['generated_tokens'],
                         left_work=sum(dx), right_work=sum(dy),
                         first_token_difference_0based=first if x['output_ids'] != y['output_ids'] else None))
    summary = dict(label=label, evidence='LIVE_MEASURED', N=len(rows),
                   **{k: sum(r[k] for r in rows) for k in
                      ['token_exact', 'depth_exact', 'length_exact', 'work_exact', 'reason_exact']},
                   left_tokens=sum(r['generated_tokens'] for r in left),
                   right_tokens=sum(r['generated_tokens'] for r in right),
                   left_work=sum(int(depths(r).sum()) for r in left),
                   right_work=sum(int(depths(r).sum()) for r in right))
    return summary, rows


def lpt(ids, weights, capacity=16):
    loads, counts = [0.] * 4, [0] * 4
    placement = {}
    for rid in sorted(ids, key=lambda rid: -weights[rid]):
        rank = min((i for i in range(4) if counts[i] < capacity),
                   key=lambda i: (loads[i], counts[i], i))
        placement[rid] = rank
        loads[rank] += weights[rid]
        counts[rank] += 1
    return placement


def rank_wave(records):
    """One no-refill cohort, one fresh token per request per wave."""
    n = max((len(depths(r)) for r in records), default=0)
    exit_depths = np.zeros((len(records), n), dtype=np.int16)
    for i, r in enumerate(records):
        exit_depths[i, :len(depths(r))] = depths(r)
    active = np.array([(exit_depths >= step).sum(axis=0) for step in range(1, 5)])
    return active


def wave_features(active):
    b = active[active > 0].astype(float)
    # Native graph buckets are read from the archived trace, then checked below.
    padded = np.array([next(p for p in BUCKETS if p >= v) for v in b])
    return np.array([len(b), b.sum(), padded.sum()], dtype=float)


def replay(records, placement, coefficients=None):
    rank_results = []
    for rank in range(4):
        local = [r for r in records if placement[r['id']] == rank]
        active = rank_wave(local)
        f = wave_features(active)
        rank_results.append(dict(rank=rank, requests=len(local), work=int(f[1]),
                                 core_calls=int(f[0]), padded_row_work=int(f[2]),
                                 predicted_core_ms=float(f @ coefficients) if coefficients is not None else None,
                                 by_depth_calls=[int((x > 0).sum()) for x in active],
                                 max_decode_waves=active.shape[1]))
    work = [r['work'] for r in rank_results]
    calls = [r['core_calls'] for r in rank_results]
    return dict(ranks=rank_results, work_CV=cv(work), max_work=max(work),
                max_core_calls=max(calls), mean_core_calls=float(np.mean(calls)),
                predicted_core_critical_ms=max(r['predicted_core_ms'] for r in rank_results)
                if coefficients is not None else None)


def paired_ratio(a, b):
    a, b = np.asarray(a), np.asarray(b)
    indices = np.random.default_rng(20261019).integers(0, len(a), (2000, len(a)))
    return dict(speedup=float(np.median(a) / np.median(b)),
                CI95=np.quantile(np.median(a[indices], axis=1) / np.median(b[indices], axis=1), [.025, .975]).tolist(),
                bootstrap_unit='Paired whole-cohort repeat', N=len(a))


def trace_rows(C, arm, raw):
    lookup = {(r['id'], e['position']): e['depth'] for r in raw['outputs'] for e in r['depths']}
    rows = []
    previous_depth = None
    for index, s in enumerate(raw['stages']):
        row = dict(C=C, arm=arm, GPU=raw['outputs'][0]['gpu'], index=index,
                   stage=s['stage'], batch=s['batch'], gpu_ms=s['gpu_ms'],
                   gpu_start_s=s['gpu_start_s'], gpu_end_s=s['gpu_end_s'],
                   sampled_resident=s['occupancy'].get('resident'),
                   sampled_waiting=s['occupancy'].get('waiting'),
                   evidence='LIVE_MEASURED', scope='Trace stream bracket, not exclusive kernel service')
        if s['stage'] == 'compute_recurrent_batch':
            ds = [v['depth'] for v in s['items']]
            assert len(set(ds)) == 1
            previous_depth = ds[0]
            final = [lookup[(v['id'], v['position'])] for v in s['items']]
            row.update(current_step=previous_depth, current_step_variance=0.,
                       final_exit_depth_mean=float(np.mean(final)), final_exit_depth_variance=float(np.var(final)),
                       padded_rows=s['occupancy']['core_padded_rows'],
                       IDs=[v['id'] for v in s['items']], positions=[v['position'] for v in s['items']])
        elif s['stage'] == 'exit_KV_copy':
            row['after_step'] = previous_depth
        rows.append(row)
    return rows


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    speed, rank_rows, balance_rows, parity_rows, parity_summaries = {}, [], [], [], []
    request_rows, stats_rows, time_summary = [], [], []
    manifest = read(ROOT / 'artifacts/arm_manifest.json')
    historical = read(ROOT / 'artifacts/speed_reference_depth_work.json')
    for C in [8, 16, 32, 64]:
        times = {arm: [] for arm in ['A1', 'A2']}
        for repeat in range(5):
            for arm in ['A1', 'A2']:
                parent, ranks, records = case(C, arm, repeat)
                speed[C, arm, repeat] = parent, ranks, records
                times[arm].append(max(r['finish_s'] for r in records) - parent['first_submitted_s'])
                for rank, raw in enumerate(ranks):
                    rec = raw['outputs']
                    ds = np.concatenate([depths(r) for r in rec])
                    work = [int(depths(r).sum()) for r in rec]
                    hist = {int(k.rsplit(':', 1)[1]): v for k, v in raw['batch_histogram'].items()
                            if k.startswith('compute_recurrent_batch:')}
                    assert sum(k * v for k, v in hist.items()) == sum(work)
                    assert raw['config'] == ranks[0]['config']
                    rank_rows.append(dict(C=C, arm=arm, repeat=repeat, GPU=4 + rank, rank=rank,
                        evidence='LIVE_MEASURED', requests=len(rec), IDs=[r['id'] for r in rec],
                        dataset_counts=dict(collections.Counter(dataset(r) for r in rec)),
                        output_tokens=sum(r['generated_tokens'] for r in rec), decode_tokens=len(ds),
                        prompt_tokens=sum(r['prompt_tokens'] for r in rec),
                        prefill_loop_row_work=raw['stats']['prefill_recurrent_steps'],
                        depth_mean=float(ds.mean()), depth_variance=float(ds.var()),
                        R2_fraction=float((ds == 2).mean()), R3_fraction=float((ds == 3).mean()),
                        R4_fraction=float((ds == 4).mean()), total_loop_work=sum(work),
                        historical_work=sum(historical[r['id']] for r in rec),
                        request_work_mean=float(np.mean(work)), request_work_variance=float(np.var(work)),
                        rank_finish_s=max(r['finish_s'] for r in rec) - parent['first_submitted_s'],
                        host_decode_span_s=max(r['finish_s'] for r in rec) - min(r['first_token_s'] for r in rec),
                        host_TTFT_max_s=max(r['first_token_s'] for r in rec) - parent['first_submitted_s'],
                        host_queue_mean_ms=1000 * np.mean([r['first_scheduled_s'] - parent['first_submitted_s'] for r in rec]),
                        core_calls=sum(hist.values()), core_row_mean=sum(work) / sum(hist.values()),
                        prefill_batches=raw['stats']['prefill_batches'],
                        scheduler_refills=raw['stats']['scheduler_refills'],
                        graph_captures=raw['stats']['recurrent_graph_captures'],
                        peak_HBM_GiB=raw['peak_memory'] / 2**30,
                        exclusive_clean_GPU_decode_time='NOT_AVAILABLE',
                        whole_GPU_busy_idle='NOT_AVAILABLE'))
            x, y = speed[C, 'A1', repeat], speed[C, 'A2', repeat]
            assert all(a['config'] == b['config'] for a, b in zip(x[1], y[1]))
            summary, detail = parity(x[2], y[2], f'speed_C{C}_r{repeat}')
            parity_summaries.append(summary)
            parity_rows.extend(detail)
        time_summary.append(dict(C=C, evidence='LIVE_MEASURED', A1_makespans_s=times['A1'],
            A2_makespans_s=times['A2'], A1_median_s=float(np.median(times['A1'])),
            A2_median_s=float(np.median(times['A2'])), **paired_ratio(times['A1'], times['A2'])))
        for arm in ['A1', 'A2']:
            for repeat in range(5):
                r = [row for row in rank_rows if row['C'] == C and row['arm'] == arm and row['repeat'] == repeat]
                finish = [v['rank_finish_s'] for v in r]
                work = [v['total_loop_work'] for v in r]
                histwork = [v['historical_work'] for v in r]
                balance_rows.append(dict(C=C, arm=arm, repeat=repeat, evidence='LIVE_MEASURED',
                    work_CV=cv(work), work_max_mean=max(work) / np.mean(work), max_work=max(work),
                    historical_work_CV=cv(histwork), historical_work_max=max(histwork),
                    request_count_CV=cv([v['requests'] for v in r]),
                    output_tokens_CV=cv([v['output_tokens'] for v in r]),
                    rank_finish_CV=cv(finish), rank_finish_max_mean=max(finish) / np.mean(finish),
                    finish_skew_ms=1000 * (max(finish) - min(finish)),
                    no_local_request_tail_ms=[1000 * (max(finish) - f) for f in finish],
                    max_core_calls=max(v['core_calls'] for v in r),
                    core_calls_CV=cv([v['core_calls'] for v in r]),
                    host_decode_span_CV=cv([v['host_decode_span_s'] for v in r])))

    # Different cohorts remain separate. Repeats are not new independent requests.
    quality = {}
    for arm in ['A1', 'A2']:
        values = []
        for part in range(4):
            for rank in range(4):
                raw = read(ROOT / f'outputs/quality_DP4_v1/rank_{rank}/quality_{arm}_part{part}.json')
                values.extend(raw['outputs'])
        assert len({r['id'] for r in values}) == 256
        quality[arm] = values
        cohorts = [(f'speed_C64_{arm}_r0', speed[64, arm, 0][2]), (f'quality_{arm}', values)]
        for name, records in cohorts:
            request_rows.extend(req_summary(r, name) for r in records)
            stats_rows.extend(cohort_summary(records, name))
    qsum, qrows = parity(quality['A1'], quality['A2'], 'quality_256_A1_A2')
    parity_summaries.append(qsum)
    parity_rows.extend(qrows)
    # Include previous v3 requests as their own trace cohort, not extra factorial samples.
    old = read(ROOT.parent / 'poc_decode_first_depth_aware_dp_v3/artifacts/request_depth_summary.json')
    oldfile = ROOT.parent / 'poc_decode_first_depth_aware_dp_v3/artifacts/decode_depth_traces.jsonl.gz'
    INPUTS[str(oldfile.relative_to(ROOT.parent))] = hashlib.sha256(oldfile.read_bytes()).hexdigest()
    byid = collections.defaultdict(list)
    with gzip.open(oldfile, 'rt') as f:
        for line in f:
            e = json.loads(line)
            byid[e['id']].append(e)
    old_records = []
    for r in old:
        es = byid[r['id']]
        assert sum(e['depth'] for e in es) == r['total_loop_work']
        old_records.append(dict(id=r['id'], generated_tokens=r['output_tokens'], depths=es))
    stats_rows.extend(cohort_summary(old_records, 'previous_v3_128'))
    same_length = [r for r in speed[64, 'A1', 0][2] if r['generated_tokens'] == 256]
    stats_rows.extend(cohort_summary(same_length, 'speed_A1_exact_256_output_tokens'))

    trace_events, trace_summaries, observer, copy_groups = [], [], [], []
    for C in [8, 16, 32, 64]:
        for arm in ['A1', 'A2']:
            parent, ranks, records = case(C, arm, trace=True)
            clean = speed[C, arm, 0]
            op, _ = parity(clean[2], records, f'trace_C{C}_{arm}_vs_clean_r0')
            op['trace_clean_wall_ratio'] = ((max(r['finish_s'] for r in records) - parent['first_submitted_s']) /
                (max(r['finish_s'] for r in clean[2]) - clean[0]['first_submitted_s']))
            observer.append(op)
            for rank, raw in enumerate(ranks):
                rows = trace_rows(C, arm, raw)
                trace_events.extend(rows)
                core = [s for s in rows if s['stage'] == 'compute_recurrent_batch']
                active = rank_wave(raw['outputs'])
                assert active.sum() == sum(s['batch'] for s in core)
                # Wave identity checks the actual no-refill schedule, not a made-up simulator.
                assert [(active[i] > 0).sum() for i in range(4)] == [sum(s['current_step'] == i for s in core) for i in range(1, 5)]
                assert all(s['padded_rows'] == next(p for p in BUCKETS if p >= s['batch']) for s in core)
                bystage = {s: sum(r['gpu_ms'] for r in rows if r['stage'] == s) for s in sorted({r['stage'] for r in rows})}
                copies = [s for s in rows if s['stage'] == 'exit_KV_copy']
                for step in [2, 3]:
                    cs = [s for s in copies if s.get('after_step') == step]
                    copy_groups.append(dict(C=C, arm=arm, GPU=rank + 4, after_step=step, flushes=len(cs),
                        exited_rows=sum(s['batch'] for s in cs), stream_bracket_ms=sum(s['gpu_ms'] for s in cs)))
                batch = [s['batch'] for s in core]
                trace_summaries.append(dict(C=C, arm=arm, GPU=rank + 4, evidence='LIVE_MEASURED',
                    core_calls=len(core), real_rows=sum(batch), padded_rows=sum(s['padded_rows'] for s in core),
                    graph_padding_amp=sum(s['padded_rows'] for s in core) / sum(batch),
                    active_batch=dist(batch), by_depth_calls=[sum(s['current_step'] == i for s in core) for i in range(1, 5)],
                    by_depth_batch_mean=[float(np.mean([s['batch'] for s in core if s['current_step'] == i])) for i in range(1, 5)],
                    final_exit_variance_batch_mean=float(np.mean([s['final_exit_depth_variance'] for s in core])),
                    measured_ms=bystage, exit_copy_flushes=len(copies),
                    stream_ms_per_row=sum(s['gpu_ms'] for s in core) / sum(batch),
                    anchor_uncertainty_ms=raw.get('GPU_anchor_uncertainty_ms'),
                    queue_samples=len(raw['queue_timeline']),
                    max_sampled_active=max((s.get('active', 0) for s in raw['queue_timeline']), default=0),
                    max_sampled_waiting=max((s.get('waiting', 0) for s in raw['queue_timeline']), default=0),
                    kernel_count='NOT_AVAILABLE. One core call executes many kernels or a graph.',
                    whole_GPU_idle='NOT_AVAILABLE'))

    # Descriptive cost calibration: leave-one-configuration-out on GPU bracket time.
    # It predicts recurrent-core brackets ONLY. It does not predict clean E2E.
    f = np.array([[r['core_calls'], r['real_rows'], r['padded_rows']] for r in trace_summaries], float)
    y = np.array([r['measured_ms']['compute_recurrent_batch'] for r in trace_summaries])
    scales = f.max(axis=0)
    coef = nnls(f / scales, y)[0] / scales
    validation = []
    for C in [8, 16, 32, 64]:
        test = np.array([r['C'] == C for r in trace_summaries])
        cf = nnls(f[~test] / scales, y[~test])[0] / scales
        pred = f[test] @ cf
        validation.append(dict(held_out_C=C, MAE_ms=float(np.mean(abs(pred - y[test]))),
                               MAPE_pct=float(np.mean(abs(pred / y[test] - 1)) * 100)))
    model = dict(evidence='SIMULATED', coefficients_ms=coef.tolist(), features=['core_calls', 'real_rows', 'padded_rows'],
                 calibration='32 rank brackets from archived trace lane, non-negative least squares',
                 validation=validation, full_E2E='NOT_AVAILABLE',
                 limitation='Nested request sets. Not independent request-group CV. Single trace per configuration. No host/copy/prefill cost.')

    ids = manifest['speed_ids']
    frozen_by_id = {r['id']: r for r in speed[64, 'A1', 0][2]}
    frozen = [frozen_by_id[rid] for rid in ids]
    weights = {r['id']: int(depths(r).sum()) for r in frozen}
    length = {r['id']: len(depths(r)) for r in frozen}
    depth = {r['id']: float(depths(r).mean()) for r in frozen}
    mappings = {
        'Native_RR': {rid: i % 4 for i, rid in enumerate(ids)},
        'Legacy_historical_LPT': lpt(ids, historical),
        'Length_only_LPT': lpt(ids, length),
        'Depth_only_normalized_length_LPT': lpt(ids, depth),
        'Depth_plus_length_LPT': lpt(ids, weights),
    }
    actualmap = {a['id']: a['rank'] for a in speed[64, 'A2', 0][0]['assignments']}
    assert mappings['Legacy_historical_LPT'] == actualmap
    # Bounded hindsight search, not an exact oracle. Hold 16 requests per rank.
    best = mappings['Depth_plus_length_LPT'].copy()
    best_cost = replay(frozen, best, coef)['predicted_core_critical_ms']
    rng = np.random.default_rng(SEED)
    accepted = 0
    for _ in range(1500):
        a, b = rng.choice(ids, 2, replace=False)
        if best[a] == best[b]:
            continue
        candidate = best.copy()
        candidate[a], candidate[b] = candidate[b], candidate[a]
        cost = replay(frozen, candidate, coef)['predicted_core_critical_ms']
        if cost < best_cost - 1e-9:
            best, best_cost = candidate, cost
            accepted += 1
    mappings['Bracket_cost_swap_heuristic'] = best
    replay_rows = []
    base = replay(frozen, mappings['Native_RR'], coef)
    for policy, mapping in mappings.items():
        result = replay(frozen, mapping, coef)
        replay_rows.append(dict(policy=policy, evidence='TRACE_REPLAY',
            max_work=result['max_work'], work_CV=result['work_CV'], max_core_calls=result['max_core_calls'],
            additive_work_speedup=base['max_work'] / result['max_work'],
            unit_core_call_speedup=base['max_core_calls'] / result['max_core_calls'],
            SIMULATED_core_ms=result['predicted_core_critical_ms'],
            SIMULATED_core_speedup=base['predicted_core_critical_ms'] / result['predicted_core_critical_ms'],
            LIVE_E2E='NOT_AVAILABLE', placement=mapping, ranks=result['ranks']))

    # Mechanism cohorts: use real recorded positions only. No gate change.
    # The equal-length diagnostic uses first 128 positions from sufficiently long quality requests.
    pool = []
    for r in quality['A1']:
        if len(depths(r)) >= 128:
            x = dict(r, depths=r['depths'][:128], generated_tokens=129)
            pool.append(x)
    pool.sort(key=lambda r: (float(depths(r).mean()), r['id']))
    assert len(pool) >= 32
    highvar = pool[:16] + pool[-16:]
    mechanism = []
    for name, records in [('Natural_C64', frozen), ('High_depth_variance_equal_128_positions', highvar)]:
        rids = [r['id'] for r in records]
        ws = {r['id']: int(depths(r).sum()) for r in records}
        cap = len(records) // 4
        sorted_ids = sorted(rids, key=lambda rid: -ws[rid])
        policies = dict(Natural_RR={rid: i % 4 for i, rid in enumerate(rids)},
                        Adversarial={rid: i // cap for i, rid in enumerate(sorted_ids)},
                        Balanced_LPT=lpt(rids, ws, capacity=cap))
        baseline = replay(records, policies['Natural_RR'], coef)
        for policy, mapping in policies.items():
            result = replay(records, mapping, coef)
            mechanism.append(dict(cohort=name, policy=policy, evidence='TRACE_REPLAY',
                N=len(records), recorded_decode_tokens=sum(len(depths(r)) for r in records),
                request_depth_mean=dist([float(depths(r).mean()) for r in records]),
                max_work=result['max_work'], work_CV=result['work_CV'], max_core_calls=result['max_core_calls'],
                additive_work_speedup=baseline['max_work'] / result['max_work'],
                unit_core_call_speedup=baseline['max_core_calls'] / result['max_core_calls'],
                SIMULATED_core_ms=result['predicted_core_critical_ms'],
                SIMULATED_core_speedup=baseline['predicted_core_critical_ms'] / result['predicted_core_critical_ms'],
                IDs=rids, placement=mapping, LIVE_E2E='NOT_AVAILABLE'))
    # Smallest synthetic unit check is in tests, never used as research evidence.
    write('cohort_statistics.json', stats_rows)
    csvout('request_statistics.csv', request_rows)
    csvout('rank_clean.csv', rank_rows)
    csvout('balance.csv', balance_rows)
    write('timing_comparison.json', time_summary)
    write('parity_summary.json', parity_summaries)
    csvout('parity_requests.csv', parity_rows)
    csvout('trace_events.csv.gz', trace_events)
    write('trace_summary.json', trace_summaries)
    csvout('trace_copy_by_depth.csv', copy_groups)
    write('observer_validity.json', observer)
    write('core_cost_model.json', model)
    write('frozen_replay.json', replay_rows)
    write('mechanism_cohorts.json', mechanism)
    write('same_length_selection.json', dict(N=len(same_length), IDs=[r['id'] for r in same_length]))
    # This bound applies only to the additive row-work surrogate.
    # It is not an upper bound on physical latency speedup.
    total = sum(weights.values())
    wave_lower_bound = max(max(weights.values()), total / (4 * 16))
    write('surrogate_bounds.json', dict(
        evidence='SIMULATED', additive_perfect_balance_work_lower_bound=total / 4,
        additive_speedup_upper_bound=base['max_work'] / (total / 4),
        unit_core_call_lower_bound=wave_lower_bound,
        unit_core_call_speedup_upper_bound=base['max_core_calls'] / wave_lower_bound,
        limitation='Loose bounds for stated surrogate models only. Neither is measured or achievable E2E headroom.'))
    # Preserve the archived evaluator result. Do not re-score answers here.
    qpath = ROOT / 'artifacts/quality_raw.jsonl'
    INPUTS[str(qpath.relative_to(ROOT.parent))] = hashlib.sha256(qpath.read_bytes()).hexdigest()
    scored = [json.loads(line) for line in qpath.read_text().splitlines()]
    quality_direct = []
    for domain in ['gsm8k', 'math500']:
        pair = [{r['id']: r for r in scored if r['arm'] == arm and r['dataset'] == domain}
                for arm in ['A1', 'A2']]
        common = sorted(pair[0])
        assert set(common) == pair[1].keys()
        delta = np.array([int(pair[1][rid]['answer_matches']) - int(pair[0][rid]['answer_matches']) for rid in common])
        idx = np.random.default_rng(SEED).integers(0, len(common), (2000, len(common)))
        quality_direct.append(dict(dataset=domain, N=len(common), evidence='LIVE_MEASURED',
            A1_correct=sum(r['answer_matches'] for r in pair[0].values()),
            A2_correct=sum(r['answer_matches'] for r in pair[1].values()),
            delta_pp=float(delta.mean() * 100), CI95_pp=np.quantile(delta[idx].mean(axis=1) * 100, [.025, .975]).tolist(),
            bootstrap_unit='Paired request/question',
            scoring='Archived answer_matches, includes parsed capped answers. Not re-scored.',
            A1_reasons=dict(collections.Counter(r['finish_reason'] for r in pair[0].values())),
            A2_reasons=dict(collections.Counter(r['finish_reason'] for r in pair[1].values()))))
    write('quality_direct_A1_A2.json', quality_direct)
    repeat_stability = []
    for C in [8, 16, 32, 64]:
        for arm in ['A1', 'A2']:
            for repeat in range(1, 5):
                p, _ = parity(speed[C, arm, 0][2], speed[C, arm, repeat][2], f'{C}_{arm}_r0_r{repeat}')
                repeat_stability.append(p)
    write('within_arm_repeat_parity.json', repeat_stability)
    write('analysis_contract.json', dict(
        primary='A1 vs A2, no-refill, physical GPUs 4,5,6,7',
        dataset_inference='Recorded cohorts only. Do not generalize capped speed traces to full quality.',
        token_denominator='Decode exit traces exclude the first token from full-depth prefill.',
        variance='Population variance (ddof=0). CV = standard deviation / mean.',
        timeline='Host-relative request times, same controller origin. Rank finish is not exclusive GPU service.',
        trace='CUDA event stream brackets, not per-kernel timing. No non-overlap E2E attribution.',
        hindsight='Historical scalar sum of decode depths, greedy LPT, not a perfect oracle.',
        replay='Frozen A1 outputs and depths. Counterfactual placements do not establish output parity.',
        cost_model='Trace recurrent-core bracket only. No valid prediction of clean request makespan.',
        new_GPU_experiment=False, scheduler_or_model_change=False,
        heuristic_swap_iterations=1500, accepted_swaps=accepted, seed=SEED))

    # Plots use archived measurements and label structural models separately.
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, arm in enumerate(['A1', 'A2']):
        rr = [r for r in rank_rows if r['C'] == 64 and r['repeat'] == 0 and r['arm'] == arm]
        ax.bar(np.arange(4) + .34 * i, [r['total_loop_work'] for r in rr], width=.34, label=arm)
    ax.set(xticks=np.arange(4) + .17, xticklabels=['GPU4', 'GPU5', 'GPU6', 'GPU7'], ylabel='Recorded decode loop-row work',
           title='C64 clean repeat 0 — LIVE_MEASURED')
    ax.legend(); fig.tight_layout(); fig.savefig(DEST / 'gpu_work.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for arm in ['A1', 'A2']:
        ds = np.concatenate([depths(r) for r in speed[64, arm, 0][2]])
        axs[0].hist(ds, bins=[1.5, 2.5, 3.5, 4.5], weights=np.ones(len(ds)) / len(ds), alpha=.5, label=arm)
        axs[1].hist([float(depths(r).mean()) for r in speed[64, arm, 0][2]], bins=14, alpha=.5, label=arm)
    axs[0].set(xlabel='Actual exit depth', ylabel='Token fraction', xticks=[2, 3, 4])
    axs[1].set(xlabel='Request mean exit depth', ylabel='Requests')
    for ax in axs: ax.legend()
    fig.suptitle('C64 clean repeat 0 — LIVE_MEASURED'); fig.tight_layout(); fig.savefig(DEST / 'depth_histogram.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for r in [r for r in trace_summaries if r['C'] == 64]:
        label = f"{r['arm']}/GPU{r['GPU']}"
        measured = r['measured_ms']['compute_recurrent_batch']
        axs[0].scatter(r['real_rows'], measured)
        axs[0].annotate(label, (r['real_rows'], measured), fontsize=7, xytext=(-3, 4),
                        textcoords='offset points', ha='right' if r['real_rows'] > 12400 else 'left')
        axs[1].scatter(r['final_exit_variance_batch_mean'], measured)
        axs[1].annotate(label, (r['final_exit_variance_batch_mean'], measured), fontsize=7,
                        xytext=(-3, 4), textcoords='offset points',
                        ha='right' if r['final_exit_variance_batch_mean'] > .14 else 'left')
    axs[0].xaxis.set_major_locator(MaxNLocator(5))
    axs[0].set(xlabel='Loop-row work', ylabel='Core stream bracket (ms)')
    axs[1].set(xlabel='Mean within-batch final-exit variance', ylabel='Core stream bracket (ms)')
    fig.suptitle('C64 trace lane — descriptive only, 8 rank observations'); fig.tight_layout()
    fig.savefig(DEST / 'work_depth_vs_core_time.png', dpi=160); plt.close(fig)
    csvout('cohort_statistics_flat.csv', [dict(cohort=r['cohort'], dataset=r['dataset'], requests=r['requests'],
        decode_tokens=r['recorded_decode_tokens'], token_mean=r['token_depth']['mean'], token_variance=r['token_depth']['variance'],
        token_CV=r['token_depth']['CV'], request_mean_CV=r['request_mean_depth']['CV'],
        request_work_CV=r['request_total_work']['CV']) for r in stats_rows])
    inventory = []
    for relative in ['artifacts/placements.jsonl', 'artifacts/rank_timeline.jsonl.gz', 'artifacts/metrics_raw.csv',
                     'reports/final_report.md', 'reports/timing_contract.md', 'contract.py',
                     'scripts/freeze.py', 'scripts/prepare_plans.py', 'scripts/run.py',
                     'scripts/worker.py', 'scripts/analyze.py', 'scripts/trace_analysis.py']:
        path = ROOT / relative
        INPUTS[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append(dict(path=str(path.relative_to(ROOT.parent)), exists=True, bytes=path.stat().st_size))
    cdb = Path('/home/esjung/external/decode-first-cdb-upstream')
    external = {}
    for relative in ['src/looped_cdb/continuous_depth_batching/continuous_api.py',
                     'src/looped_cdb/continuous_depth_batching/cache.py',
                     'src/looped_cdb/continuous_depth_batching/model_runner.py',
                     'src/looped_cdb/kv_cache_policy.py']:
        path = cdb / relative
        external[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = dict(required_artifacts=inventory, external_source_sha256=external,
        cdb_checkout_SHA=subprocess.check_output(['git', '-C', str(cdb), 'rev-parse', 'HEAD'], text=True).strip(),
        both_arms_config_identical=True, manifest_model_revision=manifest['model_revision'],
        topology='Four independent TP1/EP1/PP1 replicas, GPU 4/5/6/7. No EP All-to-All.',
        CPU_only_reanalysis=True)
    write('source_audit.json', audit)
    write('input_manifest.json', dict(source_commit='9726dc30729f160f7772ac484bc5ee01d596351c',
                                    prior_commit='0eb9c04d1eef6af33c30a0ca2c0244fb26e41950', sha256=INPUTS))
    print(json.dumps(dict(status='ANALYSIS_COMPLETE', inputs=len(INPUTS), rank_records=len(rank_rows),
                          trace_events=len(trace_events), request_summaries=len(request_rows),
                          C64=time_summary[-1], parity=parity_summaries[15]), indent=2))


if __name__ == '__main__':
    main()
