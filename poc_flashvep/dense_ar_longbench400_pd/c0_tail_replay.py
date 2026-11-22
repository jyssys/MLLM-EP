"""Test a state-updating empirical decode-tail model before any SLO oracle."""
import collections
import hashlib
import heapq
import json
import numpy as np
from .common import ROOT, write
from .c0_queue_model import features, part
from .metrics import delivery_intervals, summarize


def signature(p):
    # Integer layer count and normalized suffix preserve the calibrated action family.
    return p.upper, round(p.suffix / p.length, 2)


def templates(training):
    rows = []
    for name in training:
        folder = ROOT / 'raw' / name
        cfg = json.loads((folder / 'configuration.json').read_text())
        for record in map(json.loads, (folder / 'requests.jsonl').read_text().splitlines()):
            i = record['sample_id']
            p = part(cfg, i, record['input_tokens'])
            intervals = delivery_intervals(record['token_events'], record['output_tokens'])
            if not intervals:
                continue
            rows.append(dict(signature=signature(p), length=p.length, intervals=intervals,
                finish_gap_ms=max(0., 1000 * (record['finish'] - record['token_events'][-1]['time']))))
    return rows


def simulate(arrivals, lengths, partitions, output, model, pool, seed):
    rng = np.random.default_rng(seed)
    decode = []
    for p in partitions:
        same = [r for r in pool if r['signature'] == signature(p)]
        if not same:
            same = sorted(pool, key=lambda r: abs(r['signature'][0] - p.upper)
                + 36 * abs(r['signature'][1] - p.suffix / p.length))[:24]
        nearby = sorted(same, key=lambda r: abs(np.log(r['length'] / p.length)))[:12]
        draw = nearby[int(rng.integers(len(nearby)))]
        source = draw['intervals']
        # Whole measured blocks preserve local dependence. Longer output extrapolation remains a hypothesis.
        shift = int(rng.integers(len(source)))
        intervals = [source[(shift + j) % len(source)] for j in range(output - 1)]
        decode.append((intervals, draw['finish_gap_ms']))
    events = [(float(t), 0, i) for i, t in enumerate(arrivals)]
    heapq.heapify(events)
    pending = collections.deque()
    ready = collections.deque()
    owned = set()
    pactive = dactive = None
    records = [{} for _ in arrivals]
    while events:
        t, event, i = heapq.heappop(events)
        if event == 0:
            pending.append(i)
        elif event == 1:
            assert pactive == i
            pactive = None
            ready.append(i)
        elif event == 2:
            owned.remove(i)
            records[i]['first'] = t
        else:
            assert dactive == i
            dactive = None
            records[i]['finish'] = t
        if dactive is None and ready:
            j = ready.popleft()
            dactive = j
            first = max(.001, float(np.dot(features(partitions[j], 'D'), model['D']))) / 1000
            intervals, gap = decode[j]
            heapq.heappush(events, (t + first, 2, j))
            heapq.heappush(events, (t + first + (sum(intervals) + gap) / 1000, 3, j))
        if pactive is None and pending and len(owned) < 4:
            j = pending.popleft()
            pactive = j
            owned.add(j)
            cost = max(.001, float(np.dot(features(partitions[j], 'P'), model['P']))) / 1000
            heapq.heappush(events, (t + cost, 1, j))
    ttft = [1000 * (r['first'] - a) for r, a in zip(records, arrivals)]
    all_itl = [x for intervals, _ in decode for x in intervals]
    tpots = [sum(intervals) / len(intervals) for intervals, _ in decode]
    good = [delay <= 5000 and max(intervals, default=0) <= 34 and tpot <= 33
            for delay, (intervals, _), tpot in zip(ttft, decode, tpots)]
    duration = max(r['finish'] for r in records)
    return dict(TTFT_p95_ms=float(np.percentile(ttft, 95)),
        ITL_p99_ms=float(np.percentile(all_itl, 99)), TPOT_p95_ms=float(np.percentile(tpots, 95)),
        Joint_SLO_attainment=float(np.mean(good)),
        SLO_qualified_output_tokens_per_second=output * sum(good) / duration,
        generated_output_tokens_per_second=output * len(records) / duration)


def main():
    contract = dict(version=1, Monte_Carlo_draws_per_trace=16,
        validation='All unique held-out C0 axis-confirmation starts. No validation output enters calibration.',
        median_relative_error_limit_percent=dict(TTFT_p95_ms=10, ITL_p99_ms=20, TPOT_p95_ms=20),
        p90_relative_error_limit_percent=dict(TTFT_p95_ms=25, ITL_p99_ms=40, TPOT_p95_ms=40),
        joint_SLO_absolute_error_limits_percentage_points=dict(median=10, p90=20),
        SLO_goodput_absolute_error_percent_of_actual_raw_limits=dict(median=10, p90=20),
        phase='POSTHOC_MODEL_EXTENSION_WITH_FROZEN_ERROR_LIMITS_BEFORE_THIS_MODEL_OUTPUT',
        scope='Measured contiguous ITL templates condition on static action and nearby input length. '
            'Repeated blocks extrapolate output length and can fail. No claim of independent token samples. '
            'This is C0 whole-request D service,not C3 mixed batching or an online safe-window predictor. '
            'A failed gate blocks SLO-oracle gain claims. Simulation draws are not independent server starts.')
    path = ROOT / 'configs/c0_tail_replay_validation_v1.json'
    if path.exists():
        assert json.loads(path.read_text()) == contract
    else:
        write(path, contract)
    source = json.loads((ROOT / 'analysis/c0_host_service_model_v1.json').read_text())
    model = source['model']
    pool = templates(source['training'])
    protocol = json.loads((ROOT / 'configs/heldout64_confirmation_protocol_r2.json').read_text())
    names = sorted({c[a] for c in protocol['cells'] for a in ('Native', 'Layer', 'Token')})
    rows = []
    for index, name in enumerate(names):
        cfg = json.loads((ROOT / 'raw' / name / 'configuration.json').read_text())
        data = json.loads((ROOT / 'data' / cfg['dataset']).read_text())[:cfg['count']]
        lengths = [r['input_tokens'] for r in data]
        parts = [part(cfg, i, length) for i, length in enumerate(lengths)]
        actual = summarize(name)
        base = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        draws = [simulate(cfg['arrival_offsets'], lengths, parts, cfg['output_cap'], model, pool, base + i)
                 for i in range(16)]
        prediction = {k: float(np.median([d[k] for d in draws])) for k in draws[0]}
        errors = {k: 100 * abs(prediction[k] / actual[k] - 1)
                  for k in ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms')}
        errors['joint_SLO_percentage_points'] = 100 * abs(prediction['Joint_SLO_attainment'] - actual['Joint_SLO_attainment'])
        errors['goodput_percent_of_actual_raw'] = 100 * abs(prediction['SLO_qualified_output_tokens_per_second']
            - actual['SLO_qualified_output_tokens_per_second']) / actual['generated_output_tokens_per_second']
        rows.append(dict(name=name, predicted=prediction, actual={k: actual[k] for k in prediction}, errors=errors))
        if index % 40 == 0:
            print('TAIL_REPLAY_VALIDATION_PROGRESS', index, len(names), flush=True)
    limits = {k: (contract['median_relative_error_limit_percent'][k], contract['p90_relative_error_limit_percent'][k])
              for k in contract['median_relative_error_limit_percent']}
    limits.update(joint_SLO_percentage_points=(10, 20), goodput_percent_of_actual_raw=(10, 20))
    checks = {}
    for k, (median_limit, p90_limit) in limits.items():
        values = [r['errors'][k] for r in rows]
        median, p90 = float(np.median(values)), float(np.percentile(values, 90))
        checks[k] = dict(median=median, p90=p90, median_limit=median_limit, p90_limit=p90_limit,
                         passed=median <= median_limit and p90 <= p90_limit)
    passed = all(c['passed'] for c in checks.values())
    write(ROOT / 'analysis/c0_tail_replay_heldout_validation_v1.json', dict(rows=rows, checks=checks,
        calibration_templates=len(pool), heldout_starts=len(rows), SLO_replay_gate_passed=passed,
        online_admission_allowed=False, evidence='OFFLINE_REPLAY',
        scope=contract['scope'] + ' Passing these error limits still does not prove a dynamic serving speedup.'))
    print('TAIL_REPLAY_GATE', passed, checks, flush=True)


if __name__ == '__main__':
    main()
