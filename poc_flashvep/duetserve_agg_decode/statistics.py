"""Use independent starts, not tokens, as the uncertainty unit."""
import itertools
import json
import math
import statistics

from .common import ROOT, write


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo, hi = int(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def mean_ci(values):
    values = list(values)
    mean = statistics.mean(values)
    if len(values) < 3:
        return dict(mean=mean, n_starts=len(values), CI95=None, status='INSUFFICIENT_INDEPENDENT_STARTS')
    draws = [statistics.mean(values[i] for i in indices)
             for indices in itertools.product(range(len(values)), repeat=len(values))]
    return dict(mean=mean, n_starts=len(values), CI95=[percentile(draws, .025), percentile(draws, .975)],
        status='EMPIRICAL_START_BOOTSTRAP', bootstrap_draws=len(draws))


def paired_ratio(numerator, denominator):
    assert len(numerator) == len(denominator) and numerator
    n = len(numerator)
    ratio = statistics.mean(numerator) / statistics.mean(denominator)
    if n < 3:
        return dict(ratio=ratio, n_pairs=n, CI95=None, status='POINT_ESTIMATE_ONLY')
    draws = [statistics.mean(numerator[i] for i in indices) / statistics.mean(denominator[i] for i in indices)
             for indices in itertools.product(range(n), repeat=n)]
    return dict(ratio=ratio, n_pairs=n, CI95=[percentile(draws, .025), percentile(draws, .975)],
        bootstrap_draws=len(draws), status='PAIRED_ARRIVAL_SEED_START_BOOTSTRAP')


def ranks(values):
    ordered = sorted(enumerate(values), key=lambda x: x[1])
    result = [None] * len(values)
    for _, group in itertools.groupby(ordered, key=lambda x: x[1]):
        group = list(group)
        rank = statistics.mean(1 + sum(other < group[0][1] for other in values) + j for j in range(len(group)))
        for index, _ in group:
            result[index] = rank
    return result


def spearman(a, b):
    a, b = ranks(a), ranks(b)
    am, bm = statistics.mean(a), statistics.mean(b)
    den = math.sqrt(sum((x - am) ** 2 for x in a) * sum((y - bm) ** 2 for y in b))
    return sum((x - am) * (y - bm) for x, y in zip(a, b)) / den if den else None


def collect():
    groups = {}
    for path in sorted((ROOT / 'raw').glob('*/metrics.json')):
        metric = json.loads(path.read_text())
        config = json.loads((ROOT / 'configs' / (path.parent.name + '.json')).read_text())
        if config.get('correctness_only') or not metric.get('complete'):
            continue
        if not metric.get('output_gate', {}).get('passed'):
            continue
        if path.parent.name.startswith('tp_diagnostic_'):
            assert config['comparison_eligibility'] == 'DIAGNOSTIC_WITHIN_TP2_ONLY_TP1_PARITY_FAILED'
            family = 'TP2_DIAGNOSTIC'
        elif path.parent.name.startswith('confirm_'):
            family = ('PRE_SCOPE_REVISION_SMALL_BUDGET' if
                (ROOT / 'state/user_primary_scope_revision_v2.json').exists() and
                config['arm'] == 'DP2' and config['token_budget'] == 512 else 'FRESH_CONFIRMATION')
        elif path.parent.name.startswith(('sweep_', 'concurrency_')):
            family = 'DISCOVERY'
        elif path.parent.name.startswith('budget_'):
            family = 'BUDGET_CONTROL'
        elif path.parent.name.startswith('secondary_'):
            family = 'SECONDARY'
        else:
            continue
        arm = config['arm'] + f"_B{config['token_budget']}_S{config['D_slots']}_{config['ar_cudagraph_mode']}"
        key = (family, config['cap'], arm, config['arrival_rate'])
        groups.setdefault(key, []).append((config, metric))
    metrics = ('mean_TTFT_ms', 'TTFT_p95_ms', 'mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms',
               'maximum_ITL_ms', 'output_tokens_s', 'completed_QPS',
               'first_ITL_p99_ms', 'steady_ITL_p99_ms', 'mean_worker_emission_TBT_ms')
    aggregate = []
    for key, rows in sorted(groups.items()):
        family, cap, arm, qps = key
        aggregate.append(dict(family=family, output_cap=cap, arm=arm, offered_QPS=qps,
            starts=[c['name'] for c, _ in rows], seeds=[c['arrival_seed'] for c, _ in rows],
            metrics={metric: mean_ci(m[metric] for _, m in rows) for metric in metrics},
            evidence='MEASURED',
            primary_comparison_eligible=family not in ('TP2_DIAGNOSTIC', 'PRE_SCOPE_REVISION_SMALL_BUDGET'),
            scope='Mean of independent-start summaries. No token-level bootstrap. Discovery and fresh prompts remain separate.'))
    ratios = []
    keys = {(family, cap, arm) for family, cap, arm, _ in groups}
    for family, cap, arm in sorted(keys):
        points = sorted(qps for f, o, a, qps in groups if (f, o, a) == (family, cap, arm))
        if len(points) < 2:
            continue
        low, high = points[0], points[-1]
        lo = {c['arrival_seed']: m for c, m in groups[(family, cap, arm, low)]}
        hi = {c['arrival_seed']: m for c, m in groups[(family, cap, arm, high)]}
        seeds = sorted(set(lo) & set(hi))
        if not seeds:
            continue
        tests = {}
        for metric in ('mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms'):
            result = paired_ratio([hi[s][metric] for s in seeds], [lo[s][metric] for s in seeds])
            means = [statistics.mean(m[metric] for _, m in groups[(family, cap, arm, q)]) for q in points]
            result.update(spearman_QPS_metric=spearman(points, means),
                adjacent_increases=sum(b > a for a, b in zip(means, means[1:])),
                adjacent_pairs=len(points) - 1,
                first_QPS_point_ratio_at_least_2=next((q for q, value in zip(points, means) if value / means[0] >= 2), None),
                point_ratio_at_least_2=result['ratio'] >= 2)
            if result['CI95'] is None:
                result['repeat_qualified_2x'] = 'NOT_ESTABLISHED'
            elif result['CI95'][0] >= 2:
                result['repeat_qualified_2x'] = 'PASS'
            elif result['CI95'][1] < 2:
                result['repeat_qualified_2x'] = 'NOT_OBSERVED_AT_TESTED_ENDPOINTS'
            else:
                result['repeat_qualified_2x'] = 'INCONCLUSIVE_AT_2X'
            tests[metric] = result
        ratios.append(dict(family=family, output_cap=cap, arm=arm, low_QPS=low, high_QPS=high,
                           paired_seeds=seeds, tests=tests))
    isolation = []
    for (family, cap, arm, qps), rows in sorted(groups.items()):
        if not arm.startswith(('DP2_', 'TP2_')):
            continue
        native_key = (family, cap, 'PD_G1_B32000_S16_FULL_DECODE_ONLY', qps)
        if native_key not in groups:
            continue
        native = {c['arrival_seed']: m for c, m in groups[native_key]}
        observed = {c['arrival_seed']: m for c, m in rows}
        seeds = sorted(set(native) & set(observed))
        if not seeds:
            continue
        isolation.append(dict(family=family, output_cap=cap, arm=arm, offered_QPS=qps,
            paired_seeds=seeds, tests={metric: paired_ratio(
                [observed[s][metric] for s in seeds], [native[s][metric] for s in seeds])
                for metric in ('mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms')},
            scope='Same-QPS Aggregated/Native isolation ratio. This is not the high-QPS/low-QPS load ratio.'))
    write(ROOT / 'analysis/independent_start_aggregates.json', aggregate)
    write(ROOT / 'analysis/load_degradation_tests.json', ratios)
    write(ROOT / 'analysis/same_QPS_isolation_tests.json', isolation)
    print('AGGREGATES', len(aggregate), 'LOAD_TESTS', len(ratios))
    return aggregate, ratios


if __name__ == '__main__':
    collect()
