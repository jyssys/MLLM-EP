"""Keep failed selections separate from confirmed lower-rate capacity."""
import collections
import csv
import itertools
import json
import statistics
import time

from poc_flashvep.dense_ar_slo_layer_pd.common import ROOT, write
from poc_flashvep.dense_ar_slo_layer_pd.final_analysis import load, mean_CI
from poc_flashvep.dense_ar_slo_layer_pd.analyze import percentile

REGIMES = ('TTFT_priority', 'Balanced', 'Decode_priority')
METHODS = ('Native', 'Colocated', 'Static', 'Dynamic')
FULL_METRICS = (
    'TTFT_p50_ms', 'TTFT_p95_ms', 'TTFT_p99_ms',
    'TPOT_p50_ms', 'TPOT_p95_ms', 'TPOT_p99_ms',
    'ITL_p50_ms', 'ITL_p95_ms', 'ITL_p99_ms', 'maximum_ITL_ms',
    'E2E_p50_ms', 'E2E_p95_ms', 'E2E_p99_ms',
    'raw_completed_output_tokens_s', 'completed_output_requests_s',
)


def independent_ratio(numerator, denominator):
    a = [statistics.mean(numerator[i] for i in indices)
         for indices in itertools.product(range(len(numerator)), repeat=len(numerator))]
    b = [statistics.mean(denominator[i] for i in indices)
         for indices in itertools.product(range(len(denominator)), repeat=len(denominator))]
    ratios = [left / right for left in a for right in b if right > 0]
    return dict(ratio_of_means=statistics.mean(numerator) / statistics.mean(denominator),
        independent_start_counts=[len(numerator), len(denominator)],
        empirical_independent_bootstrap_95_CI=[percentile(ratios, .025), percentile(ratios, .975)],
        bootstrap_draws=len(ratios),
        scope='The compared methods use different arrival seeds. Resample their independent starts separately. Do not call this a paired interval. Three starts give a coarse empirical interval.')


def main():
    assert (ROOT / 'state/CAMPAIGN_CLOSED.json').exists()
    values = load()
    inventory = json.loads((ROOT / 'analysis/campaign_result_inventory.json').read_text())
    actual = {row['name']: row for row in inventory['runs']}
    grouped = collections.defaultdict(list)
    for row in values:
        if row['name'].startswith(('heldout_capacity_', 'heldout_backoff_')):
            grouped[(row['method'], row['rate'])].append(row)
    groups = []
    for (method, rate), rows in sorted(grouped.items()):
        record = dict(method=method, offered_QPS=rate, independent_starts=len(rows),
            names=[row['name'] for row in rows], seeds=[row['seed'] for row in rows],
            metrics={key: mean_CI([row[key] for row in rows]) for key in FULL_METRICS},
            realized_QPS=mean_CI([row['stability']['arrival_rate'] for row in rows]),
            SLO_regimes={}, migration=[])
        for regime in REGIMES:
            passes = [row['SLO_regimes'][regime]['qualified'] for row in rows]
            record['SLO_regimes'][regime] = dict(
                confirmed=len(passes) == 3 and all(passes),
                passed_starts=sum(passes), total_starts=len(passes),
                qualified_tokens_s=mean_CI([row['SLO_regimes'][regime]['SLO_qualified_output_tokens_s'] for row in rows]),
                arrival_joint=mean_CI([row['SLO_regimes'][regime]['arrival_cohort_joint_attainment'] for row in rows]),
                completion_joint=mean_CI([row['SLO_regimes'][regime]['completion_window_joint_attainment'] for row in rows]),
                run_rejections={row['name']: row['SLO_regimes'][regime]['rejections'] for row in rows})
        for row in rows:
            observed = actual[row['name']]
            count = observed['measurement_arrival_request_count']
            requests = observed['measurement_arrival_migrated_requests']
            work = observed['measurement_arrival_migrated_layer_work']
            record['migration'].append(dict(name=row['name'], measurement_arrivals=count,
                migrated_requests=requests, migrated_layer_work=work,
                migrated_request_fraction=requests / count,
                migrated_prefill_layer_work_fraction=work / (36 * count)))
        groups.append(record)
    confirmed = {}
    for regime in REGIMES:
        confirmed[regime] = {}
        for method in METHODS:
            eligible = [row for row in groups if row['method'] == method and row['SLO_regimes'][regime]['confirmed']]
            confirmed[regime][method] = max(eligible, key=lambda row: row['offered_QPS']) if eligible else dict(
                status='NOT_ESTABLISHED', reason='No tested rate passed all3 independent starts for this regime. This is not zero capacity.')
    primary = confirmed['Decode_priority']
    comparison = dict(status='NOT_ESTABLISHED')
    static_comparison = dict(status='NOT_ESTABLISHED')
    if 'names' in primary['Dynamic'] and 'names' in primary['Colocated']:
        index = {row['name']: row for row in values}
        dynamic = [index[name] for name in primary['Dynamic']['names']]
        colocated = [index[name] for name in primary['Colocated']['names']]
        comparison = dict(
            qualified_tokens_s=independent_ratio(
                [row['SLO_regimes']['Decode_priority']['SLO_qualified_output_tokens_s'] for row in dynamic],
                [row['SLO_regimes']['Decode_priority']['SLO_qualified_output_tokens_s'] for row in colocated]),
            raw_tokens_s=independent_ratio(
                [row['raw_completed_output_tokens_s'] for row in dynamic],
                [row['raw_completed_output_tokens_s'] for row in colocated]),
            status='MEASURED',
            scope='Compare confirmed operating rates under the same frozen Decode-priority SLO. This is not a comparison at equal offered QPS. No exact global maximum is claimed.')
    if 'names' in primary['Dynamic'] and 'names' in primary['Static']:
        index = {row['name']: row for row in values}
        dynamic = [index[name] for name in primary['Dynamic']['names']]
        static = [index[name] for name in primary['Static']['names']]
        static_comparison = dict(
            qualified_tokens_s=independent_ratio(
                [row['SLO_regimes']['Decode_priority']['SLO_qualified_output_tokens_s'] for row in dynamic],
                [row['SLO_regimes']['Decode_priority']['SLO_qualified_output_tokens_s'] for row in static]),
            TTFT_p95_ms=independent_ratio(
                [row['TTFT_p95_ms'] for row in dynamic], [row['TTFT_p95_ms'] for row in static]),
            status='MEASURED_UNMATCHED_WORK_CONTROL',
            scope='Both confirmed rates are4.4, but arrival seeds, migrated work and realized group geometry differ. This is not a matched-budget or paired comparison. It cannot establish an independent scheduling contribution.')
    persistence = [dict(name=row['name'], regime_results=row['SLO_regimes'], stability=row['stability'])
        for row in values if row['name'].startswith('heldout_persistence_')]
    result = dict(time=time.time(), groups=groups, confirmed_by_regime=confirmed,
        Dynamic_vs_Colocated_confirmed_points=comparison,
        Dynamic_vs_Static_confirmed_points_unmatched_work=static_comparison,
        persistence=persistence,
        evidence='MEASURED',
        scope='Use three independent starts at an unchanged method configuration. A lower rate does not erase a failed higher-rate selection. Treat incomplete or failed rate groups as NOT_ESTABLISHED, never zero throughput. Preserve the separate1200-second persistence result. Infinite-horizon stability and exact global maxima remain unproven.')
    write(ROOT / 'analysis/confirmed_capacity_with_backoff.json', result)
    fields = ['method', 'offered_QPS', 'independent_starts', 'Decode_passed_starts',
        *FULL_METRICS, 'Decode_qualified_tokens_s', 'Decode_arrival_joint', 'confirmed_Decode_point']
    with (ROOT / 'analysis/confirmed_capacity_groups.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in groups:
            slo = row['SLO_regimes']['Decode_priority']
            writer.writerow(dict(method=row['method'], offered_QPS=row['offered_QPS'],
                independent_starts=row['independent_starts'], Decode_passed_starts=slo['passed_starts'],
                **{key: row['metrics'][key]['mean'] for key in FULL_METRICS},
                Decode_qualified_tokens_s=slo['qualified_tokens_s']['mean'],
                Decode_arrival_joint=slo['arrival_joint']['mean'], confirmed_Decode_point=slo['confirmed']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for position, (method, color) in enumerate(zip(METHODS, ('#666666', '#0072b2', '#009e73', '#d55e00'))):
        row = primary[method]
        if 'names' not in row:
            ax.text(position, 8, 'NOT\nESTABLISHED', ha='center', fontsize=8)
            continue
        value = row['SLO_regimes']['Decode_priority']['qualified_tokens_s']
        mean = value['mean']
        lo, hi = value['CI95']
        ax.bar(position, mean, color=color)
        ax.errorbar(position, mean, yerr=[[max(0, mean - lo)], [max(0, hi - mean)]], color='black', capsize=4)
        ax.text(position, mean + 2, f"QPS {row['offered_QPS']}\n3/3 starts pass", ha='center', fontsize=8)
    ax.set_xticks(range(4), METHODS)
    ax.set_xlim(-.6, 3.6)
    ax.set_ylabel('Decode-priority qualified completed output tokens/s')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=.2)
    ax.set_title('Largest confirmed tested rates, not exact global maxima.\nFrozen method, with a declared lower-rate supplement.')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/final_9_confirmed_capacity_with_backoff.svg')
    plt.close(fig)
    print('CONFIRMED_CAPACITY_WITH_BACKOFF_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
