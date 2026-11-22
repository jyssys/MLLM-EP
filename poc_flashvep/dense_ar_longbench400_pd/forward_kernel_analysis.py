"""Check all short-forward observer triplets without changing the five-percent rule."""
import collections
import csv
import json
import numpy as np
from .common import ROOT, write
from .metrics import summarize
from .service_accounting import analyze as service
from .operator_repeat_analysis import paired_ci
from .profile_analysis import union


def name_categories():
    associations = collections.defaultdict(set)
    contract = json.loads((ROOT / 'configs/operator_repeat_observer_contract_v3.json').read_text())
    for test in contract['tests']:
        for role in ('prefill', 'decode'):
            path = ROOT / 'raw' / test['jobs'][1] / ('activity_' + role)
            scopes = json.loads((path / 'correlation_scopes.json').read_text())
            ops = {r['external_id']: r for r in scopes['operators']}
            ids = {int(r['correlation']): int(r['external_id'])
                   for r in csv.DictReader((path / 'kernels.csv.external.csv').open())
                   if int(r['external_kind']) == 3}
            names = {int(r['name_id']): r['kernel_name']
                     for r in csv.DictReader((path / 'kernels.csv.names.csv').open())}
            for row in csv.DictReader((path / 'kernels.csv').open()):
                if row['kind'] != 'kernel':
                    continue
                op = ops.get(ids.get(int(row['correlation'])))
                if op is None:
                    continue
                category = op['category']
                if category.startswith('GEMM'):
                    category = 'GEMM_INCLUDING_LM_HEAD'
                associations[names[int(row['name_id'])]].add(category)
    return {name: next(iter(values)) if len(values) == 1 else 'AMBIGUOUS_NAME'
            for name, values in associations.items()}


def kernels(name, role, categories):
    folder = ROOT / 'raw' / name
    queries = json.loads((folder / f'actual_queries_{role}.json').read_text())['rows']
    assert all(r['start'] < r['original_input_tokens'] for r in queries), 'Unexpected decode work in output1 control'
    path = folder / ('activity_' + role)
    counters = json.loads((path / 'activity_counters.json').read_text())
    assert counters['passed'] and counters['counters']['dropped'] == counters['counters']['invalid'] == 0
    names = {int(r['name_id']): r['kernel_name']
             for r in csv.DictReader((path / 'kernels.csv.names.csv').open())}
    groups = collections.defaultdict(list)
    copies = []
    all_kernels = []
    for row in csv.DictReader((path / 'kernels.csv').open()):
        interval = (int(row['start_ns']), int(row['end_ns']))
        if row['kind'] == 'kernel':
            all_kernels.append(interval)
            groups[categories.get(names[int(row['name_id'])], 'UNCLASSIFIED')].append(interval)
        elif row['kind'] in ('memcpy', 'memcpy2', 'memset'):
            copies.append(interval)
    total = union(all_kernels) / 1e6
    costs = {key: union(values) / 1e6 for key, values in groups.items()}
    unknown = costs.get('UNCLASSIFIED', 0) + costs.get('AMBIGUOUS_NAME', 0)
    return dict(kernel_union_ms_per_request=total / 16,
        named_category_union_ms_per_request={k: v / 16 for k, v in costs.items()},
        observed_copy_memset_union_ms=union(copies) / 1e6, kernel_count=len(all_kernels),
        named_category_coverage_fraction=1 - unknown / total,
        counters_passed=True,
        scope='Exact kernel-name associations come from independent call-correlated serving captures. '
            'Unknown or conflicting names remain unclassified. GEMM includes LM-head work. '
            'Category unions can overlap across streams. Do not sum them as an exclusive total.')


def main():
    categories = name_categories()
    results = []
    for case in (0, 1):
        contract = json.loads((ROOT / 'configs' / f'forward_kernel_only_case{case}_observer_v1.json').read_text())
        for axis in ('Layer', 'Token'):
            tests = [t for t in contract['triplets'] if t['axis'] == axis]
            assert len(tests) == 3
            triplets = []
            for test in tests:
                metric = []
                for name in test['jobs']:
                    m = summarize(name)
                    assert m['n'] == 16 and m['output_tokens'] == 16 and m['ITL_interval_count'] == 0
                    s = service(name)['means']
                    metric.append(dict(TTFT_p95_ms=m['TTFT_p95_ms'], TTFT_mean_ms=m['TTFT_mean_ms'],
                        P_submit_to_ready_ms=s['P_submit_to_ready_ms'],
                        D_submit_to_first_worker_ms=sum(s[k] for k in ('D_submit_to_accept_ms',
                            'D_accept_to_prepare_ms', 'D_prepare_to_first_worker_ms'))))
                clean = {k: (metric[0][k] + metric[2][k]) / 2 for k in metric[1]}
                triplets.append(dict(start=test['start'], jobs=test['jobs'], profile=metric[1], mean_clean=clean,
                    profile_over_mean_clean={k: metric[1][k] / clean[k] for k in clean},
                    roles={role: kernels(test['jobs'][1], role, categories) for role in ('prefill', 'decode')}))
            ci = {k: paired_ci([t['profile'][k] for t in triplets], [t['mean_clean'][k] for t in triplets])
                  for k in triplets[0]['profile']}
            passed = all(.95 <= v['CI95'][0] <= v['CI95'][1] <= 1.05 for v in ci.values())
            row = dict(input_bucket=0, cut=tests[0]['cut'], axis=axis, triplets=triplets,
                observer_CI=ci, observer_passed=passed,
                evidence='ISOLATED_KERNEL_MEASUREMENT' if passed else 'DIAGNOSTIC',
                mean_role_costs={role: dict(kernel_union_ms_per_request=float(np.mean([
                    t['roles'][role]['kernel_union_ms_per_request'] for t in triplets])),
                    named_categories_ms_per_request={k: float(np.mean([
                        t['roles'][role]['named_category_union_ms_per_request'].get(k, 0) for t in triplets]))
                        for k in sorted({k for t in triplets for k in t['roles'][role]['named_category_union_ms_per_request']})})
                    for role in ('prefill', 'decode')})
            results.append(row)
            print('SHORT_FORWARD_OBSERVER', axis, row['cut'], passed,
                  {k: v['CI95'] for k, v in ci.items()}, row['mean_role_costs'], flush=True)
    write(ROOT / 'analysis/short_forward_kernel_only_observer_v1.json', dict(cells=results,
        name_association_count=len(categories),
        scope='Three clean/profile/clean triplets per axis and cut. Output1 excludes subsequent decode. '
            'All prompt shapes were warmed. This is not online serving, a natural-output result, '
            'a decode-safety measurement, or a measurement of SM utilization and HBM traffic.'))


if __name__ == '__main__':
    main()
