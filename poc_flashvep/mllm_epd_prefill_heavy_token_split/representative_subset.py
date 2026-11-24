"""Freeze a representative Main100 without changing Main400 or its requests."""
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import time

import numpy as np

from .common import ROOT, digest, preserve, write

SEED = 2026112405
COUNT = 100
ITERATIONS = 60000
STARTS = 4
QUANTILES = [0, 10, 25, 50, 75, 90, 95, 99, 100]
METRICS = ['total_input_tokens', 'visual_tokens', 'text_tokens', 'special_tokens',
           'image_count', 'page_count', 'natural_output_tokens']


def quotas(rows, discovery_ids, count):
    """Keep source proportions and inherited splits, with integer rounding."""
    groups = {}
    for i, row in enumerate(rows):
        key = (row['source'], 'discovery' if row['request_id'] in discovery_ids else 'heldout')
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    targets = {key: len(groups[key]) * count / len(rows) for key in keys}
    base = {key: int(np.floor(targets[key])) for key in keys}
    remaining = count - sum(base.values())
    # The number of source/split strata is small. Check integer roundings exactly.
    choices = []
    for extra in combinations([key for key in keys if targets[key] > base[key]], remaining):
        trial = {key: base[key] + int(key in extra) for key in keys}
        if sum(v for (_, split), v in trial.items() if split == 'discovery') != count // 2:
            continue
        error = sum((trial[key] - targets[key]) ** 2 for key in keys)
        choices.append((error, extra, trial))
    assert choices, 'Source/split rounding cannot preserve the requested split sizes'
    chosen = min(choices, key=lambda x: (x[0], x[1]))[2]
    return groups, chosen


def feature_matrix(rows, outputs):
    """Match measured workload features, never method latency or answer quality."""
    values = {key: np.asarray([
        outputs[row['request_id']]['output_tokens'] if key == 'natural_output_tokens'
        else (row.get(key) or 0) for row in rows], dtype=float) for key in METRICS}
    vectors = []
    labels = []

    def add(label, vector, weight=1.0):
        vector = np.asarray(vector, dtype=float)
        if np.ptp(vector) > 0:
            vectors.append(vector * np.sqrt(weight))
            labels.append(dict(name=label, weight=weight))

    for key, x in values.items():
        add(key + ':mean', x / max(float(x.mean()), 1), 12)
        # Tail second moments can change sharply when a rare request is removed.
        # This low weight does not force inclusion of a single 2048-token cap hit.
        add(key + ':second_moment', x*x / max(float((x*x).mean()), 1), .05)
        for q in [10, 25, 50, 75, 90, 95, 99]:
            threshold = float(np.percentile(x, q))
            for factor in [.98, 1, 1.02]:
                add(f'{key}:CDF:{q}:{factor}', x <= threshold * factor, 1)
        for source in sorted({row['source'] for row in rows}):
            mask = np.asarray([row['source'] == source for row in rows])
            add(f'{key}:{source}:mean', mask*x / max(float(x[mask].mean()), 1), 2)
    input_bin = np.searchsorted(np.percentile(values['total_input_tokens'], [25, 50, 75]),
                                values['total_input_tokens'], side='right')
    output_bin = np.searchsorted(np.percentile(values['natural_output_tokens'], [25, 50, 75]),
                                 values['natural_output_tokens'], side='right')
    for i in range(4):
        for j in range(4):
            add(f'input_output_quartile:{i}:{j}', (input_bin == i) & (output_bin == j), 2)
    return np.asarray(vectors).T, labels, values


def select(rows, outputs, discovery_ids, count=COUNT, seed=SEED,
           iterations=ITERATIONS, starts=STARTS):
    groups, allocation = quotas(rows, discovery_ids, count)
    matrix, labels, values = feature_matrix(rows, outputs)
    target = matrix.sum(axis=0) * count / len(rows)
    rng = np.random.default_rng(seed)
    best = None
    movable = [np.asarray(indices) for key, indices in sorted(groups.items())
               if 0 < allocation[key] < len(indices)]
    for start in range(starts):
        mask = np.zeros(len(rows), dtype=bool)
        for key, indices in sorted(groups.items()):
            mask[rng.choice(indices, allocation[key], replace=False)] = True
        residual = matrix[mask].sum(axis=0) - target
        score = float(residual @ residual)
        for step in range(iterations):
            group = movable[int(rng.integers(len(movable)))]
            selected = group[mask[group]]
            excluded = group[~mask[group]]
            old = int(selected[int(rng.integers(len(selected)))])
            new = int(excluded[int(rng.integers(len(excluded)))])
            delta = matrix[new] - matrix[old]
            change = float(2 * (residual @ delta) + delta @ delta)
            temperature = 3 * (1 - step / iterations) ** 3
            if change < 0 or (temperature > 0 and rng.random() < np.exp(-min(change / temperature, 700))):
                mask[old], mask[new] = False, True
                residual += delta
                score += change
            if best is None or score < best[0]:
                best = (score, mask.copy(), start, step)
    indices = np.flatnonzero(best[1]).tolist()
    assert len(indices) == count
    for key, group in groups.items():
        assert sum(best[1][group]) == allocation[key]
    return indices, dict(objective=best[0], chosen_start=best[2], chosen_step=best[3],
                        source_split_allocation=[dict(source=k[0], split=k[1], count=v)
                                                 for k, v in sorted(allocation.items())],
                        features=labels), values


def statistics(x):
    x = np.asarray(x, dtype=float)
    result = dict(count=len(x), mean=float(x.mean()), std_population=float(x.std()))
    result.update({('min' if q == 0 else 'max' if q == 100 else 'median' if q == 50 else f'p{q}'):
                   float(np.percentile(x, q)) for q in QUANTILES})
    return result


def compare(values, indices):
    result = {}
    for key, x in values.items():
        parent, subset = statistics(x), statistics(x[indices])
        support = np.unique(x)
        full_cdf = np.searchsorted(np.sort(x), support, side='right') / len(x)
        sub_cdf = np.searchsorted(np.sort(x[indices]), support, side='right') / len(indices)
        result[key] = dict(parent=parent, subset=subset, CDF_max_absolute_difference=float(np.max(abs(full_cdf-sub_cdf))),
                          mean_relative_difference=subset['mean']/parent['mean']-1)
    return result


def main():
    preserve()
    freeze = ROOT/'state/dataset_main100_freeze.json'
    assert not freeze.exists(), 'Main100 is already frozen. Do not select it again.'
    parent_path = ROOT/'data/main_manifest_template_v2.json'
    raw_path = ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    parent_hash, raw_hash = digest(parent_path), digest(raw_path)
    parent = json.loads(parent_path.read_text())
    rows = parent['requests']
    outputs = {r['request_id']: r for r in map(json.loads, raw_path.read_text().splitlines())}
    assert len(rows) == len(outputs) == 400
    assert all(outputs[r['request_id']]['status'] == 'MEASURED_ONLINE' and
               outputs[r['request_id']]['prompt_SHA256'] == r['prompt_SHA256'] for r in rows)
    plan = dict(time=time.time(), parent_manifest=str(parent_path), parent_SHA256=parent_hash,
                reference_outputs=str(raw_path), reference_outputs_SHA256=raw_hash,
                target=COUNT, seed=SEED, starts=STARTS, iterations_per_start=ITERATIONS,
                procedure='Source/split rounded quotas, then deterministic feature-balanced subset selection.',
                criteria='Input, visual, text, special, image/page count, measured Natural output length, and input/output quartiles.',
                prohibited_inputs=['method latency', 'method throughput', 'SLO outcomes', 'answer accuracy'],
                rare_tail_policy='Do not force the min/max or the single 2048-token cap hit. Report all tail changes.',
                split_policy='Inherit original discovery/heldout ownership. Keep 50/50. Keep Calibration32 unchanged.',
                performance_selection=False, output_length_selection=True,
                caveat='Length-balanced sampling changes inclusion probabilities. It does not reproduce all Main400 statistics exactly.')
    write(ROOT/'configs/main100_selection_plan.json', plan)
    selected, selection, values = select(rows, outputs, set(parent['discovery_ids']))
    subset = [rows[i] for i in selected]
    ids = {r['request_id'] for r in subset}
    manifest = dict(parent)
    manifest.update(name='MLLM-PrefillHeavy100-v1', requests=subset, target_count=100, actual_count=100,
                    parent_manifest=str(parent_path), parent_SHA256=parent_hash,
                    discovery_ids=[i for i in parent['discovery_ids'] if i in ids],
                    heldout_ids=[i for i in parent['heldout_ids'] if i in ids],
                    context_components_parent=parent['context_components'], context_components=None,
                    selection_basis=plan['criteria'], selection_seed=SEED,
                    selection_plan_SHA256=digest(ROOT/'configs/main100_selection_plan.json'),
                    natural_output_status='REFERENCE_LENGTHS_AVAILABLE_IN_SEPARATE_PROFILE',
                    output_length_selection=True, performance_selection=False)
    from .dataset_build import keys
    a = set().union(*(keys(r) for r in subset if r['request_id'] in manifest['discovery_ids']))
    b = set().union(*(keys(r) for r in subset if r['request_id'] in manifest['heldout_ids']))
    calibration = json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    c = set().union(*(keys(r) for r in calibration))
    assert not a & b and not (a | b) & c
    assert len(manifest['discovery_ids']) == len(manifest['heldout_ids']) == 50
    manifest_path = ROOT/'data/main100_manifest_template_v2.json'
    write(manifest_path, manifest)
    comparison = compare(values, selected)
    profile = dict(time=time.time(), evidence='PROCESSOR_MEASUREMENT plus reused official single-GPU Natural output lengths',
                   performance_status='NOT_RUN', task_quality_status='NOT_MEASURED',
                   main100_manifest=str(manifest_path), main100_SHA256=digest(manifest_path),
                   parent_SHA256=parent_hash, reference_outputs_SHA256=raw_hash,
                   source_counts=Counter(r['source'] for r in subset), comparison=comparison,
                   discovery_count=50, heldout_count=50, calibration_count=32, image_document_disjoint=True,
                   parent_cap_hits=sum(bool(r.get('output_cap_hit')) for r in outputs.values()),
                   subset_cap_hits=sum(bool(outputs[r['request_id']].get('output_cap_hit')) for r in subset),
                   selected_request_ids=[r['request_id'] for r in subset], selection=selection,
                   total_input_tokens=sum(r['total_input_tokens'] for r in subset),
                   total_visual_tokens=sum(r['visual_tokens'] for r in subset),
                   total_natural_output_tokens=sum(outputs[r['request_id']]['output_tokens'] for r in subset),
                   total_image_references=sum(r['image_count'] for r in subset))
    write(ROOT/'analysis/main100_profile_comparison.json', profile)
    with (ROOT/'analysis/main100_profile_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'statistic', 'Main400', 'Main100', 'difference'])
        writer.writeheader()
        for key, v in comparison.items():
            for stat in v['parent']:
                writer.writerow(dict(metric=key, statistic=stat, Main400=v['parent'][stat],
                                     Main100=v['subset'][stat], difference=v['subset'][stat]-v['parent'][stat]))
    with (ROOT/'analysis/main100_request_profile.csv').open('w', newline='') as f:
        columns = ['request_id', 'source', 'split', *METRICS, 'output_cap_hit', 'prompt_SHA256']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in selected:
            r = rows[i]
            record = {k: r.get(k) for k in columns}
            record.update(split='discovery' if r['request_id'] in manifest['discovery_ids'] else 'heldout',
                          natural_output_tokens=outputs[r['request_id']]['output_tokens'],
                          output_cap_hit=outputs[r['request_id']].get('output_cap_hit'))
            writer.writerow(record)
    active = dict(time=time.time(), dataset='MLLM-PrefillHeavy100-v1', main_manifest=str(manifest_path),
                  main_SHA256=digest(manifest_path), count=100, discovery_count=50, heldout_count=50,
                  calibration_manifest=str(ROOT/'data/calibration_manifest_template_v2.json'),
                  supersedes_future_main400_runs=True, archived_main400_unchanged=True,
                  selection_plan_SHA256=digest(ROOT/'configs/main100_selection_plan.json'))
    assert digest(parent_path) == parent_hash and digest(raw_path) == raw_hash
    write(freeze, dict(active, status='FROZEN', profile_SHA256=digest(ROOT/'analysis/main100_profile_comparison.json')))
    write(ROOT/'configs/active_workload.json', active)
    figures(values, selected)
    report(profile, manifest_path)
    preserve()
    print(json.dumps({k: profile[k] for k in ['main100_SHA256', 'source_counts', 'subset_cap_hits',
                                            'total_input_tokens', 'total_natural_output_tokens', 'total_image_references']}))


def figures(values, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key in zip(axes, ['total_input_tokens', 'natural_output_tokens', 'image_count']):
        for label, x in [('Main400', values[key]), ('Main100', values[key][selected])]:
            x = np.sort(x)
            ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', label=label)
        ax.set_xlabel(key)
        ax.set_ylabel('Empirical CDF')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Workload profile comparison | Not online EPD performance')
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(ROOT/'figures'/f'main100_profile_comparison.{extension}', dpi=160)
    plt.close(fig)


def report(profile, manifest_path):
    lines = ['# MLLM100 Representative Workload Contract', '',
             '기존 Main400과 Raw Evidence는 보존했다. 이후 주 실험은 Main100을 사용한다.',
             '원본 400개 안에서 실제 요청 100개를 선정했다. 이미지·질문·정답·Prompt는 수정하지 않았다.',
             '성능·SLO·정확도는 선정 입력으로 사용하지 않았다. 이미 측정한 자연 출력 길이는 선정에 사용했다.',
             '따라서 출력 길이를 보기 전의 무작위 표본은 아니다. 전체 Production Traffic의 대표 표본도 아니다.', '',
             f'Manifest: `{manifest_path}`', f'SHA256: `{profile["main100_SHA256"]}`',
             f'Parent Main400 SHA256: `{profile["parent_SHA256"]}`', f'Selection Seed: `{SEED}`', '',
             '## 요청 구성', '', '| Source | Main400 | Main100 |', '|---|---:|---:|']
    parent = json.loads((ROOT/'data/main_manifest_template_v2.json').read_text())
    counts = Counter(r['source'] for r in parent['requests'])
    for source, count in sorted(profile['source_counts'].items()):
        lines.append(f'| {source} | {counts[source]} | {count} |')
    lines += ['', 'Discovery50과 Heldout50은 기존 Split 소속을 유지한다. Calibration32도 그대로 유지한다.',
              '세 집합 사이의 이미지·문서 중복 검사를 통과했다. 요청 ID는 100개 모두 고유하다.', '',
              '## 통계 비교', '', '| Metric | Statistic | Main400 | Main100 |', '|---|---|---:|---:|']
    for key in METRICS:
        v = profile['comparison'][key]
        for stat in ['mean', 'median', 'p95', 'min', 'max']:
            lines.append(f'| {key} | {stat} | {v["parent"][stat]:.3f} | {v["subset"][stat]:.3f} |')
    lines += ['', '전체 Percentile·표준편차·CDF 차이는 대응 CSV와 JSON에 보존했다.',
              '400개와 100개의 통계량이 모두 정확히 같지는 않다. 드문 긴 출력과 최댓값은 특히 달라질 수 있다.',
              '선정용 page_count 통계는 페이지 정보가 없는 요청을 0으로 취급했다. 분모는 각각 전체 400개와 100개다.',
              '이 값은 문서 요청만의 평균 페이지 수가 아니다. 원본 요청의 빈 페이지 정보는 수정하지 않았다.',
              f'Output Cap Hit는 Main400에서 {profile["parent_cap_hits"]}/400개, Main100에서 {profile["subset_cap_hits"]}/100개다.',
              '상한 도달 요청을 잘라내거나 EOS를 수정하지 않았다. 표본에 포함되지 않은 요청도 Main400에 남아 있다.',
              f'선택한 요청의 Input Token 합계는 {profile["total_input_tokens"]:,}개다.',
              f'자연 출력 Token 합계는 {profile["total_natural_output_tokens"]:,}개다.',
              f'Image Reference 합계는 {profile["total_image_references"]:,}개다.', '',
              '## 실행 계약과 한계', '',
              'Model·Processor Revision과 생성 계약은 Main400과 같다. Natural EOS 상한은 2,048 Tokens다.',
              '현재 표는 Processor 측정과 공식 단일 GPU의 기존 출력 길이를 재사용했다.',
              'Main100의 EPD 성능은 아직 미실행이다. Task Quality도 아직 미확정이다.',
              '같은 QPS에서 도착 구간은 대략 짧아진다. 초기화·Compile·Warm-up·Drain 시간은 같은 비율로 줄지 않는다.',
              '모든 방법에 같은 Main100·요청 순서·Arrival Trace를 적용한다. Main400 수치와 섞어 평균을 계산하지 않는다.',
              '대응 표는 `analysis/main100_profile_comparison.csv`다.',
              '요청별 표는 `analysis/main100_request_profile.csv`다.',
              '선정 기준은 `configs/main100_selection_plan.json`에 실험 전 고정했다.',
              '향후 Client 기본 Manifest는 `configs/active_workload.json`에서 읽는다.', '']
    (ROOT.parent/'mllm_prefill_heavy100_dataset_contract.md').write_text('\n'.join(lines))


def verify(regenerate=False):
    preserve()
    frozen=json.loads((ROOT/'state/dataset_main100_freeze.json').read_text())
    manifest_path=Path(frozen['main_manifest'])
    assert digest(manifest_path)==frozen['main_SHA256']
    manifest=json.loads(manifest_path.read_text())
    parent_path=Path(manifest['parent_manifest'])
    assert digest(parent_path)==manifest['parent_SHA256']
    parent=json.loads(parent_path.read_text())
    lookup={r['request_id']:r for r in parent['requests']}
    assert len(manifest['requests'])==100
    assert len({r['request_id'] for r in manifest['requests']})==100
    assert all(row==lookup[row['request_id']] for row in manifest['requests'])
    assert set(manifest['discovery_ids'])<=set(parent['discovery_ids'])
    assert set(manifest['heldout_ids'])<=set(parent['heldout_ids'])
    profile_path=ROOT/'analysis/main100_profile_comparison.json'
    assert digest(profile_path)==frozen['profile_SHA256']
    profile=json.loads(profile_path.read_text())
    plan_path=ROOT/'configs/main100_selection_plan.json'
    assert digest(plan_path)==frozen['selection_plan_SHA256']
    plan=json.loads(plan_path.read_text())
    raw_path=Path(plan['reference_outputs'])
    assert digest(raw_path)==plan['reference_outputs_SHA256']
    outputs={r['request_id']:r for r in map(json.loads,raw_path.read_text().splitlines())}
    ids=set(profile['selected_request_ids'])
    selected=[i for i,r in enumerate(parent['requests']) if r['request_id'] in ids]
    _,_,values=feature_matrix(parent['requests'],outputs)
    assert compare(values,selected)==profile['comparison']
    if regenerate:
        figures(values,selected)
        report(profile,manifest_path)
    proof=dict(time=time.time(),status='PASS',requests=100,unchanged_original_rows=100,
               parent_manifest_SHA256=digest(parent_path),main100_manifest_SHA256=digest(manifest_path),
               reference_raw_SHA256=digest(raw_path),profile_recalculation='EXACT_MATCH',
               source_SHA256=digest(Path(__file__)),regenerated_figure_and_report=regenerate,
               active_default_manifest=str(manifest_path),GPU_generation_repeated=False)
    write(ROOT/'state/main100_subset_verification.json',proof)
    preserve()
    print(json.dumps(proof))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--regenerate-report',action='store_true')
    args=parser.parse_args()
    if args.verify or args.regenerate_report:verify(args.regenerate_report)
    else:main()
