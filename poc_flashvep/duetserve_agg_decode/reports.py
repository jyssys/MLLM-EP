"""Render Korean reports from completed starts without promoting diagnostics."""
import collections
import json
import statistics
import time

from .common import BASE, BRANCH, REPO, ROOT, preserve, write
from .publication import DOCUMENTS
from .statistics import collect, paired_ratio

PAPER = 'https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf'
LARGE = 'DP2_B8192_S1024_FULL'
SMALL = 'DP2_B512_S16_FULL'
NATIVE = 'PD_G1_B32000_S16_FULL_DECODE_ONLY'
LABELS = {LARGE: 'Agg-DP2 B8192/S1024', SMALL: 'Agg-DP2 B512/S16', NATIVE: 'Native PD 1P1D'}


def load(relative):
    return json.loads((ROOT / relative).read_text())


def fmt(value, digits=3):
    return 'NOT_MEASURED' if value is None else f'{value:.{digits}f}'


def ratio(value):
    ci = value['CI95']
    return fmt(value['ratio']) + (' [' + ', '.join(fmt(x) for x in ci) + ']' if ci else ' [CI NOT_ESTABLISHED]')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)]) + '\n'


def start_rows(family, cap=200):
    result = []
    for path in sorted((ROOT / 'raw').glob('*/metrics.json')):
        m = json.loads(path.read_text())
        c = load('configs/' + path.parent.name + '.json')
        if family == 'FRESH_CONFIRMATION' and (ROOT / 'state/user_primary_scope_revision_v2.json').exists() \
                and c['arm'] == 'DP2' and c['token_budget'] == 512:
            continue
        expected = ('confirm_' if family == 'FRESH_CONFIRMATION' else
                    'tp_diagnostic_' if family == 'TP2_DIAGNOSTIC' else 'secondary_')
        if path.parent.name.startswith(expected) and c['cap'] == cap and not c.get('correctness_only'):
            assert m['complete'] and m['output_gate']['passed']
            result.append((c, m))
    return result


def performance_table(aggregate, family='FRESH_CONFIRMATION', qps=None, cap=200):
    rows = []
    for value in aggregate:
        if value['family'] != family or value['output_cap'] != cap:
            continue
        if qps is not None and value['offered_QPS'] not in qps:
            continue
        m = value['metrics']
        rows.append([LABELS.get(value['arm'], value['arm']), value['offered_QPS'],
            m['mean_TBT_ms']['n_starts'], fmt(m['mean_TTFT_ms']['mean'] / 1000),
            fmt(m['mean_TBT_ms']['mean']), fmt(m['ITL_p99_ms']['mean']),
            fmt(m['TPOT_p95_ms']['mean']), fmt(m['output_tokens_s']['mean'])])
    return table(['Method', 'Offered QPS', 'Start 수', 'Mean TTFT (s)', 'Mean TBT (ms)',
                  'ITL p99 (ms)', 'TPOT p95 (ms)', 'Output Token/s'], rows)


def main():
    preserve()
    revised = (ROOT / 'state/user_primary_scope_revision_v2.json').exists()
    if revised:
        assert load('state/primary_confirmation_complete_v2.json')['status'] == 'COMPLETE'
        assert load('state/primary_long_windows_complete_v2.json')['status'] == 'COMPLETE'
        assert load('state/primary_secondary_complete_v2.json')['status'] == 'COMPLETE'
    else:
        assert load('state/fresh_confirmation_complete.json')['status'] == 'COMPLETE'
        assert load('state/long_replication_complete.json')['passed']
        assert load('state/secondary_plan_result.json')['returncode'] == 0
    assert load('state/tp_diagnostic_characterization_complete.json')['status'] == 'COMPLETE'
    aggregate, ratios = collect()
    primary = [r for r in ratios if r['family'] == 'FRESH_CONFIRMATION' and r['output_cap'] == 200]
    assert {r['arm'] for r in primary} == ({LARGE, NATIVE} if revised else {LARGE, SMALL, NATIVE})
    assert all(len(r['paired_seeds']) == 3 for r in primary)
    tests = next(r for r in primary if r['arm'] == LARGE)['tests']
    discovery_low = load('raw/concurrency_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111611_v1/metrics.json')
    discovery_high = load('raw/concurrency_DP2_B8192_S1024_FULL_N128_O200_q10_s2026111611_v1/metrics.json')
    discovery_TBT_ratio = discovery_high['mean_TBT_ms'] / discovery_low['mean_TBT_ms']
    if tests['mean_TBT_ms']['CI95'][0] > 1:
        verdict = 'QPS_DEPENDENT_MEAN_TBT_DEGRADATION_REPRODUCED'
    else:
        verdict = 'NO_CONFIRMED_QPS_DEPENDENT_MEAN_TBT_DEGRADATION'
    ratio_table = table(['Method', 'Metric', 'QPS10 / QPS2 [95% CI]', '2배 판정', 'Spearman', '인접 증가/8'],
        [[LABELS[row['arm']], metric, ratio(value), value['repeat_qualified_2x'],
          fmt(value['spearman_QPS_metric']), f"{value['adjacent_increases']}/{value['adjacent_pairs']}"]
         for row in primary for metric, value in row['tests'].items()])
    gates = table(['Metric', 'QPS10 / QPS2 [95% CI]', '2배 판정'],
        [[metric, ratio(value), value['repeat_qualified_2x']] for metric, value in tests.items()])
    large_starts = start_rows('FRESH_CONFIRMATION')
    large_points = collections.defaultdict(dict)
    for cfg, metric in large_starts:
        if cfg['arm'] == 'DP2':
            large_points[cfg['arrival_rate']][cfg['arrival_seed']] = metric
    per_load = {}
    for metric in tests:
        values = []
        for qps in sorted(large_points):
            seeds = sorted(set(large_points[2]) & set(large_points[qps]))
            value = paired_ratio([large_points[qps][s][metric] for s in seeds],
                                 [large_points[2][s][metric] for s in seeds])
            values.append(dict(QPS=qps, **value))
        per_load[metric] = dict(points=values,
            maximum=max(values, key=lambda value: value['ratio']),
            first_point_2x=next((v['QPS'] for v in values if v['ratio'] >= 2), None),
            first_CI_lower_2x=next((v['QPS'] for v in values if v['CI95'] and v['CI95'][0] >= 2), None))
    write(ROOT / 'analysis/fresh_primary_per_QPS_load_tests.json', dict(evidence='MEASURED', tests=per_load,
        scope='Paired independent-start ratios against the same configuration at QPS2. The maximum and first crossing describe the tested grid. These are exploratory pointwise intervals without multiple-comparison correction.'))
    crossing_table = table(['Metric', '최대 비율의 QPS', '최대 / QPS2 [95% CI]', '점 추정 2배 최초 QPS', 'CI 하한 2배 최초 QPS'],
        [[metric, value['maximum']['QPS'], ratio(value['maximum']),
          value['first_point_2x'] if value['first_point_2x'] is not None else 'NOT_OBSERVED',
          value['first_CI_lower_2x'] if value['first_CI_lower_2x'] is not None else 'NOT_ESTABLISHED']
         for metric, value in per_load.items()])
    per_seed = table(['Method', 'Arrival Seed', 'QPS', '실현 Arrival QPS', 'Mean TTFT (s)', 'Mean TBT (ms)', 'ITL p99 (ms)', 'TPOT p95 (ms)', '최대 ITL (ms)', 'Output Token/s'],
        [[LABELS.get(c['arm'] + f"_B{c['token_budget']}_S{c['D_slots']}_{c['ar_cudagraph_mode']}", c['arm']),
          c['arrival_seed'], c['arrival_rate'], fmt(m['realized_arrival_QPS']), fmt(m['mean_TTFT_ms'] / 1000),
          fmt(m['mean_TBT_ms']), fmt(m['ITL_p99_ms']), fmt(m['TPOT_p95_ms']), fmt(m['maximum_ITL_ms']),
          fmt(m['output_tokens_s'])] for c, m in start_rows('FRESH_CONFIRMATION')])
    slo_rows = []
    grouped = collections.defaultdict(list)
    geometries = collections.defaultdict(list)
    for c, m in start_rows('FRESH_CONFIRMATION'):
        arm = c['arm'] + f"_B{c['token_budget']}_S{c['D_slots']}_{c['ar_cudagraph_mode']}"
        geometries[(arm, c['arrival_rate'])].append(m)
        for regime, value in m['SLO_regimes'].items():
            grouped[(arm, c['arrival_rate'], regime)].append(value)
    for (arm, qps, regime), values in sorted(grouped.items()):
        slo_rows.append([LABELS[arm], qps, regime,
            fmt(100 * statistics.mean(v['Joint_attainment'] for v in values)),
            fmt(statistics.mean(v['Output_token_SLO_goodput'] for v in values)),
            fmt(statistics.mean(v['ITL_threshold_exceedances'] for v in values))])
    slo_table = table(['Method', 'QPS', 'Frozen SLO', 'Joint Pass (%)', 'SLO-qualified Output Token/s', 'ITL 초과 횟수/Start'], slo_rows)
    geometry_table = table(['Method', 'QPS', 'Prefill Chunk Mean / p95', 'Decode Batch p95', 'Mixed Graph Replay Counter', 'First / Steady ITL p99 (ms)'],
        [[LABELS[arm], qps,
          fmt(statistics.mean(m['actual_prefill_chunk_tokens']['mean'] for m in ms)) + ' / ' +
          fmt(statistics.mean(m['actual_prefill_chunk_tokens']['p95'] for m in ms)),
          fmt(statistics.mean(m['decode_batch_p95'] for m in ms)),
          fmt(statistics.mean(m['mixed_fullgraph_replays'] for m in ms)),
          fmt(statistics.mean(m['first_ITL_p99_ms'] for m in ms)) + ' / ' +
          fmt(statistics.mean(m['steady_ITL_p99_ms'] for m in ms))]
         for (arm, qps), ms in sorted(geometries.items())])
    fixed = [load(str(p.relative_to(ROOT))) for p in sorted((ROOT / 'analysis').glob('long_*_fixed_window.json'))]
    assert len(fixed) == (8 if revised else 10)
    long_table = table(['Run', '측정 초', '측정 도착/완료', 'Arrival-cohort Mean TBT (ms)', 'Active-window Mean TBT (ms)', '완료 Output Token/s', '실제 방출 Token/s', 'Outstanding 증가'],
        [[r['name'], r['measurement_seconds'], f"{r['measurement_arrivals']}/{r['measurement_completions']}",
          fmt(r['mean_TBT_ms']), fmt(r['active_window_observed_mean_TBT_ms']),
          fmt(r['raw_completed_output_tokens_s']), fmt(r['actually_emitted_output_tokens_s']),
          r['stability']['outstanding_delta']] for r in fixed])
    long_ratios = {}
    for metric in ('mean_TBT_ms', 'active_window_observed_mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms'):
        low, high = {}, {}
        for r in fixed:
            c = load('configs/' + r['name'] + '.json')
            if not r['name'].startswith('long_DP2_B8192_S1024_'):
                continue
            (low if c['arrival_rate'] == 2 else high)[c['arrival_seed']] = r[metric]
        seeds = sorted(set(low) & set(high))
        assert len(seeds) == 3
        long_ratios[metric] = paired_ratio([high[s] for s in seeds], [low[s] for s in seeds])
    write(ROOT / 'analysis/long_window_paired_load_ratios.json', dict(evidence='MEASURED', ratios=long_ratios,
        scope='Three independent large-budget server starts per QPS. QPS6/QPS2. Active-window intervals and fully followed arrival-cohort intervals remain separate. Queue growth prevents a sustainability claim.'))
    long_ratio_table = table(['600초 Metric', 'QPS6 / QPS2 [95% CI]'], [[k, ratio(v)] for k, v in long_ratios.items()])
    budget_rows = []
    for value in aggregate:
        if value['family'] != 'BUDGET_CONTROL':
            continue
        m = value['metrics']
        budget_rows.append([value['offered_QPS'], value['arm'], fmt(m['mean_TTFT_ms']['mean'] / 1000),
            fmt(m['mean_TBT_ms']['mean']), fmt(m['ITL_p99_ms']['mean']), fmt(m['TPOT_p95_ms']['mean']),
            fmt(m['output_tokens_s']['mean'])])
    budget_table = table(['QPS', '설정', 'Mean TTFT (s)', 'Mean TBT (ms)', 'ITL p99 (ms)', 'TPOT p95 (ms)', 'Output Token/s'], budget_rows)
    concurrency_rows = []
    for slots in (16, 32, 1024):
        for qps in (2, 6, 10):
            prefix = 'sweep' if slots == 16 else 'concurrency'
            value = load(f'raw/{prefix}_DP2_B8192_S{slots}_FULL_N128_O200_q{qps}_s2026111611_v1/metrics.json')
            concurrency_rows.append([slots, qps, fmt(value['mean_TBT_ms']), fmt(value['ITL_p99_ms']),
                fmt(value['TPOT_p95_ms']), fmt(value['output_tokens_s']), value['decode_batch_p95']])
    concurrency_table = table(['상한 S', 'QPS', 'Mean TBT (ms)', 'ITL p99 (ms)', 'TPOT p95 (ms)',
                               'Output Token/s', '실제 Decode Batch p95'], concurrency_rows)
    work_rows = []
    for cfg, _ in start_rows('FRESH_CONFIRMATION'):
        if cfg['arrival_rate'] not in (2, 6, 10):
            continue
        value = load('analysis/' + cfg['name'] + '_scheduled_work.json')
        work_rows.append([cfg['name'], value['expected_prompt_rows'], value['scheduled_prompt_rows'],
                          value['duplicate_scheduled_prompt_rows'], fmt(100 * value['duplicate_fraction_of_expected'])])
    for run in fixed:
        value = load('analysis/' + run['name'] + '_scheduled_work.json')
        work_rows.append([run['name'], value['expected_prompt_rows'], value['scheduled_prompt_rows'],
                          value['duplicate_scheduled_prompt_rows'], fmt(100 * value['duplicate_fraction_of_expected'])])
    work_table = table(['Run', '고유 Prompt Rows 분모', 'Scheduled Prompt Rows', '중복 Scheduled Rows',
                        '분모 대비 중복 비율 (%)'], work_rows)
    tail_rows = []
    for cfg, _ in start_rows('FRESH_CONFIRMATION'):
        if cfg['arm'] == 'DP2' and cfg['arrival_rate'] == 6:
            phase = load('analysis/' + cfg['name'] + '_cohort_phases.json')
            value = phase['groups']['NO_NEW_PROMPT_TAIL']
            tail_rows.append([cfg['name'], value['intervals'], fmt(100 * value['interval_fraction']),
                              fmt(value['mean_TBT_ms'])])
    for run in fixed:
        if run['name'].startswith('long_DP2_B8192_S1024_') and '_q6.0_' in run['name']:
            phase = load('analysis/' + run['name'] + '_cohort_phases.json')
            value = phase['groups']['NO_NEW_PROMPT_TAIL']
            tail_rows.append([run['name'], value['intervals'], fmt(100 * value['interval_fraction']),
                              fmt(value['mean_TBT_ms'])])
    tail_table = table(['QPS6 실행', '새 Prompt 종료 후 Interval 수', '전체 Interval 대비 (%)',
                        '이 구간의 Mean TBT (ms)'], tail_rows)
    coverage_rows = []
    for qps in (4, 8):
        for maximum, name in ((256, f'budget_DP2_B512_S128_FULL_N128_O200_q{qps}_s2026111614_v1'),
            (512, f'coverage_DP2_B512_S128_FULL_CAP512_N128_O200_q{qps}_s2026111614_v1')):
            v = load('raw/' + name + '/metrics.json')
            coverage_rows.append([qps, maximum, fmt(v['mean_TBT_ms']), fmt(v['ITL_p99_ms']),
                                  fmt(v['TPOT_p95_ms']), fmt(v['output_tokens_s']), v['mixed_fullgraph_replays']])
    coverage_table = table(['QPS', 'Capture 최대 Token 수', 'Mean TBT (ms)', 'ITL p99 (ms)', 'TPOT p95 (ms)',
                            'Output Token/s', 'Mixed Full-graph Replay Counter'], coverage_rows)
    # Use only descriptive, matched-input interval accounting for this split.
    decomposition = load('analysis/large_budget_QPS_mixture_decomposition.json')
    mixture_table = table(['측정 평균 변화의 항등식', '변화 (ms)', '평균 변화 중 비율 (%)'],
        [[k, fmt(v), fmt(100 * decomposition['fractions'][k])] for k, v in decomposition['symmetric_mixture_identity'].items()])
    profile_rows = []
    for path in sorted((ROOT / 'analysis').glob('*_interference_gate.json')):
        r = json.loads(path.read_text())
        for label, value in r.get('comparisons', {}).items():
            errors = value.get('whole_run_relative_changes', {})
            profile_rows.append([r['prefix'], label, value.get('qualified', False),
                fmt(100 * errors.get('mean_TBT_ms', 0)) if errors else 'NOT_RUN_OR_FAILED',
                fmt(100 * errors.get('TTFT_p95_ms', 0)) if errors else 'NOT_RUN_OR_FAILED',
                fmt(100 * errors.get('TPOT_p95_ms', 0)) if errors else 'NOT_RUN_OR_FAILED'])
    profile_table = table(['Run family', '계측', '5% Gate', 'Mean TBT 변화 (%)', 'TTFT p95 변화 (%)', 'TPOT p95 변화 (%)'], profile_rows)
    kernels = [json.loads(p.read_text()) for p in sorted((ROOT / 'analysis').glob('*_kernel_union.json'))]
    kernel_table = table(['Run', 'Role/GPU', '구간 (s)', '기록 Kernel Union Busy (%)', 'Clean 판정'],
        [[r['name'], f"{r.get('role', 'replica0')}/GPU0", fmt(r['diagnostic_values']['window_seconds']),
          fmt(r['diagnostic_values']['kernel_union_busy_percent']), r['clean_physical_kernel_measurement']] for r in kernels])
    api28_rows = []
    for path in sorted((ROOT / 'analysis').glob('api28_*_serving_gate.json')):
        result = json.loads(path.read_text())
        for role, window in result['role_windows'].items():
            api28_rows.append([result['method'], role, result['status'], fmt(window['window_seconds']),
                fmt(window['busy_percent']), fmt(window['idle_percent']), window['idle_window_count'],
                fmt(window['idle_p95_ms']), fmt(window['idle_max_ms']), window['counters']['invalid']])
    api28_table = table(['API28 실행', 'Role', 'Clean Gate', '측정 초', '기록 Busy (%)', '기록 Idle (%)',
                         'Idle 구간 수', 'Idle p95 (ms)', 'Idle Max (ms)', 'Invalid 기록 수'], api28_rows)
    api28_gate = ROOT / 'analysis/api28_Native_serving_gate.json'
    from .tail_outliers import analyze as rare_ITL
    rare_ITL()
    api28_control_note = ''
    if api28_gate.exists():
        check = json.loads(api28_gate.read_text())
        if 'whole_run_control_A2_A1_relative_change' in check:
            api28_control_note = ('Native A/A의 Output Token/s 차이는 ' +
                fmt(100 * check['whole_run_control_A2_A1_relative_change']['output_tokens_s']) +
                '%다.\n계측 실행의 처리량 변화는 두 Control 평균 대비 ' +
                fmt(100 * check['whole_run_relative_change']['output_tokens_s']) +
                '%다.\nControl 안정성과 계측 실행의 오차를 별도로 판정했다.\n')
    qualification = load('analysis/fresh_reference_gate.json')
    assert qualification['passed']
    from .native_ttft_attribution import analyze as native_TTFT
    native_host = native_TTFT()
    native_TTFT_table = table(['QPS', 'Mean TTFT (s)', 'Arrival→P-ready (s)', 'P-ready→D Socket (s)',
                              'D Socket→First Output (s)', '중간 대기의 TTFT 비율 (%)'],
        [[v['QPS'], fmt(v['metrics']['mean_TTFT_ms']['mean'] / 1000),
          fmt(v['metrics']['arrival_to_source_ready_ms']['mean'] / 1000),
          fmt(v['metrics']['source_ready_to_D_socket_end_ms']['mean'] / 1000),
          fmt(v['metrics']['D_socket_end_to_first_output_ms']['mean'] / 1000),
          fmt(100 * v['metrics']['D_host_wait_fraction_of_TTFT']['mean'])]
         for v in native_host['QPS_summary']])
    environment = load('state/runtime_versions.json')
    docs = {}
    docs['duetserve_agg_dp2_tp2_tbt_crossover_final_summary.md'] = f'''# DuetServe Figure 2 및 Aggregated Decode 간섭 최종 요약

## 핵심 판정

판정은 `{verdict}`다.
동일한 B8192/S1024에서 QPS만 변경한 결과다.
모든 Decode 지표가 2배 증가했다고 묶어서 표현하지 않는다.
지표별 판정은 다음과 같다.

{gates}
{crossing_table}
최초 교차와 최대값은 검사한 QPS Grid의 탐색적 요약이다.
여러 QPS 비교에 대한 동시 신뢰구간으로 해석하지 않는다.
기존 Discovery 단일 Start의 Mean TBT 비율은 {fmt(discovery_TBT_ratio)}배였다.
그 값은2배 미만이며 기존 결과를 그대로 보존했다.
따라서2배 경계를 모든 Prompt Pool과 Arrival에서 보장한다고 주장하지 않는다.
이 결과는 Qwen3-8B와 현재 Runtime에 한정한다.
B8192/S1024는 논문 재현용 기준선이다.
B는 `max_num_batched_tokens`다.
이는 Replica별 한 Iteration에서 Prefill과 Decode를 합한 Token 상한이다.
두 GPU 전체를 합친 상한이 아니다.
S는 `max_num_seqs`이며 동시 Sequence 상한이다.
QPS나 실제 Decode Batch 크기와 같지 않다.
사용자 지시에 따라 Primary 비교를 B8192/S1024와 Native로 한정했다.
B512/S16의 추가 실행은 중단했다.
이미 완료한 작은 Budget 결과는 보충 자료로 보존했다.
이는 모든 튜닝된 Colocated 설정보다 우수하다는 검증이 아니다.
이번 측정으로 Dynamic Layer PD의 우위를 주장하지 않는다.

## 환경과 논문 대응

Model은 `Qwen/Qwen3-8B`다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Original BF16과 Exact Attention을 유지했다.
Greedy와 `ignore_eos=True`를 사용했다.
Thinking-disabled 생성 Suffix를 Token ID로 확인했다.
36개 Layer를 유지했다.
GPU0/1의 H100 SXM 80GB 두 대를 사용했다.
GPU당 KV Budget은 42 GiB다.
Prefix Caching은 비활성이다.
Compile Mode는0이다.
Aggregated는 FULL Mixed CUDA Graph를 사용한다.
실제 Capture 범위를 벗어난 Prefill Shape는 Eager 경로를 사용할 수 있다.
Batch-invariant 실행과 기존 Numerical Correction을 모든 적격 비교에서 유지했다.
vLLM Package Version은 `{environment['vllm']}`다.
PyTorch는 `{environment['torch']}`다.
Torch CUDA는 `{environment['torch_CUDA']}`다.
NIXL은 `{environment['nixl']}`다.
GPU UUID와 실제 구성은 Evidence의 `state/environment.json`에 있다.

논문 Figure 2는 DP2다.
Figure 2는 Budget8192, Round-robin과 Input8000/Output200을 명시한다.
공통 평가 설정은 최대 Batch Size1024를 명시한다.
우리는 이를 `max_num_seqs=1024`로 대응했다.
Figure 2의 실제 CLI가 동일하다고 확인한 것은 아니다.
논문의 vLLM 0.10.1과 현재 vLLM은 다르다.
논문의 KV 메모리 비율 0.9와 현재 고정 42 GiB도 다르다.
논문 Figure 7은 Qwen3-14B TP2다.
이번 Qwen3-8B TP2는 Figure 7의 정확한 재현이 아니다.
근거는 [DuetServe 원문]({PAPER})이다.

| 항목 | 논문 | 현재 실행 |
| --- | --- | --- |
| Figure 2 Model | Qwen3-8B, Revision 미기재 | Frozen Revision b968826d |
| GPU | H100 두 대 | GPU0/1 H100 SXM 80GB 두 대 |
| vLLM | 0.10.1–v1 | {environment['vllm']} |
| PyTorch / CUDA | 2.8.0 / 13.0 | {environment['torch']} / {environment['torch_CUDA']} |
| Driver | 580.95.05 | 570.211.01 |
| KV 설정 | Memory Utilization 0.9 | GPU당 KV 42 GiB 고정 |
| Figure 2 Budget / Router | 8192 / Round-robin | 8192 / Round-robin |
| 최대 Sequence | 공통 평가 Batch1024, Figure 2 CLI 미확인 | S1024 명시 |
| Workload | 8000/200, 공식 Demo 기반 | 실제 문서 Prefix 8000/200 |
| 생성 처리량 | Figure 2 Total Tokens/s | Output-only와 Input+Output를 분리 |

Figure 2의 Total Tokens/s가 입력을 포함하는지는 해당 설명만으로 확정하지 않는다.
우리 주 그래프는 출력 Token만 센다.
Input+Output 처리율도 원시 Metrics에 보존했다.
고정 8000/200에서는 Input+Output가 Output-only의 41배다.
이는 Token 수의 구조적 환산이며 별도의 Speedup 측정이 아니다.

## 측정 설계와 정합성

Discovery는 실제 원문 64개를 N128로 순환했다.
Fresh Confirmation은 이전 튜닝 문서와 분리한 LongBench 원문 128개를 사용했다.
도착 Seed는 2026111622, 2026111623, 2026111624다.
Primary 비교군의 축소는 일부 Confirmation 이후의 사용자 결정이다.
이미 완료한 동일한 B8192/S1024와 Native 실행도 모두 포함했다.
Seed1624는 이 범위 변경 이후에 처음 실행했다.
모든 Arm은 같은 QPS에서 동일한 Prompt와 예정 Arrival Trace를 사용했다.
각 QPS와 Method에 독립 Server Start 3개가 있다.
각 Start는 128개 요청과 요청당 200개 출력 Token을 포함한다.
Fresh Pool SHA256은 `{qualification.get('pool_SHA256', load('state/confirmation_freeze.json')['pool_SHA256'])}`다.
Reference 반복의 128개 출력이 일치했다.
독립 Full-model N32에서 출력, 모든 유효 Prompt KV와 최종 Hidden을 비교했다.
Native N32에서는 실제 D에 Import한 모든 Prompt KV도 비교했다.
N128 성능 실행의 출력은 전체 Reference와 비교했다.
모든 N128의 KV를 직접 Hash 비교했다고 확대하지 않는다.

기존 RMSNorm Phase Correction을 유지했다.
Aggregated는 표준 Continuous Scheduler를 사용했다.
수정 없는 vLLM Binary와 완전히 같다고 표현하지 않는다.
Native는 기존 NIXL 및 Cohort-isolated D Ingress를 사용했다.
이는 논문 Dynamo와 다른 Runtime Contract다.
Native D 상한16과 Aggregated 상한1024의 차이도 Isolation 비율에 영향을 줄 수 있다.
Native의 TTFT에는 Cohort Ingress 대기가 크게 포함될 수 있다.
그 Host 경로의 요청별 분해는 별도 Native 보고서에 제시했다.
따라서 Native의 순수 P Compute Saturation은 별도의 Gate다.
Clean P Kernel Busy와 현재 독립 P Capacity Reference의 공동 검증은 미확정이다.
그 물리적 조건의 판정은 `INCONCLUSIVE_MEASUREMENT`다.
D의 Clean 추가 Compute Capacity는 `NOT_MEASURED`다.
TTFT Queue 증가만으로 두 조건을 PASS로 처리하지 않았다.
추가 API28의 짧은 Kernel 측정은 원인 보고서에서 별도로 판정한다.
짧은 구간이 적격이어도 전체 실행의 Busy나 추가 Offloading Capacity를 뜻하지 않는다.

## 주요 결과

다음 값은 Start별 통계의 평균이다.
ITL p99는 세 Start의 p99 평균이다.
전체 Token을 합친 p99와 다르다.
Output Token/s는 전체 요청을 Drain한 Finite Cohort의 처리율이다.
지속 가능한 최대 처리 용량이 아니다.

{performance_table(aggregate, qps=(2, 6, 10))}
전체 QPS, Seed별 결과와 CI는 [DP2 상세 보고서](aggregated_dp2_qps_degradation_results.md)에 있다.
고정된 QPS에서 Aggregated/Native 비율은 별도 Isolation 지표다.
이 비율을 High-QPS/Low-QPS 악화 비율과 혼동하지 않는다.

## 600초 관측

Warmup60초, 측정600초, 추가 도착90초를 분리했다.
측정 구간의 모든 도착 요청을 최종 완료까지 추적했다.
별도로 측정 시간 안에서 관측한 Token Interval도 계산했다.
두 Latency 분모를 섞지 않았다.
긴 Window의 B8192/S1024는 QPS별 독립 Start 3개다.
Native의 긴 Window는 QPS별 1개다.
B512/S16의 긴 Window는 사용자 지시에 따라 실행하지 않았다.

{long_ratio_table}
Queue가 증가하는 결과는 Overload Diagnostic이다.
600초를 실행했다는 이유만으로 정상상태나 Sustainable Capacity를 주장하지 않는다.
상세 값은 [Native 및 긴 구간 비교](native_pd_decode_isolation_comparison.md)에 있다.

## Output16 및 Output64의 보조 측정

Input8000과 Primary의 B8192/S1024를 유지했다.
Output별로 N32 출력, 전체 Prompt KV와 Hidden 정합성을 먼저 확인했다.
그 뒤 QPS2/6/10에서 Method별 N128을 실행했다.
Arrival Seed는 2026111615다.
기존 Discovery Prompt Pool을 사용했다.
각 점은 단일 Start이며 독립 반복 CI는 없다.
Output200의 Fresh Confirmation과 같은 증거 강도로 해석하지 않는다.

### Output16

{performance_table(aggregate, family='SECONDARY', cap=16)}
### Output64

{performance_table(aggregate, family='SECONDARY', cap=64)}
Output Length가 다른 점들을 하나의 Curve로 연결하지 않았다.
이 측정으로 새로운 Dynamic Offloading 효과를 주장하지 않는다.

## 원인과 한계

큰 Prefill Chunk가 들어간 Mixed Iteration에서 Decode Token 방출이 늦어졌다.
QPS가 증가하면 Mixed Interval의 비중과 Decode Batch 구성도 변했다.
Mean TBT와 ITL p99는 서로 다른 양상을 보였다.
ITL p99는 드문 긴 Interval에 민감하다.
Request TPOT는 요청 안의 199개 Interval을 평균한다.
고정 Output200에서는 Mean TBT와 Mean Request TPOT가 같은 평균을 갖는다.
두 지표의 상위 분위는 같지 않다.
QPS7–10에서는 약3.5–5.1초의 드문 Maximum ITL도 관측했다.
1초 이상인24개 Interval은 이미 Decode 중이던 요청의 Prompt 재예약과 모두 겹쳤다.
이 재계산·대기는 약300ms Mixed Iteration과 별도로 분석했다.
자세한 기록은 [간섭 원인 보고서](aggregated_decode_interference_root_cause.md)에 있다.

Stock Aggregated Scheduler는 Token Budget을 사용한다.
별도의 Decode-safe 시간 Budget이나 Deadline을 집행하지 않는다.
따라서 이번 현상을 시간 예측기의 Budget 초과로 설명하지 않는다.
GPU Kernel은 일반 Scheduler가 즉시 선점할 수 없다고 취급했다.
SM, HBM, Cache, Launch의 배타적 원인 기여도는 `NOT_IDENTIFIED`다.
Clean GPU Kernel Busy와 Idle은 Profiler Gate 결과를 따라 판정했다.
NVML 사용률의 보수값을 Physical Idle로 사용하지 않았다.

TP2는 TP1 Reference 출력과 불일치했다.
Within-TP2 Diagnostic을 주 성능 비교와 분리했다.
자세한 한계는 [TP2 보고서](aggregated_tp2_qps_degradation_results.md)에 있다.
Static/Dynamic Layer 개선은 이번 Characterization에서 `NOT_RUN`이다.
기존 결과를 새 Offloading 성과로 재사용하지 않았다.
적격 TP2 기준선과 Clean D 추가 실행 예산을 확보하지 못했다.
현재 단계의 결과를 새로운 Layer Offloading 순이익으로 확대하지 않았다.
후속 Layer Admission과 Decode 보호는 별도 검증이 필요하다.

## 실패 실행과 복구

초기 DP2 Qualification은 CPU Fixture 누락으로 시작 전에 실패했다.
초기 Native QPS2는 Canonical KV Contract 누락으로 Worker 준비에 실패했다.
두 실패는 GPU 성능 결과로 집계하지 않았다.
의존 파일을 확인한 뒤 새 이름의 실행으로 검증했다.
최초 Nsight 실행은 감시기가 Profiler의 새 Process Session을 외부 작업으로 분류했다.
기존 실패 Receipt를 PASS로 고치지 않았다.
Live Process Ancestry를 확인하는 별도 감시기로 새 실행을 수행했다.
그 실행도 Profiler 성능 간섭 Gate는 통과하지 못했다.
별도 Normalization Calibration에서는 Lease 해제와 CUDA 자식 종료 사이의 경합이 발생했다.
Resource Guard가 다음 Model 실행을 시작 전에 거절했다.
확인되지 않은 GPU Process를 종료하지 않았다.
이후 CPU 부모가 CUDA 자식의 실제 종료와 NVML 해제를 확인할 때까지 Lease를 보유했다.
원시 실패 로그와 재시도 이름을 모두 보존했다.
실패는 `FAILED`이며 알고리즘적 성능 악화와 구분한다.
범위 변경 당시의 B512/S16 실행은 사용자 지시에 따른 취소다.
이 부분 실행의 Latency를 최종 표에 포함하지 않았다.

## 산출물과 보존

사용자가 요청한 4개 Discovery 그림은 `poc_flashvep/` 바로 아래에 저장했다.
[4개 Discovery 그림 합본](../duetserve_qps_four_metrics_discovery_o200.png)을 제공한다.
이는 단일 Start의 예비 그림이다.
최종 반복 그림은 Evidence의 `figures/`에서 구분한다.
[독립 반복 4개 그림 합본](../duetserve_qps_four_metrics_confirmed_o200.png)도 `poc_flashvep/` 바로 아래에 저장했다.
Source는 `poc_flashvep/duetserve_agg_decode/`다.
Evidence는 `duetserve_agg_dp2_tp2_tbt_crossover_20261116/`다.
Publication Parent는 `{BASE}`다.
Branch는 `{BRANCH}`다.
사용자 Checkout, Index와 외부 Runtime 보존을 확인한다.
Commit 및 Remote SHA는 최종 Publication Receipt로 확인한다.
실패한 실행과 원시 기록을 삭제하지 않는다.
'''
    docs['aggregated_dp2_qps_degradation_results.md'] = f'''# Aggregated DP2의 QPS별 Decode 악화

## 고정 설정과 판정

B8192/S1024를 사용자 지정 Primary로 유지했다.
Primary 안에서는 QPS2–10 전체에서 설정을 고정했다.
한 QPS에서 다른 S로 바꾸지 않았다.
범위 변경 전에 완료한 S16의 B512/B8192 비교도 별도로 보존했다.
S128의 Budget-only 비교는 보충 자료다.
부분 완료한 B512 Confirmation을 Primary의 N128×3 결과처럼 합치지 않았다.

{ratio_table}
{crossing_table}
비율의 CI는 Arrival Seed를 짝지은 독립 Start Bootstrap이다.
각 비교는 3쌍과 27개의 재표본 조합을 사용한다.
작은 표본의 경험적 CI다.
Token이나 Request를 독립 Server Start처럼 재표본하지 않았다.
Spearman과 인접 변화는 탐색적 곡선 설명이다.
Plateau나 감소를 단조 증가로 바꾸지 않는다.

## 전체 QPS의 Start 평균

{performance_table(aggregate)}
## Seed별 원시 요약

{per_seed}
## 실제 Scheduler Geometry

{geometry_table}
각 값은 Start별 통계의 평균이다.
Mixed Graph Replay Counter는 Graph로 실행한 횟수다.
모든 Mixed Iteration의 횟수와 같지 않다.
일부 Counter는 Warmup을 포함한 Process Lifetime 범위다.
이를 측정 구간의 정확한 Mixed Iteration 수로 확대하지 않았다.
여기의 Decode Batch p95는 Prefill Row가 없는 Model Batch만 계산했다.
Mixed Batch의 Decode Row 수는 별도 Model Input Trace에 있다.
실제 Model Input Trace에서 새 Prefill, 기존 Decode Row와 재계산을 따로 기록했다.

## 범위 변경 전의 Sequence 상한 비교

B8192를 유지한 기존 Discovery 결과다.
QPS와 Seed는 각 비교에서 같다.
각 점은 단일 Start다.
S는 RPS나 실제 Batch Size가 아니라 Scheduler의 Sequence 상한이다.

{concurrency_table}
실제 Batch 크기는 KV Capacity와 Residency에도 제한된다.
S별 Graph Capture 집합은 각 실행 Config에 보존했다.
그 집합도 달라지므로 완전히 순수한 S-only 인과 실험으로 해석하지 않는다.
이후의 Primary는 S1024로 고정했다.

## 사전에 고정한 SLO

Output200의 TTFT-priority는 TTFT2000ms, ITL53ms, TPOT53ms다.
Balanced는 TTFT5000ms, ITL34ms, TPOT33ms다.
Decode-priority는 TTFT10000ms, ITL30ms, TPOT29ms다.
Joint Pass는 세 조건의 요청별 AND다.
ITL 조건은 첫 ITL을 포함한 모든 Interval에 적용한다.
Absolute SLO와 Native-relative 5% 비열등성은 다른 지표다.
이번에 새 Dynamic Method의 비열등성을 검증했다고 주장하지 않는다.

{slo_table}
이 결과는 `MEASURED`다.
이는 Finite Cohort이며 Sustainable Capacity가 아니다.
'''
    tp_rows = [r for r in aggregate if r['family'] == 'TP2_DIAGNOSTIC']
    tp_table = performance_table(tp_rows, family='TP2_DIAGNOSTIC')
    diagnosis = load('analysis/TP2_numerical_diagnosis.json')
    failure_table = table(['진단 실행', 'TP1 불일치 요청/8', '첫 TP2와 달라진 요청'],
        [[r['name'], len(r['original_output_gate']['failures']), len(r['differs_from_first_TP2_requests'])]
         for r in diagnosis['experiments']])
    docs['aggregated_tp2_qps_degradation_results.md'] = f'''# Aggregated TP2의 정합성 및 부하 Diagnostic

## 적격성 판정

TP2의 strict TP1 Output Parity는 `FAILED`다.
주 성능 비교의 적격 TP2 Curve는 `NOT_MEASURED`다.
실행 자체의 실패와 출력 정합성 실패를 구분한다.
TP2는 표준 Continuous vLLM의 단일 TP2 Instance로 실행했다.
두 TP2 Replica를 사용하지 않았다.
두 GPU가 전체 36개 Layer의 Tensor Shard를 보유한다.
Original BF16과 모델 Revision을 유지했다.

{failure_table}
반복, 순차 Eager와 Budget512에서 같은 TP2 출력이 나왔다.
그러나 TP1 Reference와 다른 Token Sequence를 생성했다.
이 진단은 Chunk Boundary나 Mixed Batch만으로 차이가 생겼다는 설명을 지지하지 않는다.
TP 분할에 따른 BF16 Reduction 순서는 가능한 원인이다.
동일 Prefix의 Logit과 Reduction별 비교가 없어 배타적 원인은 `NOT_IDENTIFIED`다.
차이를 무조건 허용 오차로 처리해 PASS로 바꾸지 않았다.

## Within-TP2 부하 진단

아래 수치는 `MEASURED`이지만 Primary Comparison에는 부적격이다.
별도의 TP2 N64 Reference를 사용했다.
독립 순차 Eager N8 출력과 그 Prefix를 비교했다.
이는 TP1 출력 또는 전체 KV/Hidden 정합성을 해결한 것이 아니다.
같은 TP2 안에서 QPS 변화의 양상을 설명하는 Diagnostic이다.
각 점은 N128의 단일 Server Start다.
TP2 Client는 Engine의 직접 수신 경계다.
DP2 Coordinator의 IPC 전달 경계와 다르다.
따라서 절대 Latency를 동일 Client 경계로 해석하지 않는다.

{tp_table}
TP Collective의 실제 GPU 실행시간과 Host 비용 분해는 `NOT_MEASURED`다.
TP2가 본질적으로 DP2보다 느리거나 빠르다고 일반화하지 않는다.
논문 Figure 7은 Qwen3-14B TP2다.
현재 Qwen3-8B 결과로 Figure 7의 정확한 재현을 주장하지 않는다.
'''
    docs['chunked_prefill_token_budget_sensitivity.md'] = f'''# Chunked Prefill Token Budget 민감도

## 통제 조건

Input8000/Output200을 유지했다.
QPS4와 QPS8에서 Budget256/512/1024/2048/4096/8192를 비교했다.
각 비교는 S128, 동일 Prompt, Arrival Seed와 Graph Capture 집합을 사용했다.
Per-request Long Prefill Threshold도 동일하게 유지했다.
이는 Budget의 고정 QPS 비교다.
QPS 증가 실험과 합치지 않는다.
각 점은 단일 Start다.
이 Budget Sweep은 Primary 범위 변경 전에 완료한 보충 실험이다.
이후 B512/S16의 추가 실행은 하지 않았다.

{budget_table}
고정 Capture 최대값은 256이다.
Budget에 따라 실제 Shape와 Graph/Eager 사용 비중은 달라질 수 있다.
이 변화는 실제 Scheduler 경로의 매개 효과다.
Budget512에 Capture512를 추가한 별도 Control도 보존했다.
그 값으로 Graph Coverage 영향을 별도로 확인한다.

{coverage_table}
이 Control의 다른 설정과 예정 Arrival은 동일하다.
단일 Start의 차이를 확정적인 인과 기여도나 CI로 표현하지 않는다.
큰 Budget이 모든 Workload와 SLO에서 더 효율적이라고 가정하지 않는다.
작은 Budget도 Mixed CUDA Graph를 사용할 수 있는 현재 Runtime의 기존 결과를 보존했다.
Raw Token/s와 Joint SLO-goodput을 함께 평가한다.
모든 수치는 `MEASURED`다.
각 Prefill Chunk 및 실제 Batch 분포는 Raw Metrics에 있다.
'''
    isolation = load('analysis/same_QPS_isolation_tests.json')
    iso_table = table(['Method', 'QPS', 'Metric', 'Aggregated/Native [95% CI]'],
        [[LABELS[r['arm']], r['offered_QPS'], metric, ratio(v)] for r in isolation
         if r['family'] == 'FRESH_CONFIRMATION' and r['offered_QPS'] in (2, 6, 10)
         for metric, v in r['tests'].items()])
    docs['native_pd_decode_isolation_comparison.md'] = f'''# Native PD Decode Isolation 및 긴 구간 비교

## 동일 QPS의 Isolation 비율

Native는 P0/D1이다.
기존 NIXL, Source-token Reuse와 Cohort-isolated D Admission을 유지했다.
이 경로는 논문 Dynamo의 완전한 재현이 아니다.
Agg-DP2의 같은 QPS와 Prompt를 비교했다.
아래 비율은 High/Low QPS 비율이 아니다.

{iso_table}
Native의 D 상한은16이다.
Agg-DP2 B8192의 상한은1024다.
따라서 Isolation 비율에는 Batch 구성과 Admission의 차이도 포함된다.
이 전체 비율을 순수한 Prefill Kernel 간섭 비용으로 표현하지 않는다.
동일 S16의 큰 Budget 실행군도 별도로 보존했다.

## Native의 Cohort Ingress와 TTFT

현재 Native는 기존 Decode Cohort가 끝날 때까지 다음 Import Group을 기다린다.
일반적인 Continuous D Ingress의 최적 기준선과 다르다.
아래 분해는 같은 요청의 Host Timestamp를 사용한다.
세 구간의 합은 요청별 TTFT와 일치한다.

{native_TTFT_table}
Arrival→P-ready에는 P Queue, 계산과 알림이 포함된다.
P-ready→D Socket에는 Cohort Guard와 Host Ingress 대기가 포함된다.
D Socket→First Output에는 Import, Barrier와 출력 전달이 포함된다.
이를 GPU 계산이나 Wire Transfer의 배타적 비용으로 표현하지 않는다.
Native의 높은 TTFT 전체를 Prefill Compute 병목으로 설명하지 않는다.
그 차이를 Dynamic Layer Scheduling의 잠재적 이득으로 그대로 계산하지 않는다.
원시 Start별 분해와 CI는 `analysis/native_cohort_ingress_TTFT_attribution.json`에 있다.

## 600초 관측과 Drain

{long_table}
{long_ratio_table}
Arrival-cohort Latency는 측정 구간의 모든 도착 요청을 완료까지 추적한다.
Active-window Latency는 측정 구간 안의 Token Interval만 사용한다.
두 통계를 대체하지 않는다.
완료 Output Token/s와 실제 Token 방출률도 구분했다.
Warmup과 Drain을 지속 가능한 처리 용량에 포함해 과장하지 않는다.
Queue Growth, 완료/도착 비율과 KV Lifetime을 함께 보존했다.
증가하는 Queue는 정상상태가 아니라 Overload Diagnostic이다.

## P 병목의 한계

Host Waiting Queue가 증가했다고 모두 Compute-ready로 분류하지 않는다.
WAITING_REMOTE_KVS, KV Allocation과 Source Retention을 구분한다.
Host Source-ready Rate는 순수 GPU Prefill 처리율이 아니다.
이 캠페인의 Clean P Kernel Busy는 Profiler Gate를 통과한 경우에만 인정한다.
서로 다른 실행에서 얻은 P Busy와 D 여유를 결합하지 않았다.
'''
    docs['aggregated_decode_interference_root_cause.md'] = f'''# Aggregated Decode 간섭 원인 분석

## 가장 강한 관측 근거

Budget과 QPS를 별도로 통제했다.
실제 Model Input의 Prefill/Decode 구성을 기록했다.
Output Frontier를 개별 Token Count와 연결했다.
Host Enclosure, CUDA Event Enclosure와 Kernel Interval을 구분했다.
별도 Decode-safe 시간 Budget이나 Deadline은 Stock Scheduler에 없다.
Token Budget을 30ms 같은 시간 Budget으로 해석하지 않았다.
기존 vLLM은 Prefill Row와 Decode Row를 같은 Model Forward에 넣을 수 있다.
별도의 Prefill Stream을 새로 추가해 동시 실행을 강제하지 않았다.
따라서 두 독립 Stream의 자원 경쟁으로 전체 악화를 설명하지 않는다.

## QPS2와 QPS6의 Interval 구성

같은 Host-only 계측과 Arrival Seed2026111616을 사용했다.
Prefill을 포함한 Interval과 Prefill이 없는 Interval을 실제 Model Input으로 분류했다.
Prefill 포함 분류에는 KV 압력으로 생성 Prefix를 재계산한 경우도 포함한다.
이를 모두 새로운 Prefill/Decode Mixed Batch라고 해석하지 않는다.
다음 항등식은 측정 Mean TBT 변화의 설명이다.

{mixture_table}
총 Mean TBT 변화는 {fmt(decomposition['measured_mean_TBT_change_ms'])}ms다.
이 분해는 `TRACE_ANALYSIS`다.
이는 배타적인 GPU 원인 기여도가 아니다.
Decode-only로 분류한 Interval도 앞선 Mixed Iteration 뒤에서 기다릴 수 있다.
작업 구성과 실행 지연은 서로 영향을 준다.

## 원인별 판정

1. 큰 Prefill Chunk와 긴 Mixed Iteration은 실제 로그에서 확인했다.
2. 높은 QPS의 Decode Batch와 Mixed Interval 비중 변화도 확인했다.
3. Host 수신 지연은 Worker 방출 간격과 별도로 계산했다.
4. Graph/Eager Coverage는 실제 Replay Counter와 Shape로 구분했다.
5. KV 압력에 따른 중복 Scheduled Prompt Rows는 Host Accounting으로 구분했다.
6. SM, HBM Bandwidth, Cache와 Launch의 배타적 기여도는 `NOT_IDENTIFIED`다.

RMSNorm Correction의 별도 Primitive 비용도 측정했다.
CUDA Event Enclosure이며 Kernel Union이나 Serving Speedup이 아니다.
Mask 준비와 H2D 비용을 제외한 범위를 명시했다.
그 측정만으로 전체 Serving 비용을 빼거나 더하지 않았다.

## 수 초짜리 Maximum ITL의 별도 원인

QPS7–10의 Fresh Confirmation 12개 Start를 추가로 확인했다.
각 Start에는 25,472개 ITL이 있다.
1초 이상인 Interval은 Start당 2개이며 전체 24개다.
24개 모두 같은 요청의 Prompt 재예약과 시간상 겹쳤다.
그 요청들은 이미 Decode Token을 출력한 상태였다.
일반적인 약300ms Mixed Iteration과 이 드문 대기를 구분해야 한다.

QPS10, Seed2026111622의 Req74가 한 예다.
22번째 Token과23번째 Token 사이의 Client ITL은4882.645ms였다.
Worker 방출 간격은4882.024ms였다.
두 Delivery Lag의 차이는0.620ms였다.
따라서 이 예의 수 초 지연을 Coordinator 전달 지연으로 설명하지 않는다.
가까운 Queue Sample의 KV Usage는100%이며 Free Block은0개였다.
같은 요청은 Prompt8000 Row를 Start0부터 다시 예약했다.
이 재예약 시각은 Output 공백 시작4558.639ms 이후였다.
그 시각부터 다음 Worker 출력까지는323.385ms였다.
두 구간은 Host Timestamp 차이며 배타적인 GPU 계산시간이 아니다.

이 기록은 KV 압력에 따른 Preemption 및 재계산 경로와 일치한다.
설치된 Scheduler의 `_preempt_request`는 Block을 해제하고 Computed Token 수를0으로 되돌린다.
인접 Queue Sample에서도 `PREEMPTED` 상태를 확인했다.
다만 정확한 Request별 Preemption 전이 시각은 `NOT_MEASURED`다.
GPU 재계산의 배타적 실행시간도 `NOT_MEASURED`다.
따라서 약4.88초를 하나의 Prefill Kernel 실행시간으로 해석하지 않는다.
이 드문 Tail과 Mean TBT 증가의 구성 변화는 다른 결과다.
전체 기록은 `analysis/fresh_primary_rare_ITL_prompt_reschedule.json`에 있다.
이 절의 증거 수준은 `TRACE_ANALYSIS`다.

## KV 압력과 중복 Scheduled Work

원래 Prompt의 고유 Token 구간을 Request별로 합집합 처리했다.
중복 구간을 다시 예약한 양은 별도로 계산했다.
분모는 각 실행의 Request 수 ×8000이다.

{work_table}
이 값은 `TRACE_ANALYSIS`다.
Host Scheduler의 예약 Row 수이며 실제 GPU FLOPs가 아니다.
생성된 Prefix의 추가 재계산 비용 전체를 이 표가 포함하는 것은 아니다.
중복 예약 비율만으로 ITL 증가의 배타적 기여도를 계산하지 않았다.

## Finite Cohort와 장기 관측의 차이

N128은 마지막 새 Prompt가 끝난 뒤의 Decode Tail도 평균에 포함한다.
장기 관측은 새 Prompt가 계속 유입되는 구간의 비중이 다르다.
아래 경계는 마지막 First-token 방출의 Host Timestamp다.
GPU Kernel이나 재계산 종료를 뜻하지 않는다.

{tail_table}
이는 `TRACE_ANALYSIS`다.
새 Prompt가 없는 구간에도 KV 작업과 Prefix 재계산이 있을 수 있다.
Batch 구성, KV 압력과 Prefill 빈도도 함께 달라질 수 있다.
따라서 두 실행의 Mean TBT 차이 전체를 Tail 비중만으로 설명하지 않는다.

## Profiler 간섭 검사

{profile_table}
Whole-run과 같은 Request/Token의 짧은 ROI를 모두 검사했다.
A/A 자체 변동도 5% 이하를 요구했다.
짧은 ROI의 Tail 변동으로 신뢰성 Gate가 실패한 경우도 구분했다.
계측을 통과시키기 위해 Threshold를 변경하지 않았다.

{kernel_table}
위 Kernel Union은 실제 Nsight 기록의 Stream 합집합이다.
겹치는 Kernel Duration을 단순 합산하지 않았다.
하지만 Clean Gate에 실패한 값은 계측된 Diagnostic일 뿐이다.
그 값을 Clean Native GPU Busy나 사용 가능한 Idle Capacity로 주장하지 않는다.
Memcpy와 NCCL Kernel은 Compute Kernel과 분리했다.
P-only Capture는 D가 실행한 NIXL Transfer의 Wire Time을 측정하지 않는다.
기록이 없는 통신 비용을 0으로 해석하지 않는다.
CUDA Graph Node는 실제 Kernel로 계산하고 Replay API를 다시 더하지 않았다.
GPU0의 실제 Timeline과 다른 실행이나 GPU의 시간을 합치지 않았다.

## CUDA12.9 CUPTI API28의 별도 검증

기존 고정 Host Buffer Collector를 CUDA12.9 CUPTI에 연결했다.
초기 두 Build는 CUDA Header의 버전 및 누락된 CRT Header 때문에 실패했다.
이 실패에서 Serving GPU 실행은 시작하지 않았다.
일치하는 CUDA Driver Header와 명시한 CRT Header 경로로 다시 Build했다.
단위 검증은 Eager100개와 Graph100개의 Kernel을 기록했다.
그 검증의 Dropped와 Invalid 기록은 모두0이었다.
이는 `CORRECTNESS_ONLY`이며 Serving 간섭 검증과 다르다.
이후 Native에서 제한적인 A/계측/A를 시도했다.
Background Activation 실행은 Warmup Timeout으로 실패했다.
단위 검증처럼 CUDA Context 생성 전에 활성화하는 별도 실행도 보존했다.
후자의 Capture는 전체 기록이며 분석 구간만 짧게 제한했다.
예정 분석 구간은10초 이후20초다.
양 끝의0.1초는 Union 계산에서 제외했다.
Serving Thread에는 Collector용 CUDA Fence를 추가하지 않았다.
기존 A/A 및5% 간섭 기준을 그대로 적용했다.
Native 계측 문제 이후 Aggregated의 추가 API28 실행은 하지 않았다.

{api28_table}
{api28_control_note}
위 표의 Busy와 Idle은 실제 기록의 수치다.
Clean Gate가 실패한 행은 Diagnostic이며 Clean Idle 수치가 아니다.
적격 행도 해당 짧은 관측 구간에만 유효하다.
Kernel-only Capture는 Memcpy나 NIXL Wire 비용을 측정하지 않는다.
Kernel Inactivity와 Decode-safe Layer 실행 예산은 다르다.
원시 Interval, Clock Anchor, Drop Counter와 간섭 오차는 `analysis/api28_*_serving_gate.json`에 있다.

## 종합 해석

큰 Token Budget은 긴 Model Iteration을 허용한다.
기존 Decode가 그 실행에 결합되면 Token 방출도 함께 늦어질 수 있다.
높은 동시성은 처리량과 Decode 지연의 절충점을 바꾼다.
이를 구조적인 D Compute 부족으로 단정하지 않는다.
범위 변경 전에 측정한 작은-Budget 결과도 보충 자료로 확인했다.
Clean GPU Idle과 안전한 추가 Layer Capacity는 여전히 별도 검증 대상이다.
'''
    assert set(docs) == set(DOCUMENTS) - {
        'duetserve_agg_dp2_tp2_tbt_crossover_reproduction_spec.md',
        'duetserve_agg_dp2_tp2_tbt_crossover_primary_scope_revision_v2.md'}
    destinations = {REPO / 'poc_flashvep/reports' / name: body for name, body in docs.items()}
    assert all(not path.exists() for path in destinations), 'Do not overwrite an existing report'
    for path, body in destinations.items():
        path.write_text(body)
    write(ROOT / 'state/report_generation.json', dict(time=time.time(), files=[str(p) for p in destinations],
        evidence='TRACE_ANALYSIS', scope='Reports render completed measured evidence. They do not replace frozen artifacts or raw logs.'))
    preserve()
    print('REPORTS_CREATED', len(destinations))


if __name__ == '__main__':
    main()
