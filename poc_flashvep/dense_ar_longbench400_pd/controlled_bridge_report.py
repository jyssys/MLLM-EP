"""Report the complete bridge without treating C0 results as C3 evidence."""
import json
import numpy as np
from .common import ROOT, REPO, write
from .confirmation_analysis import ratio
from .final_report_tables import interval
from .final_service_decomposition import pair, STAGES
from .service_accounting import analyze


ARMS = ('NativeStrong', 'NativeCommon', 'Layer', 'StockToken',
        'TokenMode0Control', 'Hybrid')
FIELDS = ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms',
          'SLO_qualified_output_tokens_per_second')


def output_effect(plan, arm):
    """Compare output contracts, including their reservation effects."""
    pairs = []
    for start in range(3):
        short = next(c for c in plan['cells'] if c['D_cap'] == 1
                     and c['output'] == 16 and c['start'] == start)
        long = next(c for c in plan['cells'] if c['D_cap'] == 1
                    and c['output'] == 64 and c['start'] == start)
        left, right = long['jobs'][arm], short['jobs'][arm]
        cfgs = [json.loads((ROOT / 'raw' / n / 'configuration.json').read_text())
                for n in (left, right)]
        for key in ('source_capsule_SHA256', 'dataset_SHA256', 'arrival_offsets',
                    'seed', 'gpus', 'ignore_eos', 'continuous_source_credit_limit',
                    'execution_family', 'axis_kind', 'axis_cut',
                    'suffix_by_request', 'D_running_admission_cap'):
            assert cfgs[0].get(key) == cfgs[1].get(key), (arm, key)
        metrics = [json.loads((ROOT / 'analysis' / (n + '_metrics.json')).read_text())
                   for n in (left, right)]
        paths = [analyze(n) for n in (left, right)]
        a, b = paths
        assert [(r['request_id'], r['prompt_id'], r['input_tokens']) for r in a['rows']] == [
            (r['request_id'], r['prompt_id'], r['input_tokens']) for r in b['rows']]
        delta = {k: a['means'][k] - b['means'][k] for k in a['means']}
        pairs.append(dict(left=left, right=right, metrics=metrics,
                          host_mean_delta_ms=delta))
    return dict(arm=arm, O64_over_O16={f: ratio([p['metrics'][0][f] for p in pairs],
        [p['metrics'][1][f] for p in pairs]) for f in FIELDS},
        host_mean_delta_ms={k: float(np.mean([p['host_mean_delta_ms'][k] for p in pairs]))
                           for k in pairs[0]['host_mean_delta_ms']},
        scope='Output contract intervention. Output cap can also change KV reservation. '
              'This does not isolate decode residency from every runtime effect.')


def main():
    data = json.loads((ROOT / 'analysis/controlled_continuous_I3_bridge_results_v1.json').read_text())
    assert data['complete'], 'Do not replace missing bridge outcomes with a complete table.'
    plan = json.loads((ROOT / 'configs/controlled_continuous_I3_bridge_frozen_v1.json').read_text())
    stages = []
    for cell in data['cells']:
        for arm in ('Layer', 'Hybrid', 'TokenMode0Control'):
            results = [pair(a, b) for a, b in zip(cell['arms'][arm]['names'],
                       cell['arms']['StockToken']['names'])]
            stages.append(dict(output=cell['controlled_output'], cap=cell['D_cap'],
                arm=arm, reference='StockToken', starts=3, rows=results,
                stage_mean_delta_ms={k: float(np.mean([r['stages_ms'][k] for r in results]))
                                     for k in STAGES},
                mean_TTFT_delta_ms=float(np.mean([r['mean_TTFT_delta_ms'] for r in results])),
                post_first_residency_delta_ms=float(np.mean([
                    r['post_first_output_residency_delta_ms'] for r in results]))))
    effects = [output_effect(plan, a) for a in ARMS]
    write(ROOT / 'analysis/controlled_C3_bridge_causal_boundaries_v1.json',
          dict(stages=stages, output_contract_effects=effects,
               cap_interventions=data['cap_interventions'], evidence='TRACE_ANALYSIS',
               scope='Same-request host boundaries. Additive mean TTFT, not p95 or GPU-exclusive time. '
                     'Output and admission interventions remain separate.'))
    path = REPO / 'poc_flashvep/reports/dense_ar_longbench400_controlled_continuous_bridge.md'
    marker = '## 완료한 C3 측정'
    prefix = path.read_text().split(marker)[0].rstrip()
    prefix = prefix.replace('상태: `CORRECTNESS_IN_PROGRESS`.\n아직 이 문서에서 C3 성능 우위를 판정하지 않는다.',
                            '상태: `MEASURED_ONLINE_CONTROLLED_OUTPUT`.\n54개 Start를 완료했다. 안전성 및 성능 판정은 아래 표에 분리했다.')
    lines = ['', marker, '',
        '각 값은 독립 Start 3개의 Metric 평균이다. 모든 조건은 N16이다.',
        'O16은 Start당 256 Tokens다. O64는 Start당 1024 Tokens다.',
        'CI는 같은 Seed의 Server Start를 재표본했다. 27개 조합을 완전 열거했다.',
        'Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않았다.', '',
        '| Output / D-cap | Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(output tokens/s) |',
        '|---|---|---:|---:|---:|---:|']
    for cell in data['cells']:
        for arm in ARMS:
            means = cell['arms'][arm]['means']
            lines.append(f"| {cell['controlled_output']}/{cell['D_cap']} | {arm} | " +
                         ' | '.join(f'{means[f]:.6f}' for f in FIELDS) + ' |')
    lines += ['', '### 같은 조건의 비교와 Decode Gate', '',
        '| Output / D-cap | Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] | Decode 5% | Absolute Decode≥95% |',
        '|---|---|---|---|---|---|---|---|']
    for cell in data['cells']:
        for arm, refs in cell['comparisons'].items():
            for ref, test in refs.items():
                fields = ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', FIELDS[-1])
                lines.append(f"| {cell['controlled_output']}/{cell['D_cap']} | {arm}/{ref} | " +
                    ' | '.join(interval(test['ratios'][f]) for f in fields) +
                    f" | {test['decode_5percent_CI_pass']} | {test['absolute_decode_pass']} |")
    lines += ['', '성능 우위와 Decode 보호를 별도로 검사한다.',
        'Layer와 Hybrid의 Serving 주장은 두 Native 대조군의 Gate를 모두 요구한다.',
        'Mode0 Token 대비 우위만으로 강한 Stock Token 대비 우위를 주장하지 않는다.', '',
        '### Actual Geometry와 Padding', '',
        '| Output / D-cap | Arm | P/D 최대 Request Batch | P/D 추가 Prefill Padding Query 합 |',
        '|---|---|---|---|']
    for cell in data['cells']:
        for arm in ARMS:
            row = cell['arms'][arm]
            batch = '/'.join(str(max(g[r]['maximum_requests'] for g in row['geometry']))
                             for r in ('prefill', 'decode'))
            padding = '/'.join(str(sum(p[r]['total_extra_padded_queries'] for p in row['padding']))
                               for r in ('prefill', 'decode'))
            lines.append(f"| {cell['controlled_output']}/{cell['D_cap']} | {arm} | {batch} | {padding} |")
    lines += ['', 'Padding 합은 3개 Start의 Model-input Query Slot 합이다.',
        '이를 Partial Adapter의 실제 GEMM Rows 또는 GPU-exclusive 시간으로 환산하지 않는다.', '',
        '### 같은 Request의 Host 경계', '',
        '단위는 ms다. 값은 Arm에서 Stock Token을 뺀 평균이다.',
        'Prepare는 Host 입력 준비 경계다. GPU Kernel 시작이 아니다.',
        '여섯 구간은 Mean TTFT 차이에 합산된다. TTFT p95는 합산하지 않는다.', '',
        '| Output / D-cap | Arm − StockToken | Arrival→P Prepare | P Prepare→Ready | Ready→D Submit | D Submit→Prepare | D Prepare→First Worker | Worker→Client | Mean TTFT 차이 | 이후 Residency 차이 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in stages:
        vals = [row['stage_mean_delta_ms'][k] for k in STAGES]
        vals += [row['mean_TTFT_delta_ms'], row['post_first_residency_delta_ms']]
        lines.append(f"| {row['output']}/{row['cap']} | {row['arm']} | " +
                     ' | '.join(f'{v:.6f}' for v in vals) + ' |')
    lines += ['', '이후 Residency는 첫 Output 이후의 Client 관측 구간이다.',
        '이를 TTFT 또는 순수 Decode GPU 시간에 더하지 않는다.', '',
        '### Output 및 Admission Intervention', '',
        '| Intervention | Arm | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] |',
        '|---|---|---|---|---|---|']
    for effect in effects:
        lines.append('| O64/O16, D-cap1 | ' + effect['arm'] + ' | ' +
                     ' | '.join(interval(effect['O64_over_O16'][f]) for f in FIELDS) + ' |')
    for effect in data['cap_interventions']:
        lines.append('| Cap2/Cap1, O64 | ' + effect['arm'] + ' | ' +
                     ' | '.join(interval(effect['cap2_over_cap1'][f]) for f in FIELDS) + ' |')
    lines += ['', 'Output Intervention은 Cap에 따른 KV Reservation 변화도 포함한다.',
        'Output Length와 Decode Residency만의 순수 효과로 단정하지 않는다.',
        'Admission Intervention은 Source, Prompt, Arrival, Output과 Axis를 고정한다.',
        'D-cap2는 실제 Decode Batch2를 강제하지 않는다.', '',
        '### 한계와 재현 자료', '',
        '이 결과는 기존 C0에서 선택한 작은 Prompt 집합의 Bridge다.',
        '전체 Natural400 또는 보지 않은 새로운 Request Population의 결과가 아니다.',
        'C0와 C3는 Source, Pair와 Seed가 다르다. 둘 사이의 차이는 단일 정책의 인과 효과가 아니다.',
        '강한 Natural400 Static 비교와 Dynamic Window 검증을 이 결과로 대체하지 않는다.',
        '실제 GPU Operator별 C3 비용 비중과 미래 Decode-safe Window는 이 실험으로 식별하지 못했다.', '',
        '- [전체 Metric와 Gate](dense_ar_longbench400_2d_20261121/analysis/controlled_continuous_I3_bridge_results_v1.json)',
        '- [같은 Request의 경계와 Intervention](dense_ar_longbench400_2d_20261121/analysis/controlled_C3_bridge_causal_boundaries_v1.json)', '']
    path.write_text(prefix + '\n' + '\n'.join(lines), encoding='utf-8')
    print('CONTROLLED_C3_BRIDGE_REPORT', len(stages), 'same-ID comparisons', flush=True)


if __name__ == '__main__':
    main()
