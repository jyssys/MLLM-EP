"""Build new v2 reports from measured tables and explicit scientific decisions."""
import argparse
import collections
import datetime
import json
import time
from .common import ROOT, REPO, BASE, BRANCH, START, write, digest, preserve

REPORTS = REPO / 'poc_flashvep/reports'


def read(relative, default=None):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else default


def number(value, digits=3):
    if value is None:
        return 'NOT_MEASURED'
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) else str(value)


def test(value):
    if not value:
        return 'NOT_RUN'
    ci = value.get('CI95')
    return (f"{number(value.get('estimate'), 4)} "
            + (f"[{number(ci[0], 4)}, {number(ci[1], 4)}]" if ci else '[CI 없음]'))


def table(headers, rows):
    def cell(value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |']
                     + ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]) + '\n'


def link(relative):
    path = ROOT / relative
    return f'[{path.name}]({ROOT.name}/{relative})'


def emit(name, title, sections, final=False):
    now = datetime.datetime.now(datetime.timezone.utc)
    header = [f'# {title}', '', f"상태: {'FINAL' if final else 'ACTIVE_SNAPSHOT'}.",
              f'생성 시각: {now.isoformat()}.',
              f'경과 연구 시간: {(time.time() - START) / 3600:.3f}시간.',
              f'기준 Commit: `{BASE}`.', f'연구 Branch: `{BRANCH}`.', '']
    body = '\n'.join(header + sections).rstrip() + '\n'
    path = REPORTS / name
    path.write_text(body)
    return dict(path=str(path.relative_to(REPO)), SHA256=digest(path))


def c0_report(final):
    data = read('analysis/c0_output_map_v2.json', {})
    effect = read('analysis/c0_controlled_output_effect_v2.json', {})
    host = read('analysis/c0_output_host_path_contrast_v2.json', {})
    heldout = [r for r in data.get('cells', []) if r['split'] == 'heldout']
    counts = []
    for output in (8, 16, 64, 128, 192):
        rows = [r for r in heldout if r['output'] == output and r['completed_paired_starts'] == 3]
        points = collections.Counter(r['point_winner'] for r in rows)
        cis = collections.Counter(r['statistically_confirmed'] for r in rows)
        counts.append([output, len(rows), points['LAYER_POINT_WIN'], points['TOKEN_POINT_WIN'],
                       cis['LAYER_STATISTICALLY_CONFIRMED'], cis['TOKEN_STATISTICALLY_CONFIRMED'],
                       cis['TIE_OR_INCONCLUSIVE'], sum(r['both_arms_controlled_decode_gate_pass'] for r in rows)])
    cells = [[r['input_label'], r['output'], r['cut'], r['completed_paired_starts'],
              test(dict(estimate=r.get('TTFT_ratio'), CI95=r.get('TTFT_CI95'))) if 'TTFT_ratio' in r else 'NOT_RUN',
              r['point_winner'], r['statistically_confirmed'],
              test(r.get('Layer_over_C0Native_TTFT_p95')),
              test(r.get('Layer_over_Token_SLO_qualified_output_tokens_per_second')),
              r.get('both_arms_controlled_decode_gate_pass', False)]
             for r in heldout]
    return emit('dense_ar_longbench400_v2_c0_output_crossover.md', 'C0 Output 중심 Layer/Token 비교', [
        '## 실행 계약', '',
        '증거 수준은 `MEASURED_ONLINE`이다. 이 실험은 Controlled Fixed Output을 사용한다.',
        '실제 P Prefill/D Suffix/Decode Batch는 1/1/1이다. 모든 Worker는 B8192/S1024를 사용한다.',
        'Layer와 Token은 같은 Prompt, Arrival, Pair 및 정확한 Aggregate GEMM Rows를 사용한다.',
        'C0 Token은 Mode0 경로다. 이 결과를 C3 Strong Stock Token의 성능으로 표현하지 않는다.',
        'Native 실행은 같은 Input/Output/Start의 여러 Cut에서 재사용한다. 이를 추가 독립 Start로 계산하지 않는다.',
        'Point Winner는 Start-mean TTFT p95 비율로 결정한다. CI와 Decode Gate는 별도로 표시한다.',
        'N3 CI는 27개 Paired Bootstrap 조합을 사용한다. CI는 Pointwise이며 다중 비교를 보정하지 않았다.',
        'Decode Gate는 C0 Matched Native를 사용한다. C3의 두 Native 대조군을 대신하지 않는다.', '',
        '## Output별 독립 N3 결과', '',
        table(['Output Tokens', '완료 Cell/16', 'L Point', 'T Point', 'L CI 확정', 'T CI 확정', 'CI 불확실', '양쪽 Decode PASS'], counts),
        '## 전체 Held-out Cell', '',
        table(['Input', 'Output', 'Cut', 'Start 수', 'L/T TTFT p95 [95% CI]', 'Point Winner', 'CI 판정',
               'L/Native TTFT p95 [CI]', 'L/T SLO-goodput [CI]', '양쪽 Decode PASS'], cells),
        '## Output Intervention', '',
        'Output8과 Output192는 같은 Prompt와 Arrival Seed를 사용한다.',
        'Output 변경은 Decode Residency와 Capacity-relative Load도 변경한다.',
        '따라서 총 효과를 하나의 GPU 또는 Queue 원인으로 확정하지 않는다.', '',
        table(['Input Bucket', 'Cut', '(L/T O192)/(L/T O8) [95% CI]', '증거'],
              [[r['input_bucket'], r['cut'], test(r.get('ratio_of_axis_ratios')), r['evidence']]
               for r in effect.get('rows', [])]),
        '## Host 경로의 Endpoint 차이', '',
        '다음 값은 (Layer−Token)의 Output192−Output8 차이다. 단위는 ms다.',
        '첫 다섯 Stage는 Mean TTFT에 더해지는 Host 구간이다. Decode Residency는 별도 관측값이다.',
        'Queue와 Admission의 위치를 분리한다. 이를 GPU Operator의 인과적 기여율로 표현하지 않는다.', '',
        table(['Input', 'Cut', 'P 앞 Queue/Host', 'P Host Enclosure', 'Ready→D', 'D Admission/Prepare',
               'D First Host Enclosure', 'Mean TTFT', 'Decode Residency'],
              [[r['input_bucket'], r['cut']] +
               [test(r['output192_minus8_in_Layer_minus_Token_mean_ms'].get(field)) for field in (
                   'P queue and host admission', 'P prepare to ready host enclosure', 'Ready to D submit',
                   'D admission and prepare', 'D prepare to first worker host enclosure', 'Mean TTFT', 'Decode residency')]
               for r in host.get('rows', [])]),
        '## 원시 수치와 Figure', '',
        link('figures/c0_output_map_per_start_v2.csv'), '', link('figures/c0_output_map_cells_v2.csv'), '',
        link('figures/c0_heldout_point_map_v2.png'), '', link('figures/c0_heldout_confidence_map_v2.png'), '',
        link('figures/c0_heldout_safety_map_v2.png'), '', link('figures/c0_controlled_output_effect_v2.csv'),
    ], final)


def c3_report(final):
    selection = read('configs/c3_final_static_selection_v2.json', {})
    chosen = [[family, value['item']['label'], value['item']['cap'], value['item'].get('credit', 'Config 참조'),
               value['score']['discovery_point_decode_screen']]
              for family, value in selection.get('chosen', {}).items()]
    sections = ['## 실행 계약', '',
        '증거 수준은 `MEASURED_ONLINE`이다. Main은 Natural EOS와 Output Cap2048을 사용한다.',
        'C3는 실제 Continuous Batching을 사용한다. 실제 Batch는 각 실행의 CSV에 기록한다.',
        'B8192/S1024는 Scheduler 상한이다. 이 값은 실제 Batch가 아니다.',
        'Stock Token은 원래 Compiled Full-model 경로를 사용한다. C0 Mode0 Token으로 대체하지 않았다.',
        'Hybrid는 두 축을 모두 분할한다. Layer-only 또는 Token-only를 Hybrid로 이름만 변경하지 않았다.',
        '고정 후보는 Discovery에서 선택했다. Held-out 성능을 본 뒤 후보를 변경하지 않는다.',
        'Main400은 Discovery와 Held-out 요청을 모두 포함한다. Held-out192 결과는 별도 CI로 표시한다.',
        'Natural EOS의 Output Token 총수가 5%를 초과해 다르면 Work-unmatched로 표시한다.',
        'Fixed-length Replay는 별도 Diagnostic이다. 이를 Natural Serving 이득으로 사용하지 않는다.',
        'Finite-cohort Throughput은 Sustainable Capacity가 아니다.', '',
        '## 동결 후보', '', table(['Family', '설정', 'D Cap', 'Credit', 'N1 Decode Point Screen'], chosen)]
    for stem in ('c3_main_serving_v2', 'c3_heldout_serving_v2', 'c3_heldout_replay_v2',
                 'c3_continuous_iso8_output_bridge_v2'):
        data = read(f'analysis/{stem}_analysis.json', {})
        sections += ['', '## ' + stem, '']
        if not data:
            sections += ['`NOT_RUN` 또는 분석 대기 상태다. 계획 파일과 실행 Receipt를 함께 확인한다.']
            continue
        rows = []
        for r in data['cells']:
            f = r['fields']; gate = r['decode_gate']
            rows.append([r['QPS'], r['label'], r['independent_starts'],
                         number(f['TTFT_p95_ms']['mean']), number(f['mean_TBT_ms']['mean']),
                         number(f['ITL_p99_ms']['mean']), number(f['TPOT_p95_ms']['mean']),
                         number(f['max_ITL_ms']['mean']),
                         number(f['generated_output_tokens_per_second']['mean']),
                         number(f['SLO_qualified_output_tokens_per_second']['mean']),
                         number(f['every_ITL_SLO_attainment']['mean'], 4),
                         number(f['Joint_SLO_attainment']['mean'], 4),
                         r['quality_passed_all_starts'], gate['qualified']])
        sections += [table(['QPS', 'Method', 'Starts', 'TTFT p95 ms', 'Mean TBT ms', 'ITL p99 ms', 'TPOT p95 ms/token',
                            'Max ITL Start-mean ms', 'Raw Tokens/s', 'SLO Tokens/s', '모든 ITL 통과율',
                            'Joint 통과율', '품질 PASS', 'Decode CI+Absolute PASS'], rows),
                     'Max ITL 열은 Start별 최댓값의 평균이다. 개별 Start 최댓값은 Per-start CSV에 있다.', '',
                     table(['QPS', '비교', 'Paired Starts', 'TTFT p95 비율 [95% CI]', 'SLO-goodput 비율 [95% CI]', 'Output Work 불일치'],
                           [[r['QPS'], r['left'] + '/' + r['right'], r['paired_starts'],
                             test(r['ratios']['TTFT_p95_ms']),
                             test(r['ratios']['SLO_qualified_output_tokens_per_second']), r['output_work_unmatched']]
                            for r in data['comparisons']]),
                     link(f'figures/{stem}_per_start.csv'), '', link(f'figures/{stem}_summary.csv'), '',
                     link(f'figures/{stem}_paired_comparisons.csv')]
    return emit('dense_ar_longbench400_v2_c3_continuous_serving.md', 'Strong C3 Continuous Serving', sections, final)


def safety_report(final):
    sections = ['## 동결 계약', '',
        'TTFT 제한은 5000ms다. 모든 Client-visible ITL 제한은 34ms다. Request TPOT 제한은 33ms/token이다.',
        '상대 Gate는 ITL p99와 TPOT p95의 Paired CI 상한이 모두 1.05 이하여야 한다.',
        'NativeStrong과 같은 D 정책을 사용하는 Common-policy Native를 모두 대조군으로 사용한다.',
        '기존 절대 Decode Attainment 95% 조건도 별도로 검사한다.',
        'TTFT 개선으로 Decode 실패를 상쇄하지 않는다. N1 또는 N2는 CI Gate를 통과할 수 없다.', '',
        '## 두 Native 대비 Tail Gate', '']
    rows = []
    for stem in ('c3_main_serving_v2', 'c3_heldout_serving_v2', 'c3_continuous_iso8_output_bridge_v2'):
        for r in read(f'analysis/{stem}_analysis.json', {}).get('cells', []):
            if r['label'] not in ('Layer', 'StockToken', 'Hybrid', 'Deflection'):
                continue
            gate = r['decode_gate']; refs = gate.get('tests', {})
            rows.append([stem, r['QPS'], r['label'], r['independent_starts'],
                         test(refs.get('NativeStrong', {}).get('ITL_p99_ms')),
                         test(refs.get('NativeStrong', {}).get('TPOT_p95_ms')),
                         test(refs.get('NativeCommon', {}).get('ITL_p99_ms')),
                         test(refs.get('NativeCommon', {}).get('TPOT_p95_ms')),
                         gate.get('relative_5percent_pass', False),
                         gate.get('absolute_decode_attainment_95percent_pass', False), gate['qualified']])
    sections += [table(['Trace', 'QPS', 'Method', 'Starts', 'ITL/Strong [CI]', 'TPOT/Strong [CI]',
                       'ITL/Common [CI]', 'TPOT/Common [CI]', '상대 PASS', '절대 PASS', '통합 PASS'], rows),
                 '', '## Kernel 및 Window', '']
    activity = read('analysis/c3_kernel_copy_cpc_results_v2.json', {})
    captured = []
    for r in activity.get('results', []):
        for role in r.get('roles', []):
            captured.append([r['label'], r['qps'], role['role'], r['evidence'],
                             number(100 * role['kernel_busy_fraction']), number(role['ROI_seconds']),
                             r.get('observer_gate', {}).get('status', 'FAILED')])
    sections += [table(['Method', 'QPS', 'Role', '증거', 'Kernel Busy %', 'ROI 초', 'Observer Gate'], captured),
        'Kernel Busy는 Clean/Profile/Clean Gate가 통과한 Capture만 적격 수치로 사용한다.',
        'Observer Gate는 전체 Trace와 ROI 노출 Request를 각각 검사한다. 허용 변화는 5%다.',
        'Kernel Gap은 모든 Engine의 Physical Idle 또는 실행 가능한 Decode-safe Window와 같지 않다.',
        'Worker Deadline Residual은 마지막 Token 시각+34ms의 관측 Proxy다.',
        '이 Proxy는 후보 준비, KV Credit, Stream 가용성 및 미래 Arrival 안전성을 보장하지 않는다.',
        'Online Dynamic Insertion은 검증된 Window와 Tail Predictor가 없으면 `NOT_RUN`이다.',
        'GPU Operator-exclusive GEMM/Attention, SM 및 HBM 비중은 적격 계측이 없으면 `NOT_MEASURED`다.',
        'NIXL Host API와 Kernel Gap의 시간 중첩은 인과 관계를 입증하지 않는다.', '',
        link('analysis/c3_kernel_copy_cpc_results_v2.json'), '', link('figures/c3_kernel_busy_and_queue_v2.csv')]
    return emit('dense_ar_longbench400_v2_decode_safety_and_windows.md', 'Decode Tail 보호와 Window 판정', sections, final)


def selector_report(final):
    prior = read('analysis/heldout_output_prior_prediction_v2.json', {})
    rule = read('analysis/c0_observable_axis_rule_diagnostic_v2.json', {})
    rows = [[name, r['n'], number(r['mean_absolute_error_tokens']), number(r['median_absolute_error_tokens']),
             number(r['output_bin_accuracy'], 4), r['overflow_reference_requests']]
            for name, r in prior.get('models', {}).items()]
    return emit('dense_ar_longbench400_v2_output_prior_selector.md', 'Output Prior와 Selector 검증 범위', [
        '## 관측 가능한 Output Prior', '',
        'Training은 Discovery208의 독립 Compiled C1 Output만 사용한다.',
        'Feature는 사전에 주어진 Task Label과 Tokenized Input Bucket이다.',
        'Held-out192의 미래 EOS 길이는 Feature나 Training Label로 사용하지 않는다.',
        'Task/Input 평균은 Log1p 공간에서 8개 Pseudo-observation으로 축소한다.',
        '모든 요청의 max_tokens=2048은 알려진 값이다. 하지만 요청을 구분하는 Feature는 아니다.', '',
        table(['Model', 'Held-out Requests', 'MAE Tokens', 'Median AE Tokens', 'Output-bin Accuracy', 'Reference >200'], rows),
        '예측 정확도는 Axis Winner 정확도 또는 실제 Serving 이득이 아니다.',
        'Dataset 선정을 위해 Candidate EOS를 사용했다. Held-out은 C3 Tuning에 대한 분리다.',
        '이 요청을 모든 연구 단계에서 처음 보는 완전 미관측 데이터라고 표현하지 않는다.', '',
        '## Controlled-output Rule', '',
        'C0 Rule은 Discovery의 Known Fixed Output과 Input Bucket으로 동결한다.',
        '이 실험의 Output Budget은 이미 알려져 있다. Natural EOS의 미래 길이와는 다르다.',
        'C0 Rule의 Held-out 예측은 Diagnostic이다. C3 Queue-updating Replay로 해석하지 않는다.', '',
        table(['Rule', '평가 범위', 'Cells', 'Point Winner Accuracy', 'Mean Cell TTFT p95 Regret ms', 'Mean Relative Regret'],
              [[r['model'], r['scope'], r['cells'], number(r['point_winner_accuracy'], 4),
                number(r['mean_cell_TTFT_p95_regret_ms']), number(r['mean_cell_TTFT_p95_regret_fraction'], 4)]
               for r in rule.get('summary', [])]),
        link('configs/c0_output_selector_rules_frozen_v2.json'), '',
        link('analysis/c0_observable_axis_rule_diagnostic_v2.json'), '',
        link('analysis/heldout_output_prior_prediction_v2.json'), '',
        '## Dynamic Contribution', '',
        '검증된 Decode-safe Window가 없으면 Online Axis 또는 2D Selector는 실행하지 않는다.',
        '미래 EOS를 읽는 Oracle은 Online Selector가 아니다.',
        'Queue, KV Credit 및 ITL Tail을 검증하지 못한 Oracle은 Serving Headroom으로 사용하지 않는다.',
        '복잡한 Dynamic Controller를 구현하지 않은 경우 상태를 `NOT_RUN`으로 유지한다.',
    ], final)


def figure_inventory(final):
    records = []
    for png in sorted((ROOT / 'figures').glob('*.png')):
        pdf, csv = png.with_suffix('.pdf'), png.with_suffix('.csv')
        records.append(dict(stem=png.stem, PNG=str(png.relative_to(ROOT)), PNG_SHA256=digest(png),
                            PDF=str(pdf.relative_to(ROOT)) if pdf.exists() else None,
                            CSV=str(csv.relative_to(ROOT)) if csv.exists() else None,
                            complete=pdf.exists() and csv.exists()))
    write(ROOT / 'analysis/figure_inventory_v2.json', dict(time=time.time(), figures=records))
    return emit('dense_ar_longbench400_v2_figures.md', 'Figure 및 대응 CSV 목록', [
        '각 Figure는 실제 측정, 진단 또는 Structural Accounting의 범위를 그림 안에 표시한다.',
        '측정하지 않은 QPS를 생성하지 않는다. 선은 측정점 사이의 시각적 연결이다.',
        'N1 Point에는 Start-level CI를 만들지 않는다.',
        'Snapshot 생성 시점의 파일 목록은 다음과 같다.', '',
        table(['Figure', 'PNG', 'PDF', '대응 CSV', '3개 형식 존재'],
              [[r['stem'], link(r['PNG']), link(r['PDF']) if r['PDF'] else 'NOT_RUN',
                link(r['CSV']) if r['CSV'] else 'NOT_RUN', r['complete']] for r in records]),
    ], final)


def summary(final):
    verdict = read('analysis/scientific_verdict_v2.json', {})
    freeze = read('data/balanced400_v2_freeze_manifest.json', {})
    sections = ['## 연구 판정', '', verdict.get('decision', 'ACTIVE. 최종 판정 전이다.'), '',
        '## Dataset', '',
        f"선정 요청 수는 {freeze.get('n', 'NOT_RUN')}개다. Input 구간별 요청 수는 {freeze.get('input_totals', [])}다.",
        '처음 세 Input 구간은 각 Output Cell에 25개를 확보했다.',
        '20K–36K 구간의 긴 Output 후보는 부족했다. 원문을 수정하지 않고 동결된 재배분 규칙을 적용했다.', '',
        table(['Input', 'Output1–10', 'Output11–50', 'Output51–100', 'Output101–200'],
              [[label] + values for label, values in zip(('3K–6K', '6K–10K', '10K–20K', '20K–36K'), freeze.get('selected_cells', []))]),
        f"Dataset SHA256: `{freeze.get('SHA256', 'NOT_RUN')}`.",
        '선정 시 Natural Output과 C3 Compiled C1 Output은 수치 경로가 다르다.',
        '따라서 선정 Output 구간이 실제 C3의 모든 출력 길이를 보장하지 않는다. Dataset을 다시 선정하지 않았다.', '',
        '## 최종 연구 질문', '']
    for index, question in enumerate((
        '목표 4×4 Dataset 분포를 달성했는가?',
        '긴 Output에서 Layer의 Point Win이 증가하는가?',
        'Input과 QPS를 통제한 Output 효과가 유지되는가?',
        'C0 Crossover가 C3에서도 재현되는가?',
        'Layer 또는 Hybrid가 Strong Stock Token의 TTFT를 개선하는가?',
        'Decode ITL/TBT/TPOT를 보호하는가?',
        'SLO-goodput이 증가하는가?',
        '실행 가능한 Decode-safe Window가 검증됐는가?',
        'Dynamic 2D Selector가 Best Static보다 추가 이득을 만드는가?',
        '성공하지 못한 조건의 원인은 무엇인가?'), 1):
        sections += [f'### Q{index}. {question}', '', verdict.get('answers', {}).get(str(index), '최종 분석 전이다.'), '']
    sections += ['## Evidence 해석 제한', '',
        'Point Winner, CI 확정 결과 및 Decode Safety를 별도로 표시한다.',
        'Request나 Token 수를 독립 Server Start 수로 사용하지 않는다.',
        'GPU Host Enclosure를 Operator-exclusive GPU Time으로 표현하지 않는다.',
        'Static Cut/정책 탐색은 측정한 후보 집합에 한정된다. 전역 최적을 주장하지 않는다.',
        '실패한 Gate와 미실행 조건은 삭제하지 않는다. 과거 v1 결과를 덮어쓰지 않는다.', '',
        '## 상세 보고서', '']
    for filename in ('dense_ar_longbench400_v2_candidate_availability.md', 'dense_ar_longbench400_v2_dataset_contract.md',
                     'dense_ar_longbench400_v2_c0_output_crossover.md', 'dense_ar_longbench400_v2_c3_continuous_serving.md',
                     'dense_ar_longbench400_v2_decode_safety_and_windows.md', 'dense_ar_longbench400_v2_output_prior_selector.md',
                     'dense_ar_longbench400_v2_research_decision_log.md', 'dense_ar_longbench400_v2_figures.md'):
        sections += [f'- [{filename}]({filename})']
    sections += ['', '## 재현성 및 자원', '',
                 '공식 원문은 별도 라이선스를 따를 수 있다. Bulk Context를 새로 공개 배포하지 않는다.',
                 '공개 Identity/Hash Roster와 Reconstruction Code로 Dataset을 복원한다.',
                 link('publication/provenance/dataset_rosters_v2.json'), '',
                 link('local_reconstruction_validation/reconstruction_validation_v2.json'), '',
                 link('analysis/raw_metric_recalculation_v2.json'), '',
                 link('state/resource_closeout_v2.json'), '',
                 'Commit/Push 및 Remote SHA는 별도의 최종 Publication Receipt에 기록한다.']
    return emit('dense_ar_longbench400_v2_output_aware_2d_final_summary.md',
                'LongBench400-v2 Output-aware 2D Serving', sections, final)


def run(final=False):
    preserve()
    if final:
        assert read('state/resource_closeout_v2.json', {}).get('passed')
        assert read('analysis/raw_metric_recalculation_v2.json', {}).get('passed')
        assert read('analysis/scientific_verdict_v2.json', {}).get('decision')
    records = [function(final) for function in (c0_report, c3_report, safety_report, selector_report, figure_inventory, summary)]
    write(ROOT / 'state/report_generation_v2.json', dict(time=time.time(), final=final, files=records))
    preserve()
    print('V2_REPORTS_GENERATED', len(records), 'FINAL' if final else 'ACTIVE_SNAPSHOT', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    run(parser.parse_args().final)
