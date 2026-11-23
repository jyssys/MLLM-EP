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
    if path.exists() and path.stat().st_size>=90*1024**2:
        marker=ROOT/'publication/derived_tables'/(path.parent.name+'_'+path.name+'.manifest.json')
        if marker.exists():
            return f'[{path.name} — lossless archive manifest]({ROOT.name}/{marker.relative_to(ROOT)})'
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
    chosen = []
    for family,value in selection.get('chosen',{}).items():
        cfg=read('configs/'+value['item']['jobs'][-1]+'.json',{})
        chosen.append([family,value['item']['label'],cfg.get('axis_cut',0),cfg.get('hybrid_suffix_fraction',0),
            value['item']['cap'],cfg.get('continuous_source_credit_limit',0),cfg.get('P_compile_prefix',False),
            cfg.get('D_compile_partial',False),cfg.get('fuse_short_hybrid',False),
            value['score']['discovery_point_decode_screen']])
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
        '## 동결 후보', '', table(['Family', '설정', 'Layer Cut 또는 Token Work Cut', 'Hybrid Suffix 비율',
            'D Cap', 'Source Credit', 'P Partial Compile', 'D Partial Compile', 'Short Fused', 'N1 Decode Point Screen'], chosen),
        'Stock Token의 Partial Compile 표시는 적용 대상이 아니다. Stock Token은 원래 Full-model Compiled 경로를 사용한다.',
        '## C3 Iso-work Bridge의 제한', '',
        'Bridge는 Input 3K–6K와 20K–36K의 원문을 각각 16개 사용한다.',
        'Layer Last8, Stock Token Work Cut8 및 Hybrid Last4/Suffix≈1/8은 정확한 Aggregate P/D Rows를 맞춘다.',
        'Output16/192와 QPS4를 사용한다. 각 조건은 독립 Start3회다.',
        'Axis의 D Cap은 2다. Layer/Hybrid Source Credit은 8이다. Stock Token의 Source Credit은 0이다.',
        'Credit0은 무제한 Coordinator Window다. Worker KV 및 Scheduler 제한이 없다는 뜻은 아니다.',
        'C0와 QPS, 실제 Batch 및 Compiler 경로가 다르다.',
        '따라서 C0/C3 차이를 Batch 하나의 인과 효과로 표현하지 않는다.']
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
        execution = collections.Counter()
        for block in data.get('execution_status', []):
            for label, status in block['statuses'].items():
                execution[block['qps'], label, status] += 1
        sections += ['', '동결 계획의 실행 상태는 다음과 같다. 미완료 조건의 값을 보간하지 않는다.', '',
                     table(['QPS', 'Method', '실행 상태', '계획 Start 수'],
                           [[qps, label, status, count] for (qps, label, status), count in sorted(execution.items())])]
    return emit('dense_ar_longbench400_v2_c3_continuous_serving.md', 'Strong C3 Continuous Serving', sections, final)


def safety_report(final):
    census=read('analysis/c0_heldout_client_tail_census_v2.json',{})
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
    tail_rows=[]
    heldout_plan=read('jobs/c3_heldout_serving_v2.json',{})
    for block in heldout_plan.get('blocks',[]):
        for label,name in block['arms'].items():
            value=read('analysis/'+name+'_decode_tail_host_v2.json',{}).get('summary')
            if value:
                tail_rows.append([block['qps'],block['start'],label,value['intervals'],value['tail_intervals'],
                    value['tail_with_prefill_host_step'],value['tail_with_NIXL_API_host_overlap'],
                    value['tail_with_new_registration'],number(value['tail_mean_worker_gap_ms']),
                    number(value['tail_mean_delivery_delay_delta_ms'])])
    sections += [table(['Trace', 'QPS', 'Method', 'Starts', 'ITL/Strong [CI]', 'TPOT/Strong [CI]',
                       'ITL/Common [CI]', 'TPOT/Common [CI]', '상대 PASS', '절대 PASS', '통합 PASS'], rows),
                 '', '## C3 Client Tail의 Host 경계', '',
                 '각 실행의 p99 이상 ITL 구간을 조사한다. 표는 같은 요청의 Worker Emission과 Client Delivery를 사용한다.',
                 '여러 Token의 동시 전달은 원래 0ms Client Interval을 유지한다.',
                 'Prefill, NIXL API 및 새 등록의 중첩 수는 서로 더할 수 있는 비용이 아니다.',
                 'Stream Wait의 GPU-exclusive 비중은 `NOT_MEASURED`다. 중첩만으로 개별 원인을 확정하지 않는다.', '',
                 table(['QPS','Start','Method','전체 ITL 구간','p99 이상 구간','Prefill Host 중첩',
                        'NIXL Host 중첩','새 D 등록 중첩','Tail Mean Worker Gap ms','Tail Mean Delivery 변화 ms'],tail_rows),
                 link('figures/c3_heldout_serving_v2_decode_tail_host_summary_v2.csv'),
                 '', '## C0의 모든-ITL 실패 위치', '',
                 'C0 Held-out의 고유 실행 전체에서 Client ITL>34ms 구간을 조사했다.',
                 'Native 실행은 여러 Cut에서 재사용되므로 한 번만 계산한다.',
                 'Client ITL = Worker Emission Gap + Delivery Delay 변화다.',
                 'Worker Gap≤34ms인 Client 위반은 Delivery Delay 증가에서 나타난다.',
                 '이 등식은 GC, Socket, CPU Scheduling 또는 File I/O 중 하나를 원인으로 확정하지 않는다.',
                 '모든 Client 위반은 원래 SLO 판정에 그대로 남긴다. Timestamp를 교정하지 않는다.', '',
                 table(['Arm', 'Output', '고유 실행', 'Requests', 'ITL 구간', '모든-ITL 실패 요청',
                        'Worker Gap≤34ms인 위반 구간', 'Worker Gap>34ms인 위반 구간'],
                       [[r['arm'],r['output'],r['independent_runs'],r['requests'],r['intervals'],
                         r['requests_failing_all_ITL'],r.get('worker_gap_at_most34',0),r.get('worker_gap_over34',0)]
                        for r in census.get('summary',[])]),
                 '표의 실행 수는 여러 Input/Cut 조건의 기술 통계다. 이를 한 Cell의 독립 Start 수로 사용하지 않는다.', '',
                 link('figures/c0_heldout_client_tail_violations_v2.csv'), '',
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


def compiler_report(final):
    controls=[]
    for plan_name in ('c3_policy_qualification_v2.json','c3_fused_teacher_context_supplement_v15.json'):
        for item in read('configs/'+plan_name,{}).get('items',[]):
            proof=read('state/'+item['gate'],{})
            controls.append([item['label'],proof.get('passed','NOT_RUN'),
                proof.get('strict',{}).get('passed') if proof.get('strict') else '독립 Native/State Gate',
                len(proof.get('stages',[])),item['gate']])
    fence=read('analysis/stock_pre_forward_fence_control_v14.json',{})
    knob=read('analysis/c3_knob_independent_confirmation_v2.json',{})
    absolute=read('analysis/c3_knob_absolute_and_contrast_tables_v2.json',{})
    compound=read('analysis/c3_knob_posthoc_compound_contrast_v2.json',{})
    host_rows=[]
    for stem,trace in (('c3_heldout_serving_v2','Natural EOS'),('c3_heldout_replay_v2','Fixed-length Replay')):
        host=read('analysis/'+stem+'_host_path_v2.json',{})
        host_rows += [[trace,r['left']+'/'+r['right'],r['paired_starts'],field,test(value)]
                      for r in host.get('comparisons',[]) for field,value in r['mean_host_delta_ms'].items()]
    return emit('dense_ar_longbench400_v2_compiler_and_admission_root_cause.md',
        'Compiler, Admission 및 정합성 원인 분석',[
        '## 수정 범위', '',
        '수정은 새 Source Capsule에만 적용했다. 기존 v1 Runtime과 Raw Evidence는 유지했다.',
        'P Partial Compiler는 고유 Compilation Config와 Cache를 사용한다.',
        'D Partial Compiler는 고정 Layer Interval을 별도 Graph로 실행한다.',
        '원래 Weight와 KV를 사용한다. Full36 Decode의 원래 Compiled 경로는 유지한다.',
        'Prompt의 CUDA Primitive Dispatch는 별도 Context로 제한한다.',
        'Assertion을 제거하거나 Output Tensor를 덮어쓰지 않았다.',
        'N8의 Strict State와 Natural32/400의 Coverage, Lifetime 및 품질 Gate를 모두 요구한다.',
        '계측하지 않은 Operator-exclusive 시간 절감은 주장하지 않는다.', '',
        '## Fused Hybrid Teacher의 Dispatch 오류', '',
        '처음 두 Fused 비교는 Strict State에서 실패했다. 이 실패를 그대로 보존했다.',
        '독립 Teacher의 필수 `teacher_eager_prefix_compiled_suffix` Flag가 빠진 것을 확인했다.',
        '새 Version은 이 Teacher Dispatch Flag만 복구했다. Candidate 모델 실행은 바꾸지 않았다.',
        '새 Teacher의 실제 실행 경로 Log를 요구했다. 기존 실패 Gate를 성공으로 변경하지 않았다.',
        'ContextFixV15의 새 N8/N32/N400 결과는 다음 표에 별도로 표시한다.', '',
        table(['정책 후보', '전체 Gate PASS', 'Strict8 PASS', 'Natural Stage 수', 'Gate 파일'],controls),
        '## Stock Token Strict 실패의 반증 검사', '',
        '실패한 Credit16 Probe의 Serial Strict 설정은 Coordinator Credit0을 사용했다.',
        '따라서 이 실패를 Credit16의 인과 효과라고 결론 내리지 않는다.',
        'P Prefix KV와 D Layer0 KV는 일치했다. 최초 D State 불일치는 Layer1에서 관측됐다.',
        'FX Graph는 같았다. 일부 Compiler Autotune Reduction 구성은 달랐다.',
        'R0_BLOCK2048/4096 차이는 관측 근거다. 이것만으로 불일치 원인을 확정하지 않는다.',
        '같은 Pair에서 Control → Pre-forward Fence → Control을 한 번씩 실행했다.',
        'No-fence Control도 통과하면 Fence가 오류를 해결했다는 근거가 아니다.',
        '기존 실패 Arm은 Primary 성능에서 제외한다. 통과할 때까지 반복하는 절차를 사용하지 않았다.', '',
        table(['고정 실행', '결과', 'Fence 활성화', '실제 Fence 수'],
              [[r['name'],r['status'],r.get('fence_enabled'),r.get('fence_count')] for r in fence.get('rows',[])]),
        link('analysis/stock_policy_strict_failure_audit_v13.json'), '',
        link('analysis/stock_compiler_nested_artifact_audit_v13.json'), '',
        link('analysis/stock_pre_forward_fence_control_v14.json'), '',
        '## Compiler/Credit/Admission 독립 확인', '',
        '다음 실험은 같은 192개 원문과 독립 C1 Output Length를 사용한 Fixed-length Replay다.',
        'P Compiler On/Off는 같은 Source와 D 경로를 사용한다.',
        'D Compiler On/Off도 같은 Source와 나머지 Flag를 유지한다.',
        'Credit8/16은 Source Outstanding Window를 바꾼다. 실제 GPU Batch를 직접 고정하지 않는다.',
        'Cap1/4는 같은 Admission Policy의 상한을 바꾼다. Queue와 Batch 변화는 그 총 효과의 일부다.',
        '결과는 Controlled Replay다. Natural EOS Serving 이득으로 표현하지 않는다.', '',
        table(['변경/대조', 'Paired Starts', 'TTFT p95 비율 [CI]', 'ITL p99 비율 [CI]',
               'TPOT p95 비율 [CI]', 'SLO-goodput 비율 [CI]', '변경 Arm Decode PASS', '대조 Arm Decode PASS'],
              [[r['left']+'/'+r['right'],r['paired_starts'],test(r['ratios']['TTFT_p95_ms']),
                test(r['ratios']['ITL_p99_ms']),test(r['ratios']['TPOT_p95_ms']),
                test(r['ratios']['SLO_qualified_output_tokens_per_second']),
                r['decode_gates'][r['left']]['qualified'],r['decode_gates'][r['right']]['qualified']]
               for r in knob.get('rows',[])]),
        '모든 Control은 QPS4.8의 과부하 Fixed-length Replay다. Sustainable Capacity 결과가 아니다.',
        '다음 표는 Goodput 비율의 실제 분모를 함께 보존한다.', '',
        table(['Arm','TTFT p95 Start-mean ms','SLO-goodput Start-mean Tokens/s'],
              [[label]+[number(next(r['start_mean'] for r in absolute['absolute'] if r['arm']==label and r['metric']==field))
                        for field in ('TTFT_p95_ms','SLO_qualified_output_tokens_per_second')]
               for label in sorted({r['arm'] for r in absolute.get('absolute',[])})]),
        '별도의 사후 Compound 비교는 D Compiler와 Short Fusion을 함께 바꾼다.',
        '이는 기존의 같은 N3를 재사용한다. 새 독립 실행 3회를 추가한 것이 아니다.',
        'Compound TTFT p95 비율: '+test(compound.get('TTFT_ratio'))+'.',
        'Compound SLO-goodput 비율: '+test(compound.get('goodput_ratio'))+'.', '',
        link('figures/c3_knob_TTFT_ratio_controls_v2.png'), '',
        link('figures/c3_knob_absolute_TTFT_goodput_v2.png'), '',
        link('figures/c3_knob_additive_host_path_v2.png'), '',
        '## Decode Tail의 시간 경계', '',
        'Client ITL은 Worker Emission Gap과 Delivery Delay 변화로 분해한다.',
        '같은 Notification에서 여러 Token을 전달하면 Co-delivery Interval0을 유지한다.',
        'Host NIXL 중첩과 Request 등록은 Timeline 위치다. 이를 GPU 지연의 인과적 비중으로 계산하지 않는다.',
        'NativeStrong과 Common-policy Native의 차이는 정책 Bundle의 총 효과다.',
        '이 비교만으로 Cap, Remote READ 또는 Registration 하나의 효과를 확정하지 않는다.',
        '## Held-out N3의 P/D 경로', '',
        '다음 차이는 같은 Request와 Arrival을 사용한 Start별 Mean Host 구간이다.',
        '두 Trace 모두 QPS2.0이다. Natural EOS에서는 실제 Output 작업량이 다를 수 있다.',
        'Fixed-length Replay는 요청별 독립 C1 길이를 그대로 사용한다. 두 Trace를 합쳐 독립 Start 수를 늘리지 않는다.',
        'P Prepare와 D Prepare는 Host 경계다. GPU Kernel 시작 시각이 아니다.',
        '모든 Stage의 합은 Mean TTFT 차이다. TTFT p95를 더해서 분해한 표가 아니다.', '',
        table(['Trace', '비교', 'Paired Starts', 'Host Stage', 'Mean 차이 ms [95% CI]'],host_rows),
        'Stage 차이는 지연이 나타난 위치를 설명한다. GPU 연산 또는 Credit 하나의 인과 효과를 확정하지 않는다.',
        'Layer와 Hybrid의 D Host 이득이 P 앞 Queue와 P Host 비용을 모두 상쇄하지 못할 수 있다.',
        'Credit 및 Compiler Intervention 결과는 위의 별도 Fixed-length Replay에서 확인한다.', '',
        link('figures/c3_heldout_serving_v2_host_path_v2_comparisons.csv'), '',
        '## 남은 한계', '',
        '일부 Stock Strict 불일치는 재실행에서 재현되지 않았다. 근본 원인은 미확정이다.',
        '적격 Operator-exclusive GEMM/Attention Breakdown은 별도 GPU Gate가 없으면 `NOT_MEASURED`다.',
        '최초 Geometry 선택 뒤 Compiler를 개선했다. 모든 Cut의 재튜닝은 `NOT_RUN`일 수 있다.',
        '따라서 Best는 테스트한 후보 집합 안의 선택이다. 전역 최적이라고 표현하지 않는다.',
    ],final)


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
        '90MiB 이상인 파생 Table은 원본을 로컬에 보존한다. GitHub에는 무손실 Archive와 SHA256 Manifest를 게시한다.',
        link('publication/derived_tables/index_v2.json'),
        'Snapshot 생성 시점의 파일 목록은 다음과 같다.', '',
        table(['Figure', 'PNG', 'PDF', '대응 CSV', '3개 형식 존재'],
              [[r['stem'], link(r['PNG']), link(r['PDF']) if r['PDF'] else 'NOT_RUN',
                link(r['CSV']) if r['CSV'] else 'NOT_RUN', r['complete']] for r in records]),
    ], final)


def summary(final):
    verdict = read('analysis/scientific_verdict_v2.json', {})
    freeze = read('data/balanced400_v2_freeze_manifest.json', {})
    lengths=[]
    reference=read('state/official_compiled_c1_reference_complete_v2.json',{})
    if reference:
        identities={r['prompt_id'] for r in read('data/longbench_balanced400_v2.json',[])}
        rows=[json.loads(s) for s in (ROOT/'raw'/reference['name']/'requests.jsonl').read_text().splitlines()]
        lengths=[r['output_tokens'] for r in rows if r['prompt_id'] in identities]
        assert len(lengths)==400
    stats=[]
    for label,key in (('Input Tokens','input'),('Selection Mode0 C1 Natural Output','output')):
        values=freeze.get(key,{})
        stats.append([label]+[number(values.get(field)) for field in ('mean','median','p10','p25','p75','p90','p95','p99','min','max')])
    if lengths:
        import numpy as np
        stats.append(['Original Compiled C1 Natural Output',number(np.mean(lengths)),number(np.median(lengths))]+
            [number(np.percentile(lengths,p)) for p in (10,25,75,90,95,99)]+[min(lengths),max(lengths)])
    main=read('analysis/c3_main_serving_v2_analysis.json',{})
    accounting=read('analysis/final_configuration_and_gate_inventory_v2.json',{})
    recalculation=read('analysis/raw_metric_recalculation_v2.json',{})
    closeout=read('state/resource_closeout_v2.json',{})
    archives=read('analysis/lossless_archive_integrity_v2.json',{})
    sections = ['## 연구 판정', '', verdict.get('decision', 'ACTIVE. 최종 판정 전이다.'), '',
        '## 실행 범위', '',
        'C0 Discovery80개와 Held-out80개 Cell을 측정했다. Held-out의 모든 Cell은 독립 Start3회다.',
        'Main400의 실제 완료 Start 수는 다음과 같다. 빈칸을 측정값으로 채우지 않는다.', '',
        table(['Offered QPS','Agg','NativeStrong','Layer','Stock Token','Hybrid','Deflection'],
              [[qps]+[next((r['independent_starts'] for r in main.get('cells',[]) if r['QPS']==qps and r['label']==label),'NOT_RUN')
                      for label in ('Agg','NativeStrong','Layer','StockToken','Hybrid','Deflection')]
               for qps in sorted({r['QPS'] for r in main.get('cells',[])})]),
        'QPS2.0의 별도 Held-out192도 Natural EOS N3를 확보했다.',
        'Equal-output Held-out Replay는 27개 계획 중 26개를 완료했다. Layer/Stock/Hybrid 및 그 두 Native 대조군은 N3다.',
        '마지막 Cap1 Native Replay와 선택적 단일 Role CPC는 예산 때문에 미실행으로 남겼다.',
        '모든 Method의 전체 QPS Grid와 과부하 N3 반복은 완료하지 못했다.',
        '신규 Dispatch는 시작 후 18시간에 종료했다. 이미 실행 중인 소유 작업은 정상 종료까지 기다렸다.', '',
        '## 검증 및 자원 복원', '',
        f"Configuration 실행 상태: `{accounting.get('status_counts',{})}`.",
        f"Raw Metric 재계산 상태: `{recalculation.get('status_counts',{})}`.",
        '실행 완료와 정합성·품질·Decode Gate 통과는 서로 다른 판정이다.',
        f"CPU 회귀 검사 PASS: `{read('analysis/cpu_final_regression_v2.json',{}).get('passed','NOT_RUN')}`.",
        f"승인된 GPU0–7 Burner 복원 및 소유권·사용률 확인: `{closeout.get('passed','NOT_RUN')}`.",
        '기존 사용자 Checkout과 Index를 보존했다. 다른 사용자 GPU 작업을 종료하지 않았다.', '',
        f"압축 해제 후 파일별 SHA256 검사 PASS: `{archives.get('passed','NOT_RUN')}`.",
        f"검사 Archive 수: {archives.get('archives','NOT_RUN')}. 검사 파일 수: {archives.get('files','NOT_RUN')}.",
        'Full Raw Log, Tensor 및 원문 Dataset Archive는 로컬에 보존한다.',
        'GitHub에는 재계산에 필요한 Structured Evidence, 정확한 Source Capsule 및 무손실 파생 Table을 게시한다.',
        '원문 Dataset은 공개 ID/Hash Roster와 공식 Source로 재구성한다. 원문의 개별 License를 유지한다.',
        link('analysis/lossless_archive_integrity_v2.json'), '',
        '## Dataset', '',
        f"선정 요청 수는 {freeze.get('n', 'NOT_RUN')}개다. Input 구간별 요청 수는 {freeze.get('input_totals', [])}다.",
        '처음 세 Input 구간은 각 Output Cell에 25개를 확보했다.',
        '20K–36K 구간의 긴 Output 후보는 부족했다. 원문을 수정하지 않고 동결된 재배분 규칙을 적용했다.', '',
        table(['Input', 'Output1–10', 'Output11–50', 'Output51–100', 'Output101–200'],
              [[label] + values for label, values in zip(('3K–6K', '6K–10K', '10K–20K', '20K–36K'), freeze.get('selected_cells', []))]),
        f"Dataset SHA256: `{freeze.get('SHA256', 'NOT_RUN')}`.",
        '',table(['측정 대상','Mean','Median','p10','p25','p75','p90','p95','p99','Min','Max'],stats),
        f'Compiled C1에서 200 Tokens를 초과한 요청은 {sum(v>200 for v in lengths)}/{len(lengths)}개다.',
        '선정 Manifest의 400개 요청은 400개 고유 Request ID와 287개 고유 Context를 포함한다.',
        '공유 Context 요청은 113개다. 요청 복제 또는 새 Padding/Truncation은 사용하지 않았다.',
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
                     'dense_ar_longbench400_v2_compiler_and_admission_root_cause.md',
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
    records = [function(final) for function in (c0_report, c3_report, compiler_report, safety_report, selector_report, figure_inventory, summary)]
    write(ROOT / 'state/report_generation_v2.json', dict(time=time.time(), final=final, files=records))
    preserve()
    print('V2_REPORTS_GENERATED', len(records), 'FINAL' if final else 'ACTIVE_SNAPSHOT', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    run(parser.parse_args().final)
