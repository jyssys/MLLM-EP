"""Write evidence-linked Korean reports without changing historical artifacts."""
import argparse
import collections
import json
import statistics
import time
from .common import ROOT,REPO,BASE,BRANCH,MODEL_REVISION,START,DEADLINE,write,preserve
from .publication import DOCUMENTS
from .causal_decomposition import SPAN_KEYS


def load(relative,default=None):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else default


def number(value,digits=3):
    return 'NOT_MEASURED' if value is None else f'{value:.{digits}f}'


def interval(value,digits=3):
    if not value:return 'CI 없음'
    return '['+number(value[0],digits)+', '+number(value[1],digits)+']'


def evidence_link(relative,label=None):
    return f'[{label or relative}]({ROOT.name}/{relative})'


def table(headers,rows):
    lines=['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
    return '\n'.join(lines)+'\n\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    if args.final:
        assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    preserve()
    stats=load('analysis/independent_start_statistics.json',dict(aggregate=[],comparisons=[],starts=[]))
    inventory=load('analysis/experiment_inventory.json',dict(counts={},runs=[]))
    causal=load('analysis/service_queue_causal_decomposition.json',{})
    model=load('analysis/frozen_model_validation_and_oracle.json',dict(results=[],confident_cell_forecasts=[]))
    features=load('analysis/recorded_observable_knob_validation.json',dict(heldout_cell_evaluations=[]))
    features_v3=load('analysis/recorded_observable_knob_validation_v3.json',dict(heldout_cell_evaluations=[]))
    features_v4=load('analysis/recorded_observable_knob_validation_v4.json',dict(heldout_cell_evaluations=[]))
    uncertainty=load('analysis/new_calibration_cost_rank_uncertainty.json',dict(differences={}))
    cross=load('analysis/cross_cut_restricted_oracle.json',dict(results=[],aggregate=[]))
    fine=load('analysis/fine_cohort_axis_oracle.json',dict(aggregate=[]))
    feedback=load('analysis/reservation_feedback_index.json',dict(runs=[]))
    control=load('analysis/control_boundary_replay_validation.json',dict(results=[],searches=[],aggregate=[]))
    operators=load('analysis/actual_serving_API_operator_analysis.json',{})
    environment=load('state/environment.json',{})
    manifest=load('analysis/figure_manifest.json',dict(figures=[]))
    state_gates=[json.loads(p.read_text()) for p in sorted((ROOT/'analysis').glob('*_state_gate.json'))]
    complete=len(stats['starts'])
    paired=[r for r in stats['comparisons'] if r['reference']=='Token' and r['candidate']=='Layer']
    confirmed=[r for r in paired if r['starts']>=3]
    native=[r for r in stats['comparisons'] if r['reference']=='Native' and r['starts']>=3]
    root=REPO/'poc_flashvep/reports'
    report_status=('CLOSED' if args.final else 'ACTIVE_DRAFT_NOT_FINAL')
    budget_hours=(time.time()-START)/3600
    provenance=f'''## 실행 계약과 증거 범위

상태: `{report_status}`. Base Evidence Commit은 `{BASE}`다.
새 Branch는 `{BRANCH}`다. 기존 Checkout과 Index는 변경하지 않는다.

- Model: `Qwen/Qwen3-8B`, Revision `{MODEL_REVISION}`.
- Runtime: vLLM `0.26.0+cu129`, PyTorch `2.11.0+cu129`, NIXL `1.4.1`.
- GPU: H100 SXM 80GB 두 대. P=GPU0, D=GPU1.
- GPU0 UUID: `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`.
- GPU1 UUID: `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`.
- Original BF16, Exact Causal Attention, Full36 Decode, Greedy, Thinking Disabled.
- Input8000. Output은 `ignore_eos=True`로 제어한다. GPU당 KV42GiB와 Prefix Cache Disabled를 유지한다.
- 모든 Worker의 Scheduler 상한은 B8192/S1024다. 이 값은 실제 GPU Batch가 아니다.
- Primary 실제 Batch는 P Prefill1, D Suffix1, D Decode1이다. Decode2/4 통제 실험은 별도로 표시한다.
- 성능 N36/N72는 9의 배수다. Layer2/4/8/12와 Token 인접16-Token 길이를 조합해 전체 GEMM Rows를 맞춘다.
- Prompt 원문 ID와 Input Token Hash를 보존한다. 새 Arrival Seed는 사용하지만 Prompt Distribution은 새로 분리하지 않았다.
- Common Balanced SLO는 TTFT≤5000ms, 모든 ITL≤34ms, Request TPOT≤33ms다. 기존 SLO 파일은 변경하지 않았다.
- 기존 실행 계약은 Runtime max_model_len을8000+max(Output cap,16)으로 설정한다. Output16/64/200에서8016/8064/8200이다.
- Client-visible Coordinator 수신 시각을 Primary로 사용한다. First ITL을 포함한다. Worker 시각은 별도 진단값이다.
- CI는 Paired Independent Server Start를 재표본한다. Request나 Token을 독립 Start로 세지 않는다.
- 신규 CI는 Bootstrap Draw의 Empirical CDF 분위수를 사용한다. n≤6은 전수 재표본한다. n3의27개 Draw를 선형 보간하지 않는다.

예산 시작은 2026-11-19 01:14:22 KST다. 예산 종료 목표는 11:14:22 KST다.
이 보고서 생성 시 경과시간은 {budget_hours:.2f}시간이다. 10:14:22 이후에는 새 GPU 실험을 시작하지 않는다.

{evidence_link('state/environment.json','GPU·환경 기록')},
{evidence_link('state/base_source_audit.json','Base Source 감사')},
{evidence_link('analysis/inherited_snapshot_integrity.json','상속 Python742개 Source 감사')},
{evidence_link('references/runtime_source_snapshot_manifest.json','설치 Runtime 원문 Snapshot')},
{evidence_link('data/pool_contract.json','Dataset 계약')},
{evidence_link('analysis/per_start_client_metrics.csv','전체 Start별 Metric CSV')}를 보존한다.
실험별 Config, Command, Source ZIP, Prompt Hash, Arrival Trace와 Worker Log는 별도 Raw Directory에 있다.
GitHub에서는 큰 원시파일을 검증된 Lossless 분할 Archive로 제공한다. 원본은 로컬에 유지한다.
{evidence_link('archives/large_evidence_manifest.json','Archive 복원·SHA256 Manifest')}에 각 파일의 Hash와 복원 방법을 기록한다.
'''
    limits='''
## 해석 제한

Host Service Enclosure에는 Launch, Scheduling, Transfer와 통지 비용이 들어간다.
이를 Exclusive GPU Kernel Time이나 물리적 GPU Idle로 해석하지 않는다.
Finite-cohort Drain Token/s는 Sustainable Capacity가 아니다.
과부하 실행의 미완료 요청은 제거하지 않는다. 모든 요청을 Drain까지 포함한다.
평균 구간의 합은 Mean TTFT의 정확한 항등식이다. 구간별 p95를 더하지 않는다.
단일 Start의 우위에는 CI를 붙이지 않는다. 여러 Cell의 CI는 Pointwise이며 Multiplicity 보정은 없다.
기존 결과와 신규 결과는 별도로 표시한다. Oracle은 실제 Online Speedup이 아니다.
현재 Token 기준선은 Completed-prefix KV 이후 Suffix를 실행한다. 모든 Streaming Token Pipeline과 비교한 결과는 아니다.
Whole-cohort Guard와 Reservation ACK 경로는 현재 Runtime의 계약이다. Dense AR의 구조적 필수 제약으로 일반화하지 않는다.
'''
    count_text=table(['Evidence Status','실험 수'],sorted(inventory['counts'].items()))
    state_text=f'''## 정합성

신규 State Gate는 {len(state_gates)}개다. 통과 수는 {sum(bool(r['passed']) for r in state_gates)}개다.
N8→N32에서 Original Greedy Output, Full Prompt KV, Final Hidden, Layer/Query 방문과 Lifetime을 검사한다.
Token Prefix/Suffix Hidden은 별도로 검사한다. ACK와 Credit Retirement도 확인한다.
State 복사 실행은 `CORRECTNESS_ONLY`다. 해당 실행의 Latency를 성능 Table에 넣지 않는다.
신규 성능 실행마다 Output, Actual Query Coverage, Matched Rows, Worker B/S와 Reservation Retirement를 다시 검사한다.
Decode2는 Base의 동결된 v3 Adapter State 계약도 재사용한다. Decode4는 신규 State Gate를 먼저 통과해야 한다.
N8/N32는 정확한 Aggregate GEMM Rows 비교가 아니다. 성능 N36/N72에서 정확한 Rows를 검사한다.
'''
    clean_rows=[]
    for row in stats['aggregate']:
        m=row['metrics'];g=row['common_Balanced']
        clean_rows.append([row['stage'],row['method']+str(row['layers']),row['output'],row['QPS'],
            row['count'],f"1/{0 if row['method']=='Native' else 1}/{row['batch']}",row['window'],row['starts'],
            number(m['TTFT_p95_ms']['mean']/1000),number(m['ITL_p99_ms']['mean']),
            number(m['TPOT_p95_ms']['mean']),number(m['output_tokens_s']['mean']),
            number(g['Output_token_SLO_goodput']['mean']),number(g['Joint_attainment']['mean'])])
    comparison_rows=[]
    for row in paired:
        v=row['metrics']['TTFT_p95_ms'];g=row['common_Balanced_goodput']
        comparison_rows.append([row['stage'],row['layers'],row['output'],row['QPS'],row['batch'],row['window'],
            row['starts'],number(v['ratio']),interval(v['ratio_CI95']),number(g['ratio']),interval(g['ratio_CI95'])])
    safety_rows=[]
    for r in native:
        safety_rows.append([r['stage'],r['candidate']+str(r['layers']),r['output'],r['QPS'],r['batch'],r['starts'],
            number(r['metrics']['ITL_p99_ms']['ratio']),interval(r['metrics']['ITL_p99_ms']['ratio_CI95']),
            number(r['metrics']['TPOT_p95_ms']['ratio']),interval(r['metrics']['TPOT_p95_ms']['ratio_CI95']),r['Native_relative_5pct']])
    online='# Dense AR 실제 Batch-matched Online 결과\n\n'+provenance+state_text+'\n## 실행 Matrix\n\n'+count_text+f'''
신규 Clean Start는 {complete}개다. 모든 실행 상태는 {evidence_link('analysis/experiment_inventory.json')}에 있다.
`NOT_RUN` Cell은 보간하지 않는다. Serial Observer Control은 Poisson Primary에서 제외한다.
`MEASURED_ONLINE`의 의미는 Client-visible 실측이다. 물리적 GPU 계산시간 측정을 뜻하지 않는다.

## 신규 실측 전체 Cell

단위는 TTFT초, ITL/TPOT밀리초, 처리량Output tokens/s다. Joint는 요청 비율이다.
Actual Batch 열은 P Prefill / D Prefill / D Decode를 뜻한다.
Native에는 D Prompt Prefill이 없으므로 가운데 값은0이다. 공통 준비 정책의 상한은1이다.

'''+table(['Stage','Arm','Output','QPS','N','Batch','WIP','Starts','TTFT p95(s)','ITL p99(ms)','TPOT p95(ms)','Raw token/s','Qualified token/s','Joint'],clean_rows)+'''
## Layer / Token Paired Ratio

TTFT Ratio<1이면 Layer가 빠르다. Goodput Ratio>1이면 Layer가 높다.
Output, QPS, 요청 수, Batch, Window와 Arrival Seed를 일치시킨다.
CI가1을 포함하는 Cell에는 확정 승자 Label을 붙이지 않는다.

'''+table(['Stage','Cut','Output','QPS','Decode Batch','WIP','Starts','TTFT Ratio','95% CI','Goodput Ratio','95% CI'],comparison_rows)+'''
## Native-relative Decode5% 검사

ITL p99와 TPOT p95 Ratio의 95% CI 상한이 모두1.05 이하여야 PASS다.
Absolute SLO Pass와 이 검사를 구분한다.

'''+table(['Stage','Arm','Output','QPS','Batch','Starts','ITL Ratio','95% CI','TPOT Ratio','95% CI','판정'],safety_rows)+f'''
## Queue 상태

{evidence_link('analysis/measurement_export_index.json')}와 Start별 Measurement Export에 실제 Arrival QPS를 기록한다.
Warmup을 제외하고 첫 Arrival부터 마지막 Completion까지 모든 요청을 포함한다.
Arrival Window의 Unfinished 증가율과 Drain 시간을 별도로 기록한다.
`FINITE_WINDOW_NO_STRONG_GROWTH`는 장기 안정성을 입증하는 Label이 아니다.
신규 Main Batch1 O16은 N72다. O64/200 Discovery는 N36이다.
이 두 집합을 Output만 바꾼 인과 실험으로 사용하지 않는다.
순수 Output 비교는 별도 Decode2/N72/Output16·64 Paired Seed 실험이다.
여기서 Output cap을 바꾸면 파생된 Runtime Context 상한과 KV Reservation 크기도 바뀐다.
따라서 이 실험은 Output cap의 전체 효과를 검사한다. Decode Residency만의 독립 효과로 해석하지 않는다.
'''+limits

    causal_rows=[]
    for r in causal.get('historical_mean_identities',[]):
        spans=r['mean_spans_ms']
        causal_rows.append([r['layers'],r['starts'],number(r['mean_TTFT_difference_ms']),
            *[number(spans[k]['mean']) for k in ('arrival_to_P_host_submit','P_submit_to_source_notification',
                'source_notification_to_D_host_submit','D_submit_to_first_worker_notice','first_worker_notice_to_client')]])
    window_rows=[]
    for r in causal.get('source_window_independent_start_summary',[]):
        d=r['mean_delta_ms'];rat=r['outcome_ratios']['TTFT_p95_ratio_W4_over_W32']
        window_rows.append([r['method']+str(r['layers']),r['starts'],number(d['Ready_D_mean_ms']['mean']),
            interval(d['Ready_D_mean_ms']['CI95']),number(d['P_external_wait_mean_ms']['mean']),
            number(d['P_submit_to_ready_mean_ms']['mean']),number(d['D_host_slot_per_request_ms']['mean']),
            number(rat['mean']),interval(rat['CI95'])])
    operator_rows=[[r['layers'],number(r['Token_over_Layer_D_projection_model']),
        number(r['Layer_D_projection_enclosure_model_ms']),number(r['Token_D_projection_enclosure_model_ms'])]
        for r in operators.get('matched_projection_models',[])]
    amplification=[]
    for m in (4,8,12):
        rows=[r for r in causal.get('historical_queue_amplification_proxy',[]) if r['layers']==m]
        if rows:
            amplification.append([m,len(rows),number(statistics.mean(r['Layer_minus_Token_cycle_ms_per_request'] for r in rows)),
                number(statistics.mean(r['cycle_delta_times_tail_rank_ms'] for r in rows)),
                number(statistics.mean(r['measured_Layer_minus_Token_TTFT_p95_ms'] for r in rows))])
    feedback_rows=[]
    for r in feedback.get('runs',[]):
        if r['stage']!='decode_batch_counterfactual' or r['method'] not in ('Layer','Token'):continue
        path=ROOT/'analysis'/(r['name']+'_service_queue_audit.json')
        service=json.loads(path.read_text())
        gap=r['mean_backlogged_nonwindow_P_gap_ms']
        if gap is None:continue
        p=service['P_submit_to_ready_mean_ms']+gap;d=service['D_host_slot_per_request_ms']
        feedback_rows.append([r['seed'],r['method'],r['batch'],number(service['P_submit_to_ready_mean_ms']),
            number(gap),number(r['mean_backlogged_nonwindow_gap_parts_ms']['reservation_RTT_ms']),
            number(p),number(d),'P' if p>d else 'D'])
    new_identity_rows=[]
    for r in causal.get('new_mean_identity_summary',[]):
        if r['starts']<3 or r['stage'] not in ('output_residency_counterfactual','decode_batch_counterfactual','batch4_knob_confirmation','heldout_confirmation','new_QPS_confirmation'):
            continue
        spans=r['mean_spans_ms']
        new_identity_rows.append([r['stage'],r['layers'],r['output'],r['QPS'],r['batch'],r['starts'],
            number(r['mean_TTFT_difference_ms']),*[number(spans[k]['mean']) for k in SPAN_KEYS]])
    output_cost_rows=[]
    for r in causal.get('output_cap_intervention',{}).get('summary',[]):
        values=r['mean_delta_ms']
        output_cost_rows.append([r['method'],r['starts'],
            *[number(values[k]['mean']) for k in ('P_service_ms','P_backlogged_ACK_ms','D_preparation_per_request_ms',
                'D_decode_residence_per_request_ms','D_guard_per_request_ms','Ready_D_wait_ms','P_external_wait_ms')]])
    causal_report='# GPU Service와 Host Queue의 원인 분리\n\n'+provenance+'''
## 핵심 관측

기존 Last8 Token Ready→D 대기 증가의 대부분은 이전 Cohort Guard 구간과 겹친다.
이 Guard는 D Host Submit부터 마지막 Client Completion까지다.
Transfer, Suffix 준비, Full36 Decode와 Delivery를 모두 포함한다.
따라서 이 값만으로 D GPU Compute Saturation을 주장할 수 없다.
기존 해당18개 Confirmation Start에서 Reservation Reject와 공통 Credit Retry는0건이었다.
이는 해당 실행의 주된 대기가 Credit Retry가 아니었다는 근거다. 모든 실패 상황을 배제하지는 않는다.

## 기존 Mean TTFT의 정확한 차이 분해

모든 값은 Layer−Token 밀리초다. P Queue에는 Admission/Host 대기도 들어간다.
각 행의 다섯 구간 합은 Mean TTFT 차이와 같다. 이는 p95 분해가 아니다.

'''+table(['Cut','Starts','Mean TTFT 차이','Arrival→P Submit','P Submit→Ready','Ready→D Submit','D Submit→First Worker','Worker→Client'],causal_rows)+'''
Last4는 해당 Decode2 Protocol에서 P 단계 부담이 상대적으로 컸다.
Last8/12는 D Guard 부담이 커졌다. 이때 작은 D Service 차이가 반복되어 Queue에 누적될 수 있다.
단계별 항등식만으로 독립적인 인과 효과를 확정하지 않는다.
Stage-balance Replay와 아래 Intervention으로 추가 검사한다.

## 작은 Service 차이의 Queue 증폭

같은 완료 Trace에서 유효 Cycle을 `max(P Service+Gap,D Guard/request+Gap)`로 계산했다.
두 방식의 Cycle 차이에 `0.95×(N−1)`을 곱한 값과 실측 TTFT p95 차이를 비교한다.
모든 차이의 부호는 Layer−Token이다.
다음 계산은 사후 설명용 Deterministic Pipeline 근사다. 새로운 Held-out 예측이나 M/M/1 공식이 아니다.

'''+table(['Cut','Starts','유효 Cycle 차이(ms/request)','Cycle×Tail Rank 근사(ms)','실측 TTFT p95 차이(ms)'],amplification)+'''
서비스 차이는 몇~수십 ms/request지만, 과부하 Cohort에서 반복되어 초 단위 Tail 차이가 될 수 있다.
같은 Trace에서 추정한 비용이므로 이 수치 일치만으로 인과관계를 확정하지 않는다.
별도 Discovery→Confirmation Replay와 신규 Batch/Output Intervention을 함께 본다.

## 신규 반복 실측의 Mean TTFT 차이

모든 차이는 Layer−Token 밀리초다. 다섯 구간의 합은 Mean TTFT 차이다.
같은 Batch, Input, Output, QPS, Window와 Arrival Seed를 사용한다.
Output별 상대 우위가 어느 구간에서 바뀌는지 확인한다. 이를 Exclusive Kernel Time 분해로 해석하지 않는다.

'''+table(['Stage','Cut','Output','QPS','Decode Batch','Starts','Mean TTFT 차이',
    'Arrival→P','P→Ready','Ready→D','D→First Worker','Worker→Client'],new_identity_rows)+'''
## Output16→64의 공통 통제 실험

Last4, N72, QPS6, 실제 P1/D Suffix1/Decode2를 유지한다. 모든 차이는 Output64−Output16이다.
Runtime Context 상한은8016→8064로 파생된다. 따라서 Output cap의 전체 효과를 검사한다.
Host에서 관측한 준비 비용, Decode Residency, ACK와 대기를 분리한다.
Start별 값과 Paired CI는 별도 JSON에 보존한다.

'''+table(['Axis','Starts','P Service Δ(ms)','ACK Δ(ms)','D 준비/request Δ(ms)',
    'D Decode/request Δ(ms)','D Guard/request Δ(ms)','Ready→D Δ(ms)','P 앞 대기 Δ(ms)'],output_cost_rows)+'''
Output16/64에서 전체 GEMM Rows와 Partition은 같다. Decode Work와 KV Lifetime은 달라진다.
Host Decode Residency 증가를 GPU Kernel Time 증가와 동일시하지 않는다.

## Source Outstanding Window32→4

P1, D Suffix1, Decode2와 Partition은 유지한다. Source Outstanding 상한만 변경한다.
Window는 Scheduler B/S나 GPU Batch가 아니다.
다음 값은 W4−W32다. TTFT는 W4/W32 Ratio다.

'''+table(['Arm','Paired Starts','Ready→D Δ(ms)','95% CI','P 앞 대기 Δ(ms)','P Service Δ(ms)','D Guard/request Δ(ms)','TTFT p95 Ratio','95% CI'],window_rows)+'''
대기가 D에서 P 앞으로 이동할 수 있다. 대기 위치의 감소를 E2E 성능 향상으로 해석하지 않는다.
Source Window 변화는 Backpressure와 실제 Host Service에도 영향을 줄 수 있다.
서비스 비용이 달라진 Start는 순수 Queue Relocation 실험으로 간주하지 않는다.
Last12의 추가 Seed1950/1951은 Treatment 순서도 바꿨다.

## P 입장 경로의 Reservation ACK Feedback

모든 P Prefix는 D Reservation ACK를 받은 뒤 제출된다.
D Worker는 `engine.step()` 사이에서 Control Message를 처리한다.
따라서 Credit 부족이 없어도 진행 중인 D Step 때문에 ACK가 늦을 수 있다.
실제 GPU 계산과 별개인 이 필수 의존성이 P Critical Path에 들어간다.
요청이 이미 도착했고 Source Window가 막히지 않은 전환에서 P Gap을 세 구간으로 분리했다.
구간은 이전 P Ready→Reservation 시작, Reservation RTT, ACK 수신→다음 P Submit이다.
세 구간 합은 실제 P Gap과 정확히 같다. 이를 GPU Kernel Time으로 부르지 않는다.

'''+table(['Seed','Axis','Decode Batch','P Submit→Ready(ms)','P Gap(ms)','그중 ACK(ms)','유효 P Demand(ms/request)','D Guard(ms/request)','큰 단계'],feedback_rows)+f'''
{evidence_link('analysis/reservation_feedback_index.json')}에 Request별 구간과 D Host Phase별 RTT를 기록했다.
Host Phase는 Idle/Preparation/Decode 구분이다. 실제 Kernel 실행 구간을 직접 측정한 Label은 아니다.
기존 비용표의 Median Gap과50ms 제외 규칙은 이 Bimodal Control 대기를 줄여 추정할 수 있다.
아래 v4 검증은 신규 Calibration에서 비차단 Backlog Gap의 Mean을 사용한다.
이는 GPU WCET나 안전한 Deadline Upper Bound를 뜻하지 않는다.

## 실제 Serving Linear API

`linear_batch_invariant`와 실제 Checkpoint Weight를 사용했다.
기존 `F.linear` 고립 실험과 구분한다. Hopper의 실제 Serving은 Persistent Matmul Kernel을 사용한다.
다음 값은 네 Projection의 고립 Median 합에 Layer 수를 적용한 모델이다.
Attention, RMSNorm, Transfer와 Launch 간 대기를 포함한 전체 Forward 측정값이 아니다.
고립 실험은 Layer0 Weight를 반복한다. 실제 Dense36의 서로 다른 Weight Working Set과 Cache 상태는 복원하지 않는다.

'''+table(['Cut','Token/Layer Projection 모델','Layer D 모델(ms)','Token D 모델(ms)'],operator_rows)+f'''
Evidence는 `STRUCTURAL_ACCOUNTING`이다. 입력값은 `ISOLATED_KERNEL_MEASUREMENT`의 별도 Operator Timing이다.
Profiler Gate를 통과한 Kernel Profile은 {len(operators.get('qualified_kernel_profiles',[]))}/72개다.
나머지8개는 작은 O Projection Shape의 `DIAGNOSTIC`이다.
Clean Timing과 Observer Profile을 섞어 Online Exclusive GEMM 시간을 만들지 않는다.

온라인 Event Observer의 네 Cell은 모두5% Gate를 실패했다.
추가 Serial Last4 A/B/A에서는 Layer가 통과했고 Token은 TTFT Gate를 실패했다.
따라서 직접적인 양측 Online Exclusive GPU Breakdown은 `NOT_MEASURED`다.
Serial 실행은 Poisson Serving 결과가 아니다. 이 결과 이후 Profiling을 더 확장하지 않았다.

## 정량적 구조와 한계

동일 GEMM Rows에서도 Token D Attention Pair는 Layer보다 많다.
Token/Layer D Attention Pair Ratio는 Last2/4/8/12에서 약1.9443/1.8888/1.7777/1.6666이다.
전체 P+D Attention Pair는 같다. Attention GPU Time Ratio는 측정하지 않았다.
Matched Aggregate에서 유효 KV Payload는 같다. Layer는 Request당 Hidden+Residual131,072,000bytes를 추가 전송한다.
Layer D는 선택한 m개 Layer의 Weight를 사용한다. Token D는36개 Layer의 서로 다른 Weight를 사용한다.
고유 Projection Weight Footprint는 구조적으로 계산했다. 실제 HBM Read Traffic으로 해석하지 않는다.
실제 Transfer Telemetry는 Start별 Metrics에 있다. API 완료시간을 순수 DMA 시간으로 해석하지 않는다.
SM/Tensor Core Counter, HBM Traffic와 Physical Kernel Idle은 `NOT_MEASURED`다.

{evidence_link('analysis/service_queue_causal_decomposition.json')},
{evidence_link('analysis/slot_decomposition_index.json')},
{evidence_link('analysis/actual_serving_API_operator_analysis.json')},
{evidence_link('analysis/matched_geometry_structural_accounting.json')}를 함께 참조한다.
'''+limits

    classification=features.get('heldout_cell_evaluations',[])
    feature_rows=[]
    for r in classification:
        feature_rows.append([r['stage'],r['layers'],r['output'],r['QPS'],r['batch'],r['starts'],
            r['confidence_label'],r['majority_fixed_axis_forecast']['one'],r['majority_fixed_axis_forecast']['two'],
            number(r['fixed_axis_TTFT_regret']['one']*100),number(r['fixed_axis_TTFT_regret']['two']*100)])
    rule_summary=[]
    for rule in ('one','two'):
        resolved=[r for r in classification if r['correct'][rule] is not None]
        rule_summary.append([rule,len(resolved),sum(bool(r['correct'][rule]) for r in resolved),
            number(statistics.mean(r['fixed_axis_TTFT_regret'][rule] for r in classification)*100) if classification else 'NOT_RUN'])
    for rule in ('AlwaysLayer','AlwaysToken','StaticMetadataCycle'):
        resolved=[r for r in classification if r.get('baseline_correct',{}).get(rule) is not None]
        rule_summary.append([rule,len(resolved),sum(bool(r['baseline_correct'][rule]) for r in resolved),
            number(statistics.mean(r['baseline_TTFT_regret'][rule] for r in classification)*100)
            if classification and all('baseline_TTFT_regret' in r for r in classification) else 'NOT_RUN'])
    v3_rows=[]
    for version,data in ((3,features_v3),(4,features_v4)):
        for r in data.get('heldout_cell_evaluations',[]):
            v3_rows.append([version,r['QPS'],r['batch'],r['starts'],r['confidence_label'],
                r['majority_fixed_axis_forecast']['one'],r['majority_fixed_axis_forecast']['two'],
                number(r['fixed_axis_TTFT_regret']['one']*100),number(r['fixed_axis_TTFT_regret']['two']*100),
                r.get('confidence_goodput_label','NOT_MEASURED')])
    uncertainty_rows=[[field,number(r['mean_ms']),interval(r['paired_start_CI95_ms'])]
        for field,r in uncertainty['differences'].items()]
    knob='# Minimal Knob Discovery와 Held-out 검사\n\n'+provenance+'''
## 단일 변수 후보: P/D Pressure Ratio

`R=(1+readyD/b+calibrated_active_D_residual/reference_D_cost)/(1+pendingP)`로 정의한다.
행동 비용은 동결된 Calibration Metadata다. R은 여러 Host Counter를 합친 하나의 Scalar다.
하나의 Sensor만 사용한다는 뜻은 아니다.
기존 Discovery33개 Cell에서 사전 후보 κ0.25/0.5/1/2/4를 비교해 κ4를 동결했다.
이 Training 결과를 Held-out Accuracy로 표현하지 않는다.
R은 대기가 P/D 사이에서 이동하면 달라진다. Queue 위치가 바뀌어도 실제 병목이 같을 수 있다는 반례가 있다.

## 두 변수 후보: Decode Demand와 Total Backlog

첫 변수 `d`는 Calibration Full36 Decode Cohort Residency를 실제 Decode Batch로 나눈 값이다.
두 번째 변수 `n`은 이미 도착했지만 끝나지 않은 요청 수다.
Current P Request, P Pending, D Ready와 Active D를 센다. 미래 Arrival이나 실제 미래 Completion을 읽지 않는다.

각 행동 a의 동결 비용을 `p_a=P Service+P Gap`, `u_a=D Preparation+D Gap/b`로 정의한다.
혼잡 상태의 단순 Cycle 모델은 `C_a=max(p_a,u_a+d)`다.
비혼잡 First-path 모델은 `F_a=p_a+u_a`다.
`α=clip((n-b)/(b+1),0,1)`로 두 값을 연결한다.
점수는 `αC_a+(1−α)F_a`다. 이 함수와 비용 학습 범위는 새 Batch/Output 결과 전에 고정했다.

Token의 p가 낮고 Layer의 u가 낮을 때 모델 경계는 `d*=p_Layer−u_Token`이다.
이는 단순 Max-stage 모델 안의 경계다. 실제 Online TTFT의 정리나 보장된 Deadline이 아니다.
새 Calibration에서는 일부 Cut의 P 우위 부호도 바뀌었다.
차이가1ms 수준이면 Run 간 Host 변동과 구분하기 어렵다. 유리한 부호를 사후에 선택하지 않는다.

`d`는 현재 GPU Idle이 아니다. 이 Runtime의 Whole-cohort Guard Service Demand를 나타낸다.
고정 Output/Batch에서는 d가 거의 고정되므로, 이 변수만으로 Within-workload Dynamic 이득을 주장할 수 없다.

## 더 단순한 비용 균형 Scalar

동결한 Static Metadata Cycle 기준선은 `ΔC(d)=max(p_L,u_L+d)−max(p_T,u_T+d)`로 해석할 수 있다.
ΔC<0이면 Layer, ΔC>0이면 Token을 예측한다. Threshold는0이다.
행동 비용은 Calibration Lookup이다. d는 알려진 Output cap과 실제 Decode Batch에서 추정한 공통 Decode Demand다.
이 Scalar는 현재 GPU Slack을 측정한 값이 아니다. 고정 Workload Metadata와 Calibration에 의존한다.
또한 같은 동질 Workload 안에서 요청별 Axis를 바꾸지 않는다.
따라서 이 기준선의 성공은 간단한 Cross-workload 비용 설명을 지지할 수 있다. State-aware Dynamic의 독립 기여를 입증하지는 않는다.
아래 Table에서 원래 두 규칙과 같은 Held-out Cell을 비교한다.

## 기록된 실제 의사결정 상태의 검증

기존 Static 실행의 P 제출 전 상태에서 두 규칙의 Shadow Vote를 계산한다.
이 Vote는 실제 실행 Axis를 바꾸지 않았다.
Cell 전체의 다수 Vote를 하나의 Fixed-axis 예측으로 평가한다.
Request별 Vote 수를 독립 검증 표본 수로 사용하지 않는다.
CI가1을 포함하는 Cell은 `UNRESOLVED`로 둔다.
Always Layer, Always Token과 Static Metadata Cycle 규칙도 같은 Cell에서 검사한다.
다수 Cell의 승자가 같다는 이유만으로 State-aware 규칙의 기여를 주장하지 않는다.
Static Metadata 규칙은 같은 동질 Workload 안에서 상태에 따라 Axis를 바꾸지 않는다.

'''+table(['Rule','확정 Label Cell','정답 Cell','평균 Fixed-axis TTFT Regret(%)'],rule_summary)+table(
        ['Stage','Cut','Output','QPS','Batch','Starts','측정 Label','One 예측','Two 예측','One Regret(%)','Two Regret(%)'],feature_rows)+f'''
Regret은 예측된 Fixed Axis와 측정된 Paired Cell 평균을 비교한다.
이를 Online Axis Switching의 성능 손실로 해석하지 않는다.
State-updating Rule Replay는 별도 Oracle 보고서에 있다.
Regret은 실측 Cell 평균에 대한 점 추정치다. 미확정 Cell의 작은 Regret은 입증된 손실이 아니다.
Qualified Token/s의 승자와 Regret도 별도 JSON에 기록했다. 모든 방법의 Goodput이0이면 비율 손실은 정의하지 않는다.
Prompt는 재사용했다. 새 Arrival Seed/QPS/Output/Batch 검증과 새로운 Prompt 분포 검증은 다르다.
Held-out Prompt Distribution과 Input Length Generalization은 `NOT_RUN`이다.
1~2변수가 모든 현상을 설명한다는 주장은 아직 별도 반례와 모델 검증을 필요로 한다.

{evidence_link('analysis/recorded_observable_knob_validation.json')},
{evidence_link('configs/frozen_single_pressure_rule_v1.json')},
{evidence_link('configs/frozen_action_service_costs_v2.json')},
{evidence_link('state/minimal_knob_protocol_v2_before_new_batch_output.json')}에 정의와 입력을 보존한다.
'''+'''
## 비용표 외삽 실패와 별도 v3 검증

v2는 Decode4에서 기존 Decode2의 P/Suffix 비용을 재사용했다.
첫 두 새 Seed에서는 Token P Service가 예측보다 약12ms 짧았다. 이 오차가 P 단계 순위를 바꿨다.
따라서 비용표 외삽 실패와 상태 변수의 실패를 분리하는 추가 실험을 선언했다.
Batch4 새 Calibration6개 Start에서만 별도 v3 비용표를 동결한다.
QPS3/6의 새 Seed1970/1971/1972를 그 이후에 평가한다.
규칙 함수, κ와 SLO는 바꾸지 않는다. v2의 실패 결과도 유지한다.
두 비용표가 같은 신규 Raw Trace를 예측한다. 이를 서로 다른 독립 Server 표본으로 더하지 않는다.
추가 v4는 같은 신규 Calibration만 사용한다. P Gap을 Median/50ms 절단값 대신 비차단 Backlog 전환의 Mean으로 추정한다.
이 변경은 신규 Calibration 전에 별도로 선언했다. v2/v3 파일은 변경하지 않았다.
v4는 새 Confirmation의 첫 측정 Arrival 전에 동결됐는지 Monotonic Clock으로 검사한다.
새 비용표들은 동일한 두 규칙을 평가한다. 이 이후에는 규칙이나 비용표를 다시 튜닝하지 않는다.

'''+table(['Cost Model','QPS','Batch','Starts','TTFT Label','One 예측','Two 예측','One TTFT Regret(%)','Two TTFT Regret(%)','Goodput Label'],v3_rows)+f'''
{evidence_link('analysis/recorded_observable_knob_validation_v3.json')},
{evidence_link('analysis/frozen_model_step_and_completion_errors_v3.json')},
{evidence_link('configs/frozen_action_service_costs_v3.json')}는 실행된 경우에만 존재한다.
v4의 대응 파일은 동일한 이름의 `_v4`/`v4` 파일이다.
Budget 때문에 해당 Phase가 미실행이면 `NOT_RUN`이다.

## Calibration 비용 순위의 불확실성

아래 값은 새 Batch4 Calibration의 Paired Seed3개만 사용한다. 부호는 Layer−Token이다.
P Gap은 이미 도착했고 Source Window가 막히지 않은 전환의 Mean이다.
각 비용의 별도 Median으로 만든 Lookup과 Paired Mean 차이는 서로 다를 수 있다.
CI가0을 포함하면 그 비용 순위를 확정하지 않는다. 이 분석으로 동결 모델이나 규칙을 다시 맞추지 않는다.

'''+table(['비용 항목','Paired Mean 차이(ms)','95% CI(ms)'],uncertainty_rows)+f'''
{evidence_link('analysis/new_calibration_cost_rank_uncertainty.json')}에 각 Start의 값과 출처를 보존한다.
이는 비용 Calibration의 불확실성이다. Held-out 예측 성공이나 GPU Kernel 분산을 뜻하지 않는다.
'''+limits

    oracle_rows=[]
    for r in model.get('results',[]):
        v=r['fixed_trace_validation'];best=r['oracle']['best_fixed'];o=r['oracle']['oracle']
        base=r['oracle']['fixed'][best]
        oracle_rows.append([r['stage'],r['layers'],r['output'],r['QPS'],r['batch'],r['seed'],
            r['qualified_for_counterfactual_interpretation'],best,
            number(base['qualified_output_tokens_s']),number(o['qualified_output_tokens_s']),
            number(r['modeled_SLO_headroom_over_best_same_cut_fixed']*100) if r['modeled_SLO_headroom_over_best_same_cut_fixed'] is not None else 'UNDEFINED'])
    global_rows=[]
    for r in cross.get('aggregate',[]):
        metric=r['predicted_goodput_oracle_over_frozen_fixed']
        strongest=r['predicted_goodput_oracle_over_per_trace_best_fixed']
        global_rows.append([r['QPS'],r['starts'],number(metric['ratio']),interval(metric['ratio_CI95']),
            number(strongest['ratio']),interval(strongest['ratio_CI95']),r['all_starts_all_actions_validated']])
    fine_rows=[]
    for r in fine.get('aggregate',[]):
        ratio=r['predicted_goodput_ratios']['oracle']
        fine_rows.append([r['model_version'],r['stage'],r['layers'],r['output'],r['QPS'],r['batch'],r['starts'],
            number(ratio['ratio']),interval(ratio['ratio_CI95']),r['all_fixed_validation_passed'],
            number(r['predicted_goodput_ratios']['one']['ratio']),number(r['predicted_goodput_ratios']['two']['ratio'])])
    validated_fine=[r for r in fine.get('aggregate',[]) if r['all_fixed_validation_passed'] and r['starts']>=3]
    validated_fine_rows=[]
    for r in validated_fine:
        v=r['predicted_goodput_ratios']['oracle'];ci=v['ratio_CI95']
        validated_fine_rows.append([r['stage'],r['layers'],r['QPS'],r['batch'],r['starts'],
            number((v['ratio']-1)*100,5),interval([(x-1)*100 for x in ci],5) if ci else 'CI 없음'])
    maximum_validated_fine_gain=max(((r['predicted_goodput_ratios']['oracle']['ratio']-1)*100 for r in validated_fine),default=None)
    control_rows=[]
    for r in control.get('results',[]):
        e=r['errors'];a=r['ACK_error']
        control_rows.append([r['method'],r['QPS'],r['seed'],number(e['TTFT_p95_relative']*100),
            number(e['output_tokens_s_relative']*100),number(e['Ready_D_mean_absolute_ms']),
            number(a['MAE_ms']),number(a['p95_absolute_ms']),r['timing_passed'],r['SLO_passed']])
    control_search_rows=[]
    for r in control.get('aggregate',[]):
        v=r['oracle_over_per_trace_fixed']
        control_search_rows.append([r['QPS'],r['validated_starts'],number(v['ratio']),interval(v['ratio_CI95'])])
    replay_pass=sum(bool(r['qualified_for_counterfactual_interpretation']) for r in model.get('results',[]))
    oracle='# Offline Axis Oracle의 Headroom과 한계\n\n'+provenance+f'''
## 모델 검증을 먼저 검사한다

기존 Discovery 비용으로 별도 과거 Confirmation54개 Start를 예측했다. Timing Gate는48/54개에서 통과했다.
이 결과는 신규 Held-out 결과가 아니다.
신규 비용은 Calibration/Mechanism/Source Window32의30개 Start에서만 추정했다.
Output64/200과 Decode4에는 사전 명시한 외삽이 있다. 해당 테스트 결과로 비용을 다시 맞추지 않았다.
신규 Same-cut Cell 중 Timing과 SLO Gate를 모두 통과한 수는 {replay_pass}/{len(model.get('results',[]))}개다.

Timing Gate는 TTFT p95 상대오차≤10%, Raw Token/s 상대오차≤5%다.
Ready→D 평균 오차는 max(10ms,실측20%) 이하여야 한다.
SLO Gate는 Joint Attainment 절대오차≤0.05와 Qualified Token/s 상대오차≤10%다.
실측 Goodput0이면 예측 Goodput도0이어야 한다.
Gate 실패 Cell의 Headroom은 `DIAGNOSTIC`이다.
단일 P/D Host Step 오차와 누적 Client TTFT 오차도 별도로 기록했다.
MAE, p95/p99 절대오차, Underprediction 비율과 명목값1.05배 Coverage를 포함한다.
이 Coverage는 Calibration된 Upper Bound나 WCET 보장이 아니다.

## Queue State를 갱신하는 Replay

P Queue, FIFO D Ready, Whole-cohort Guard와 Source Outstanding Window를 갱신한다.
KV는 실제 Fresh Native의19113 Free Block을 시작 Budget으로 사용한다.
Prompt+Output 길이에 따른 Uniform Request Reservation Slot을 계산한다.
이는 상세 NIXL Transfer Manager나 Hidden Buffer Simulator가 아니다.
실제 Host State와 Kernel 동시 실행을 완전히 복원했다고 주장하지 않는다.

Same-cut Oracle은18-request Block 안에서 Axis를 선택한다. Decode4에서는36-request Block을 사용한다.
Block마다9-period Token Plan과 Decode Cohort 정렬을 유지한다.
가능한 Sequence는 해당 유한 Grid 안에서 전수 검사한다.
이를 전역 최적 Systems 상한이라고 부르지 않는다.
각 선택 후 후속 Queue와 Guard를 갱신한다. Frozen-state Request별 최소값의 합이 아니다.

## Same-cut Oracle 전체 Cell

표의 Best Fixed는 해당 Trace의 모델상 최선이다. 배포 전에 선정한 설정과 구분한다.

'''+table(['Stage','Cut','Output','QPS','Batch','Seed','Timing+SLO Gate','모델 Best Fixed','Fixed Qualified token/s','Oracle Qualified token/s','모델 Headroom(%)'],oracle_rows)+'''
## Cut까지 포함한 제한적 Oracle

Layer/Token Last2/4/8/12의8개 행동을 비교한다.
Primary Fixed 설정은 새 Discovery에서 선정한 뒤 Held-out 전에 동결했다.
Cut 변경으로 총 이전 Layer Work가 달라질 수 있다. Matched-work Axis 효과와 별도로 해석한다.
아래 CI는 독립 Arrival Start를 재표본한다. 비용 모델 불확실성은 포함하지 않는다.

'''+table(['QPS','Starts','Oracle / Frozen Fixed Goodput','95% CI','Oracle / Trace Best Fixed','95% CI','모든 Fixed 행동 검증'],global_rows)+f'''
{evidence_link('configs/frozen_best_fixed_axes_v1.json')},
{evidence_link('analysis/frozen_model_validation_and_oracle.json')},
{evidence_link('analysis/cross_cut_restricted_oracle.json')}에 전체 Sequence와 검증 오차가 있다.
Native는 외부 실측 기준선이다. Oracle의 제한적 Axis 집합에는 Native를 넣지 않았다.
Oracle이 두 Offload Axis보다 높아도 Native나 Agg보다 높다는 뜻이 아니다.

## 더 세밀한 Same-cohort Search

18/36-request Block 때문에 좋은 선택 시점을 놓칠 수 있다.
별도 Search에서는 실제 Decode Cohort마다 Axis를 선택한다.
한 Cohort 안의 Axis는 같다. P1/D Suffix1과 Decode Batch를 유지하므로 Group Fragmentation을 추가하지 않는다.
16-token Prefix Plan의 Token-layer Row 차이를 DP로 추적해 전체 N개의 작업량을 정확히 맞춘다.
Beam Width8과32를 검사한다. 큰 Search에는 작은 Search의 최선과 두 Fixed 행동을 포함한다.
이것은 유한 Beam Search다. 전역 최적 상한이라고 주장하지 않는다.
Oracle은 미래 Trace를 볼 수 있다. Rule은 이미 도착한 상태와 사전 Request-index Plan만 사용한다.
유한 실험의 Row Debt 정산은 Production Stream 일반화와 다르다.

'''+table(['Cost Model','Stage','Cut','Output','QPS','Batch','Starts','Beam Oracle / Trace Best Fixed','95% CI','Fixed Trace 검증','One Rule Ratio','Two Rule Ratio'],fine_rows)+f'''
{evidence_link('analysis/fine_cohort_axis_oracle.json')}에 모든 Sequence, Row Debt와 실제 Search 범위를 보존한다.
Mean 비용표는 인접16-token 길이의 개별 시간 차이를 복원하지 않는다.
Timing 또는 SLO Gate를 실패하면 이 Search의 이득도 `DIAGNOSTIC`이다.

## 모든 Fixed Gate를 통과한 세 Start Cell만 별도 표시

다음은 두 Axis와 모든 세 Start의 Fixed-trace Timing/SLO Gate가 통과한 부분집합이다.
Headroom은 같은 모델의 Trace Best Fixed 대비 Beam32의 Qualified token/s 개선율이다.
실측 Dynamic 개선이 아니다. 유한 Search이므로 전역 상한도 아니다.

'''+table(['Stage','Cut','QPS','Decode Batch','Starts','모델 Headroom(%)','95% CI(%)'],validated_fine_rows)+'''
검증 실패 Cell을 이 Table에서 성공으로 간주하지 않는다. 전체 Cell은 위 Table과 JSON에 남긴다.
이 부분집합에서 작은 Headroom이 나온 사실만으로 모든 조건의 Dynamic 가능성을 배제하지 않는다.

## Control ACK 경계를 명시한 별도 Replay

고정 Mean P Gap은 Axis가 바뀔 때 D Step과 Reservation Reply의 의존성을 갱신하지 못한다.
따라서 마지막 구조적 대조군에서는 Reservation 요청과 ACK를 별도 Event로 계산한다.
P는 ACK 뒤에만 시작한다. D는 다음 모델 Host Step 경계에서 Reply한다.
Preparation은 요청당 하나의 비선점 Step으로 근사한다. Decode는 출력 간격별 경계를 사용한다.
실제 Async Front-matter와 Step 내부 경계는 계측하지 못했다. 이 근사는 검증 대상이지 확정된 Runtime 복제본이 아니다.

새 Calibration6개 Start만으로 Non-ACK P Gap과 Idle RTT를 고정했다.
규칙과 Simulator 구조는 새 Seed1970/1971/1972의 측정 Arrival 전에 동결했다.
아래 오차는 그 신규 Confirmation의 Fixed-axis 실행에서 계산한다. 검사 실패 후 재튜닝하지 않는다.

'''+table(['Axis','QPS','Seed','TTFT p95 오차(%)','Raw token/s 오차(%)','Ready→D 평균 오차(ms)',
    'ACK MAE(ms)','ACK 절대오차 p95(ms)','Timing Gate','SLO Gate'],control_rows)+'''
두 Fixed Axis의 Timing/SLO Gate가 모두 통과한 Trace에만 Beam Search를 적용한다.
실패한 Trace는 삭제하지 않는다. 해당 Trace의 새 Mixed Search는 `NOT_RUN`이다.
아래 Table은 조건부 통과 집합이다. 실패한 Start를 제외한 무조건적 캠페인 성공으로 해석하지 않는다.

'''+table(['QPS','조건부 검증 Start','Beam / Trace Best Fixed Qualified token/s','95% CI'],control_search_rows)+f'''
{evidence_link('configs/frozen_control_feedback_replay_inputs_v1.json')},
{evidence_link('state/control_feedback_replay_inputs_frozen.json')},
{evidence_link('analysis/control_boundary_replay_validation.json')}에 동결 시각, 전체 오차와 실패를 보존한다.
이 별도 모델도 Mean 비용을 사용한다. ITL p99 보장이나 실제 Dynamic Serving 이득을 뜻하지 않는다.

## 후속 Dynamic 연구 판단

검증된 Headroom5% 이상을 후속 연구의 목표로 삼는다.
작은 차이가 비용 예측 오차보다 작으면 Dynamic 구현을 보류한다.
평균 Decode Interval 모델은 ITL p99 또는 Native-relative5%를 보장하지 않는다.
Mixed Layer/Token State Routing의 실제 Online 정합성은 `NOT_RUN`이다.
이번 연구에서는 Online Dynamic Axis Scheduler를 구현하지 않았다.

## 다음 연구의 최소 검증 순서

먼저 실제 D Host Step의 Control 처리 경계를 계측한다. 계측 전후5% Observer Gate를 다시 검사한다.
그 다음 ACK 대기를 줄이는 Reservation Prefetch 대조군을 두 Axis에 공통 적용할 수 있는지 검토한다.
Credit, ACK와 Lifetime 정합성을 유지해야 한다. 이 대조군은 이번에 구현하거나 측정하지 않았다.
이 변경으로 Axis 우위가 사라지면 기존 이득을 알고리즘의 고유 이득으로 주장하지 않는다.
Whole-cohort Guard를 변경하는 실험도 별도 Runtime 실험으로 분리해야 한다.

새 Queue 모델은 모든 Fixed Axis의 Timing/SLO Gate를 독립 Start에서 먼저 통과해야 한다.
Strong Fixed 기준선에는 각 Cut의 Layer/Token과 Native Fallback을 포함한다.
검증된 Oracle에서5% 이상 여지가 재현될 때만 비용 균형 Scalar에 상태 변수를 추가한다.
실제 Mixed-axis State 정합성과 장기 Queue 안정성도 그 이후 별도 Gate다.
이 순서를 통과하기 전에는 복잡한 Online Controller 구현을 보류한다.
'''+limits

    figure_rows=[]
    for r in manifest.get('figures',[]):
        paths=' '.join(f"[{p['path'].rsplit('/',1)[-1]}]({r['files'][0]['path'].split('poc_flashvep/reports/',1)[-1].rsplit('/',1)[0]+'/'+p['path'].rsplit('/',1)[-1]})" for p in r['files'])
        figure_rows.append([r['stem'],r['evidence'],paths,r['description']])
    figs='# Figure 목록과 비교 계약\n\n'+provenance+'''
## 네 Panel Motivation Figure

SLO Attainment, Mean TTFT, Mean TBT, Generated Output Token/s를 표시한다.
X축은 Offered QPS2–10이다. 신규 측정이 없는 점은 `NOT_RUN`이다.
Output200 Last4/8/12를 각각 표시한다. B512/S16은 표시하지 않는다.
Agg DP2는 기존 N144 Historical 곡선이다. 신규 PD는 별도 Count/Seed의 Batch1 Control이다.
따라서 이를 동일 Arrival Seed의 Paired Four-method 성능 비교라고 주장하지 않는다.
Agg Continuous Batching과 PD의 Fixed Cohort는 실제 Batch 의미도 다르다.
동일 Resource/B/S, PD 내부 Batch Matching, 별도 Batch1 Mechanism Control을 구분한다.

새 PNG/PDF는 Evidence Figure Directory와 `poc_flashvep/` 바로 아래에 저장한다.
기존 Figure는 덮어쓰지 않는다. 빠진 QPS 점을 보간한 측정값으로 표시하지 않는다.

## 전체 Figure

'''+table(['Stem','Evidence','파일','범위'],figure_rows)+limits

    decisions_path=ROOT/'state/research_decisions.jsonl'
    decisions=[json.loads(line) for line in decisions_path.read_text().splitlines()] if decisions_path.exists() else []
    decision_report='# 연구 의사결정 기록\n\n'+provenance+'\nObserve→Hypothesize→Discriminate→Measure→Update 순서로 기록했다.\n'
    for r in decisions:
        decision_report+=f"\n## {r['id']} — {r.get('stage',r.get('phase','NOT_RECORDED'))}\n\n"
        for label,keys in [('가설',('hypothesis',)),('경쟁 가설',('competing_hypothesis','competing')),
                ('최소 실험',('experiment','test')),('사전 예측',('prediction',)),
                ('관측',('observation',)),('다음 선택',('next_action','next')),('Evidence',('evidence',))]:
            value=next((r[k] for k in keys if k in r),'NOT_RECORDED')
            if isinstance(value,list):value=', '.join(value)
            decision_report+=f"{label}: {value}\n\n"
    decision_report+=limits
    failures=[r for r in inventory['runs'] if r['status']=='FAILED']
    failure_text=table(['실행','상태'],[[r['name'],r['status']] for r in failures])
    output_key=[]
    for r in paired:
        if r['stage']!='output_residency_counterfactual' or r['starts']<3:continue
        v=r['metrics']['TTFT_p95_ms'];g=r['common_Balanced_goodput']
        safety={a:next((n['Native_relative_5pct'] for n in native if n['stage']==r['stage'] and
            n['layers']==r['layers'] and n['output']==r['output'] and n['batch']==r['batch'] and
            n['QPS']==r['QPS'] and n['candidate']==a),'NOT_ESTABLISHED') for a in ('Layer','Token')}
        output_key.append([r['output'],r['starts'],number(v['candidate_mean']/1000),number(v['reference_mean']/1000),
            number(v['ratio'],5),interval(v['ratio_CI95'],5),number(g['ratio'],5),interval(g['ratio_CI95'],5),
            safety['Layer']+'/'+safety['Token']])
    resolved=[r for r in classification if r['confidence_label']!='UNRESOLVED']
    correct_cycle=sum(bool(r.get('baseline_correct',{}).get('StaticMetadataCycle')) for r in resolved)
    correct_one=sum(bool(r['correct']['one']) for r in resolved)
    correct_two=sum(bool(r['correct']['two']) for r in resolved)
    cycle_regret=statistics.mean(r['baseline_TTFT_regret']['StaticMetadataCycle'] for r in classification)*100 if classification else None
    two_regret=statistics.mean(r['fixed_axis_TTFT_regret']['two'] for r in classification)*100 if classification else None
    knob_result=f'''검증 Cell은 {len(classification)}개다. TTFT 승자 CI가 확정된 Cell은 {len(resolved)}개다.
Pressure 규칙은 {correct_one}/{len(resolved)}, 두 변수 규칙은 {correct_two}/{len(resolved)}개를 맞췄다.
더 단순한 Static Metadata Cycle도 {correct_cycle}/{len(resolved)}개를 맞췄다.
이 기준선은 `ΔC=max(p_L,u_L+d)−max(p_T,u_T+d)`의 부호를 사용한다.
P/D 비용은 동결된 Lookup이다. d는 Output cap과 실제 Decode Batch에서 추정한 공통 Decode Demand다.
이는 하나의 파생 Scalar다. 센서 하나만 측정하거나 실제 GPU Slack을 직접 측정한다는 뜻은 아니다.
미확정 Cell을 포함한 Fixed-axis TTFT Regret 점 추정치는 Cycle {number(cycle_regret)}%, 두 변수 {number(two_regret)}%다.
이 차이의 통계적 우위나 Online Switching 이득은 입증하지 않았다.
이 Grid에서는 비용 균형 Scalar에 제한적인 Held-out 지지가 있다. 두 번째 상태 변수의 필수성은 입증되지 않았다.
새 Batch4의 승자는 미확정이다. Timing Model 검증 실패도 남아 있다.
따라서 이를 범용 Deadline-safe Admission Rule로 사용하지 않는다.

모든 세 Start의 Fixed Timing/SLO Gate를 통과한 Fine Search Cell은 {len(validated_fine)}개다.
그 부분집합의 모델상 Qualified token/s Headroom 점 추정치는 최대 {number(maximum_validated_fine_gain,5)}%다.
실제 Online 개선이나 전역 Oracle 상한이 아니다. 나머지 Cell의 모델 검증 실패는 별도로 남겼다.
요구한 검증된5% Dynamic Headroom은 확보하지 못했다. 따라서 현재는 Online Axis Scheduler 개발을 보류한다.

'''
    summary='# Dense AR Layer-vs-Token Sweet Spot 및 Minimal Knob 최종 요약\n\n'+provenance+f'''
## 핵심 판정

Stage Service Balance와 Whole-cohort Guard는 기존 Sweet Spot을 설명하는 유력한 기전이다.
작은 Token Shape의 실제 Serving GEMM 비효율도 존재한다.
그러나 Forward 이득과 TTFT 이득은 다르다. Queue와 Admission이 전체 결과를 바꿀 수 있다.
물리적 GPU 성분을 완전히 분리하지 못했으므로 Root Cause 판정은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`이다.
1~2 Knob과 Offline Headroom의 검증 범위는 아래 결과 Table과 개별 보고서로 제한한다.
전체 Workload에 적용되는 Online Dynamic Axis 우위는 `NOT_ESTABLISHED`다.

## 새 Output 통제 실험에서 확인한 역전

Last4/Token≈889, Input8000, QPS6, N72, 실제 P1/D Suffix1/Decode2다.
Partition, 전체 GEMM Rows, Source Window와 Arrival Seed를 일치시켰다.
Output cap만16→64로 바꿨다. Runtime Context 상한도8016→8064로 파생된다.
따라서 이는 Output cap의 전체 효과다. 순수 GPU Decode 시간만의 효과라고 주장하지 않는다.

'''+table(['Output','Paired Starts','Layer TTFT p95(s)','Token TTFT p95(s)',
    'Layer/Token TTFT','95% CI','Layer/Token Qualified token/s','95% CI','Native5% Layer/Token'],output_key)+'''
작은 Cut이라는 사실만으로 Layer의 TTFT 우위를 예측할 수 없다.
짧은 Output에서는 P 앞 대기가 D 준비 비용 절감을 상쇄할 수 있다.
긴 Output에서는 D Cohort Residency와 D 앞 대기가 커진다.
이때 같은 D 준비 비용 차이가 후속 요청의 대기에 누적될 수 있다.
정확한 Mean TTFT 구간 분해와 Host 비용은 원인 분석 보고서에 있다.
이 결과는 Finite-cohort 역전이다. Joint SLO95%의 장기 안정 Capacity나 Online Dynamic 이득을 뜻하지 않는다.

## 최소 변수와 Dynamic 필요성

'''+knob_result+'''
동결한 Pressure Scalar, 두 변수 규칙과 Static Metadata Cycle을 같은 새 Cell에서 검사했다.
Static Metadata Cycle도 비용 균형을 하나의 파생 Scalar로 표현할 수 있다.
하지만 Metadata 기준선만큼의 예측 성공은 State-aware Dynamic의 독립 기여가 아니다.
미확정 승자, Calibration 순위 불확실성과 새로운 Batch의 예측 실패를 함께 보존한다.
검증된 Fixed Trace에서5% 이상 Headroom이 확인되지 않으면 복잡한 Axis Scheduler를 보류한다.
상세 Accuracy, Regret와 조건부 Oracle 수치는 각 보고서에 있다.

## 신규 실험 범위

'''+count_text+state_text+'''
## 독립 Start3회 이상의 Layer / Token 결과

'''+table(['Stage','Cut','Output','QPS','Decode Batch','WIP','Starts','TTFT Ratio','95% CI','Goodput Ratio','95% CI'],[
        row for row,r in zip(comparison_rows,paired) if r['starts']>=3])+'''
## 15개 연구 질문에 대한 답

1. Last4의 과거 Token 우위는 해당 Decode2 실행의 P 단계 부담과 관련됐다. 신규 Batch1 결과에 그대로 일반화하지 않는다.
2. Last8 Token의 긴 Ready→D 대기는 대부분 이전 Cohort Guard와 겹쳤다. Guard에는 Suffix와 Decode/Delivery가 포함된다.
3. Last12에서는 두 방식의 D Guard/request가 P Source Service보다 커졌다. 이 Pipeline 불균형과 반복 대기가 주요 설명 후보다.
4. Batch를 맞춰도 Query 길이, Layer 수, Attention 분배와 Hidden Payload가 다르다. 동일 GEMM Rows가 동일 전체 FLOPs를 뜻하지 않는다.
5. 과거 Last8의 낮은 TTFT는 해당 Decode2의 P/D 균형과 일치했다. 신규 Batch1에서는 더 작은 Cut이나 Native가 강할 수 있다.
6. QPS0.5의 O16 Control에서 Layer의 짧은 First Path가 관측됐다. 각 행동1회이므로 확정적인 범용 우위는 아니다.
7. QPS 변화의 승자와 CI는 Online Table에 있다. 점 추정치 부호가 바뀐 사실과 확정 Crossover를 구분한다.
8. Output만 바꾼 인과 비교는 Decode2/N72에서 검사했다. Batch1의 N72/O16과 N36/Long Output은 Count도 달라 단독 인과 근거가 아니다.
9. 단순 Cut 크기보다 P/D Effective Service Demand와 Decode Cohort Residency가 중요한 후보였다. 실제 GPU Idle과 동일한 변수는 아니다.
10. 단일 P/D Pressure Rule은 사전 동결한 κ4를 사용했다. 더 단순한 비용 균형 Scalar의 검증 결과도 위에 제시했다.
11. 두 변수 Rule은 Decode Demand와 이미 도착한 Total Backlog를 사용했다. Static Metadata 기준선보다 높은 확정 Cell Accuracy는 얻지 못했다.
12. 새 Seed/QPS/Output/Batch를 검사했다. 새 Prompt Distribution과 Input Length 일반화는 실행하지 않았다.
13. 사용하는 Counter는 P 제출 전에 관측 가능하다. 비용 Lookup은 Calibration이다. 미래 Arrival/Completion을 읽지 않는다.
14. Offline Headroom은 상태를 갱신하는 제한적 Block Oracle로 계산했다. 검증 Gate 실패 Cell은 Diagnostic이며 Online Speedup이 아니다.
15. 현재는 비용·Queue 모델을 더 검증할 근거가 있다. 강한 Fixed Axis 대비 검증된5% Headroom과 Decode 보호가 없으면 Dynamic 구현을 진행하지 않는다.

## 실패와 미실행

'''+failure_text+'''
초기 Native v1은 Extension 생성자의 Cut 계약에서 측정 전에 실패했다.
유효한 Placeholder Cut을 유지하는 새 v2로 복구했다. 실제 Native는 Full36을 수행한다.
실패한 Config와 Log는 삭제하지 않았다.
온라인 Observer4개 Cell과 Serial Token Cell은5% Observer Gate를 실패했다.
이를 Clean Kernel Time으로 사용하지 않았다. GPU SM/HBM Counter는 `NOT_MEASURED`다.
Main QPS2–10 중 신규로 실행하지 않은 점은 `NOT_RUN`이다.
Inventory의 NOT_RUN14개는 초기 v1에서 v2로 교체한 미실행 Config다. 삭제하거나 실제 실패로 세지 않았다.
장기 Sustainable Throughput과 Online Dynamic Axis Scheduler는 이번 캠페인에서 검증하지 않았다.

## 보고서와 재현 자료

'''+''.join(f'- [{name}]({name})\n' for name in DOCUMENTS[1:])+f'''

{evidence_link('analysis/experiment_inventory.json','전체 Matrix')},
{evidence_link('analysis/per_start_client_metrics.csv','실측 CSV')},
{evidence_link('state/resume_progress_manifest.json','Resume Manifest')},
{evidence_link('references/verified_primary_sources.md','확인한 원문과 공식 코드')}를 보존한다.
최종 Commit과 Remote SHA는 별도 Publication Receipt에서 검증한다.
Burner 복원은 Resource Closeout Receipt에서 확인한다. 검증 전에는 완료했다고 표현하지 않는다.
'''+limits
    outputs={DOCUMENTS[0]:summary,DOCUMENTS[2]:online,DOCUMENTS[3]:causal_report,
        DOCUMENTS[4]:knob,DOCUMENTS[5]:oracle,DOCUMENTS[6]:figs,DOCUMENTS[7]:decision_report}
    # The independently written Phase A audit stays intact.
    for filename,text in outputs.items():
        (root/filename).write_text(text)
    write(ROOT/'analysis/report_generation_receipt.json',dict(time=time.time(),status=report_status,
        reports=list(outputs),clean_starts=complete,confirmed_axis_cells=len(confirmed),
        source_metrics='analysis/independent_start_statistics.json',
        scope='Generated new prose only. Historical reports and the independently written new Phase A audit are unchanged. Scientific interpretation requires review.'))
    preserve();print('SWEETSPOT_REPORTS_WRITTEN',len(outputs),report_status,flush=True)


if __name__=='__main__':
    main()
