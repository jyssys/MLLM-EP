"""Report the revised batch controls separately from the deferred grid."""
import json
import math
import statistics
import time
from .common import BASE, BRANCH, MODEL_REVISION, REPO, ROOT, digest, preserve, write
from .report_drafts import interval, number, table


def load(path):
    return json.loads(path.read_text())


def key(row):
    return tuple(row[k] for k in ('stage', 'reference', 'candidate', 'layers', 'cap', 'QPS', 'count', 'admission'))


def main():
    preserve()
    assert load(ROOT/'analysis/final_integrity_validation.json')['passed']
    assert load(ROOT/'state/final_GPU_resource_restoration.json')['passed']
    data_path = ROOT/'analysis/independent_start_statistics.json'
    data = load(data_path)
    sensitivity = {key(r):r for r in load(ROOT/'analysis/CI_quantile_sensitivity.json')['comparisons']}
    root = load(ROOT/'analysis/root_cause_components.json')
    aggregate = [r for r in data['aggregate'] if r['stage'] in ('batch1_confirmation', 'cohort2_confirmation')]
    comparisons = [r for r in data['comparisons'] if r['stage'] in ('batch1_confirmation', 'cohort2_confirmation')]
    headers = ['Stage', 'Output', 'Method', 'Last', 'Starts', 'TTFT p95 ms', 'ITL p99 ms',
               'TPOT p95 ms', 'Raw output tokens/s', 'Joint SLO %', 'SLO output tokens/s']
    metrics_table = table(headers, [[r['stage'], r['cap'], r['method'], r['layers'], r['starts'],
        *[number(r['metrics'][k]['mean'], 4) for k in ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'output_tokens_s')],
        number(100*r['regimes']['Common_Balanced_OutputSweep']['Joint_attainment']['mean'], 4),
        number(r['regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput']['mean'], 4)] for r in aggregate])
    ratio_rows = []
    for r in comparisons:
        if r['reference'] != 'Token':
            continue
        cdf = sensitivity[key(r)]['metrics']['TTFT_p95_ms']['empirical_CDF_CI95']
        ratio_rows.append([r['stage'], r['cap'], r['layers'], r['starts'],
            interval(r['metrics']['TTFT_p95_ms']), str(cdf),
            interval(r['metrics']['output_tokens_s']),
            interval(r['regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput'])])
    ratios = table(['Stage', 'Output', 'Last', 'Starts', 'Layer/Token TTFT p95 [Type7 95% CI]',
                    'TTFT 역CDF 95% CI', 'Layer/Token raw token/s [95% CI]',
                    'Layer/Token common-SLO token/s [95% CI]'], ratio_rows)
    safety = table(['Stage', 'Output', 'Candidate', 'Last', 'ITL p99 / Native [95% CI]',
                    'TPOT p95 / Native [95% CI]', 'Type7 Gate', '역CDF Gate'], [
        [r['stage'], r['cap'], r['candidate'], r['layers'], interval(r['metrics']['ITL_p99_ms']),
         interval(r['metrics']['TPOT_p95_ms']), r['Native_relative_5pct'],
         sensitivity[key(r)]['Native_relative_empirical_CDF_5pct']]
        for r in comparisons if r['reference'] == 'Native'])
    historical = [r for r in data['starts'] if r['stage'] in ('discovery', 'motivation_odd', 'admission_control')]
    history = table(['Stage', 'Output', 'QPS', 'N', 'Arm count'], [
        [stage, cap, q, n, sum(r['stage']==stage and r['cap']==cap and r['QPS']==q and r['count']==n for r in historical)]
        for stage, cap, q, n in sorted({(r['stage'], r['cap'], r['QPS'], r['count']) for r in historical})])
    scope = load(ROOT/'analysis/final_integrity_validation.json')
    partition = table(['Pair', 'Layer P/D layers', 'Token Suffix histogram / N144',
                       'P GEMM rows/request', 'D GEMM rows/request'], [
        [r['layers'], f"{36-r['layers']}/{r['layers']}",
         json.dumps(r['token_suffix_histogram'],sort_keys=True),
         r['totals']['layer_P_GEMM_rows']//144, r['totals']['layer_D_GEMM_rows']//144]
        for r in load(ROOT/'analysis/matched_axis_structural_work.json')['rows']])
    common = f'''## 환경과 증거 계약

Model은 Qwen/Qwen3-8B이다. Revision은 `{MODEL_REVISION}`이다.
Original BF16, Exact Causal Attention, Greedy, Thinking Disabled와 Full36 Decode를 유지했다.
Input은 8000 Tokens이다. `ignore_eos=True`를 사용했다.
GPU0/1은 H100 SXM 80GB이다. GPU당 KV Budget은 42GiB이다.
GPU UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`와 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`이다.
vLLM은 0.26.0+cu129이다. PyTorch는 2.11.0+cu129이다. NIXL은 1.4.1이다.
Prefix Cache는 비활성화했다. 모든 Worker는 실제 B8192/S1024 Gate를 통과했다.

Publication Parent는 `{BASE}`이다. Branch는 `{BRANCH}`이다.
Parent SHA를 모든 실행의 실제 Source SHA라고 표시하지 않는다.
각 `raw/<name>/command.json`에 실행한 Source ZIP의 SHA256을 저장했다.
Model Revision, Config Hash, GPU Mapping, Dataset Hash와 SLO Hash도 실행별로 저장했다.
기존 Inference Adapter 8개와 Predecessor Source 523개는 변경하지 않았다.
사용자 Checkout, Staged Index와 설치 Runtime도 보존했다.

## 사용자 요청에 따른 범위 변경

초기 공통 Admission8 Discovery는 198/232회 완료했다.
초기 Admission 대조는 52/52회 완료했다.
실제 배치 일치 요청에 따라 나머지 Discovery 34회와 기존 Confirmation 120회는 보류했다.
이 조건은 `NOT_RUN`이다. 완료로 표시하지 않았다.
기존 Native 설정 Bridge 6회도 `NOT_RUN`이다.
따라서 기존 Native 32000/4·128000/16 대비 B/S만의 인과 효과는 `NOT_MEASURED`이다.
Last2 전체 Sweep은 취소 상태를 유지했다.
이전 소규모 Last2 Forward 진단은 별도 증거로 보존했다.

새 Batch1 성능은 98/98회 완료했다.
새 준비1/Decode2 성능은 42/42회 완료했다.
두 Protocol 모두 Native와 Last4/8/12의 Layer/Token, 총 7개 Arm을 사용했다.
P 준비 배치와 D Suffix 준비 배치는 항상 실제 요청 1개였다.
Decode 배치만 Protocol에 따라 실제 요청 1개 또는 2개였다.
두 축의 실제 Query 수, Padding, Source/Destination Group과 작업량을 검사했다.
B8192/S1024만 같다는 이유로 실제 배치도 같다고 판단하지 않았다.

## 정확한 Partition과 작업량 — STRUCTURAL_ACCOUNTING

{partition}

표의 Token Suffix는 16-Token 경계에 맞춘 인접 길이 조합이다.
N72는 같은 분포를 절반의 요청 수로 사용한다.
평균 Suffix는 각각 888.888889/1777.777778/2666.666667 Tokens이다.
실행별 Query Coverage로 전체 P/D Rows 일치를 다시 검사했다.
같은 GEMM Rows는 같은 Attention FLOPs나 GPU 시간을 뜻하지 않는다.

## 표본, 정합성과 SLO

Discovery는 N72이며 조건별 독립 Start 1회이다.
Batch1은 Output16의 QPS2/4/6/8/10과 Output64의 QPS2/6/10을 사용했다.
Decode2는 Output16의 QPS2/6/10을 사용했다.
Discovery Arrival Seed는 2026111810이다.
Confirmation은 QPS6, N144이며 조건별 독립 Start 3회이다.
Confirmation Seed는 2026111820/2026111821/2026111822이다.
Discovery와 Confirmation을 CI 계산에서 합치지 않았다.
기존 실제 문서 Pool인 `matched144.json`을 사용했다.
새 Arrival Seed 검증이며, 미관측 Prompt Pool 검증은 아니다.

각 새 Protocol은 N18 Full-state Gate를 7개 Arm에서 통과했다.
Full Prompt KV, Hidden, Greedy Output, Layer 방문과 Lifetime을 검사했다.
성능 N144에서는 Output, Query/Layer Coverage, 작업량과 Retirement를 검사했다.
모든 N144 Timing 요청의 Full KV를 CPU로 복사하여 Hash한 것은 아니다.
Decode2의 초기 Version1/2 실패는 삭제하지 않았다.
Version3은 모든 State/Geometry Gate 뒤에만 성능을 실행했다.
첫 Token Cohort Barrier 대기는 Client TTFT에 포함했다.
원래 Generation Timestamp와 Token Count는 변경하지 않았다.

기존 Frozen SLO를 변경하지 않았다.
표의 공통 Balanced SLO는 TTFT≤5000ms, 모든 ITL≤34ms, TPOT≤33ms이다.
Output별 기존 Frozen SLO도 별도 JSON에 보존했다.
Goodput의 분자는 세 조건을 모두 통과한 요청의 실제 Output Tokens이다.
ITL은 첫 Interval을 포함하는 Client-visible 연속 Token 간격이다.
Mean TBT는 모든 해당 Client-visible Interval의 산술 평균이다.
Request TPOT는 요청의 첫 Token부터 마지막 Token까지의 시간을 Output Tokens−1로 나눈 값이다.
Raw token/s와 SLO token/s의 분모는 유한 Cohort의 Arrival-to-drain 시간이다.
이를 Sustainable Capacity라고 주장하지 않는다.

CI의 재표본 단위는 독립 Server Start이다.
요청이나 Token을 독립 Start로 취급하지 않았다.
표의 p95/p99는 Start 안의 분위수를 Start 사이에서 평균한 값이다.
Type7 Bootstrap CI와 역CDF 민감도 CI를 함께 제시한다.
3개 Start의 CI는 모집단 Coverage나 WCET 보장을 뜻하지 않는다.
CI는 Pointwise이다. 다중 비교 보정은 하지 않았다.
Native-relative 보호는 ITL p99와 TPOT p95의 두 CI 상한이 모두 1.05 이하여야 한다.
Absolute SLO 통과와 Native-relative 보호는 다른 Gate이다.

원시 결과는 `dense_ar_matched_axis_crossover_20261117/`에 보존했다.
`analysis/full_execution_matrix.csv`는 실행 상태 전체를 포함한다.
`analysis/per_request_metrics_and_work.csv`는 390개 성능 Start의 45,000개 요청을 포함한다.
`analysis/independent_start_statistics.json`에 반올림하지 않은 값과 CI를 저장했다.
검증은 `analysis/final_integrity_validation.json`에서 확인한다.
'''
    outcome = '''## 핵심 판정

원했던 작은 Cut Layer 우위·큰 Cut Token 우위는 이번 Online 대조에서 입증하지 못했다.
실제 배치를 맞추면 이전 자유 배치 결과와 다른 순위가 나타났다.
따라서 배치 Geometry는 중요한 혼동 변수였다.
그러나 이것만으로 모든 차이의 유일한 원인을 확정하지 않는다.

Batch1의 Output16/QPS6에서 Last4 차이는 불확실했다.
Last8과 Last12에서는 Layer의 TTFT p95와 SLO-goodput이 우세했다.
Output64에서는 차이가 대체로 작아졌다.
Last4 TTFT의 경계 판정은 CI 분위수 정의에 민감했다.
Last12의 TTFT 우위는 존재했지만 Native-relative 보호가 모든 Gate를 통과하지는 못했다.

준비1/Decode2의 Output16/QPS6에서는 반대 방향의 Partition-dependent 역전이 나타났다.
Last4는 Token의 TTFT p95가 낮았다.
Last8과 Last12는 Layer의 TTFT p95가 낮았다.
이 방향은 사용자가 처음 제시한 작은 Layer·큰 Token 가설과 반대이다.
세 Pair의 Native-relative Decode 보호는 Type7과 역CDF Gate를 모두 통과했다.
하지만 Joint SLO Attainment는 모두 95% 미만이었다.
따라서 성공한 지속 가능 SLO Serving으로 표현하지 않는다.

판정은 `MEASURED_CONDITIONAL_AXIS_BENEFIT`과 `DESIRED_CROSSOVER_NOT_ESTABLISHED`이다.
동일 배치에서 Layer의 조건부 Online 이득은 존재한다.
그 이득을 순수 GEMM 효율이나 Loop-specific 효과로 환원할 근거는 부족하다.
Strong Colocated 대비 우위나 Dynamic 정책의 독립 기여는 이번 대조에서 검증하지 않았다.
'''
    destination = REPO/'poc_flashvep/reports'
    documents = []
    previous_manifest = ROOT/'analysis/scoped_final_report_manifest.json'
    previous_documents = {r['path']:r['SHA256'] for r in load(previous_manifest)['documents']} if previous_manifest.exists() else {}
    def report(name, title, body):
        path = destination/name
        payload = '# '+title+'\n\n'+body.strip()+'\n'
        if path.exists():
            assert path.read_text() == payload or previous_documents.get(str(path.relative_to(REPO))) == digest(path), 'Preserve an unrelated report: '+name
        if not path.exists() or path.read_text() != payload:
            path.write_text(payload)
        documents.append(dict(path=str(path.relative_to(REPO)), SHA256=digest(path)))
    results = common+'\n## 독립 Confirmation 결과 — MEASURED\n\n'+metrics_table+'\n\n## 동일 작업량의 Layer/Token 비율\n\n'+ratios+'\n\n## Native-relative Decode Gate\n\n'+safety
    report('dense_ar_actual_batch_matched_online_results.md', '실제 Batch 일치 Layer/Token Online 결과', outcome+'\n'+results)
    report('dense_ar_matched_work_output_sweep_results.md', 'Matched-work Output Sweep과 변경된 실험 범위', results+'\n\n## 초기 공통 Admission8 실행 Matrix\n\n'+history+'''

초기 Matrix는 Output16/32/64/128/200을 포함한다.
초기 Matrix의 실제 Decode Geometry는 Layer와 Token에서 달랐다.
이 결과는 새 실제 Batch 일치 결과와 합치지 않는다.
새 Batch 일치에서는 Output16/64만 실행했다.
Output32/128/200의 실제 Batch 일치 성능은 `NOT_RUN`이다.
따라서 전체 Output16–200의 배치 일치 Crossover를 확인했다고 주장하지 않는다.
''')
    host_rows = []
    for stage, cap in (('batch1_confirmation', 16), ('batch1_confirmation', 64), ('cohort2_confirmation', 16)):
        for a, m in [('Native', 0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]:
            rows = [r for r in root['host_starts'] if r['stage']==stage and r['cap']==cap and r['method']==a and r['layers']==m]
            assert len(rows)==3
            avg = lambda k:statistics.mean(r[k] for r in rows)
            spans = {k:statistics.mean(r['host_path_mean_ms'][k] for r in rows) for k in rows[0]['host_path_mean_ms']}
            host_rows.append([stage,cap,a,m,*[number(spans[k],3) for k in spans],
                number(avg('summed_D_preparation_per_request_ms'),3),number(avg('summed_decode_residency_per_request_ms'),3)])
    host_table = table(['Stage','Output','Method','Last','Arrival→P ms','P→Ready ms','Ready→D ms',
                        'D→First worker ms','Worker→Client ms','D prep/cohort request ms','Worker residency/cohort request ms'], host_rows)
    diagnosis = '''## 계산 이득과 TTFT의 차이

작은 Cut의 실제 D Forward 이득은 이전 직렬 진단에서 확인했다.
Last4는 Layer30.3130ms, Token38.3212ms였다.
그러나 같은 진단의 Client TTFT p95는 318.0027ms와329.9976ms였다.
각 방식은 독립 Start 1회이다. 95% CI는 없다.
Forward의 비율을 그대로 전체 TTFT 비율로 바꾸지 않는다.

현재 TTFT에는 P 제출 대기와 D Admission 대기가 포함된다.
이전 Cohort의 Decode Residency는 다음 요청의 D 대기를 늘릴 수 있다.
준비1/Decode2에서는 준비와 Decode Group을 분리했다.
그 변경은 Batch1과 다른 실행 구조이다.
실제 같은 배치를 사용해도 Suffix Shape와 P/D Pipeline Balance는 다르다.

아래 표는 Host Clock의 평균 구간 회계이다.
처음 다섯 구간의 합은 Mean TTFT와 일치한다.
마지막 두 Cohort 지표는 그 합에 추가하는 항목이 아니다.
특히 Decode2 Worker Residency에는 남은 준비와 Barrier 전 시간이 섞일 수 있다.
Exclusive Decode Kernel Time으로 해석하지 않는다.
구간별 p95를 더하거나 빼서 TTFT p95를 분해하지 않는다.
'''
    mechanism = '''## GEMM, Attention과 전송

기존 고립 GEMM은 F.linear를 사용했다.
현재 Serving은 Batch-invariant Linear를 직접 호출한다.
H100에서 두 호출 경로는 같지 않았다.
새 고립 대조는 실제 Serving API를 사용했다.
이 과정에서 Model Weight, Precision과 Serving 수학을 바꾸지 않았다.

Layer0 Projection 중앙값을 확장한 GPU1 비용 모델에서 Token/Layer 비율은 다음과 같다.
Last4는 1.1044이다. Last8은 1.0749이다. Last12는 1.0486이다.
이는 `STRUCTURAL_ACCOUNTING`이며 Online Exclusive GEMM 실측이 아니다.
이 모델은 이득 감소를 보이지만 큰 Cut의 Token 역전을 예측하지 않는다.
이전 Loop의 Token64 결과를 이번 Dense Token889에 그대로 적용할 수 없다.

동일 GEMM Rows에서도 D의 Attention 분배는 다르다.
Token/Layer의 D Causal Pair 비율은 Last4 약1.889, Last8 약1.778, Last12 약1.667이다.
전체 시스템의 Attention Pair 합은 같다.
Pair 비율을 GPU 시간 비율로 표시하지 않는다.

같은 Pair의 유효 KV Payload 합은 Layer와 Token에서 같다.
Layer는 Request당 Hidden/Residual131072000Bytes를 추가 전송한다.
전송 API 경과시간을 순수 DMA 시간으로 표시하지 않는다.
KV Payload 합이 같아도 Credit 점유 기간과 Admission 대기는 다를 수 있다.

Clean Online Profiler의 전체 5% 간섭 Gate는 통과하지 못했다.
따라서 Online Exclusive GEMM, Attention, SM/HBM Counter는 `NOT_MEASURED`이다.
기존 Kernel Timeline은 `DIAGNOSTIC`으로 보존한다.
새 고립 API 대조의 16개 Profiler Gate 통과는 Online Gate 통과가 아니다.

현재 원인 판정은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`이다.
실제 배치와 Cohort 구조의 영향 및 긴 Host 대기는 확인했다.
전체 Online 차이를 GEMM·Attention·Transfer의 독점적 GPU 시간으로 분해하지는 못했다.
원하는 방향의 Crossover를 GEMM 효율만으로 설명할 수 있다는 주장은 미입증이다.
'''
    report('dense_ar_layer_token_crossover_root_cause.md', '배치 일치 Layer/Token 순위와 원인 분석', outcome+'\n'+common+'\n'+diagnosis+'\n'+host_table+'\n\n'+mechanism)
    report('dense_ar_matched_axis_crossover_final_summary.md', 'Dense AR Matched-axis Crossover 최종 요약', outcome+'\n'+results+'''

## 연구 질문의 직접 답변

네 방식의 초기 비교에는 실제 B8192/S1024를 적용했다.
초기 PD Admission 상한은 8로 같았다. 실제 배치는 같지 않았다.
사용자 요청 뒤 실제 준비1/Decode1과 준비1/Decode2를 별도로 통제했다.
Last4/8/12의 Token Suffix는 인접 16-Token 경계 조합이다.
N72/N144 전체 P/D GEMM Token-layer Rows를 정확히 맞췄다.
모든 지정 요청을 실제 이전했다. 성능 비교에 숨긴 Native Fallback은 없다.

동일 배치에서도 Layer의 Online TTFT와 Goodput 우위가 일부 확인됐다.
하지만 작은 Cut일수록 Serving 이득이 커진다는 단순 관계는 확인되지 않았다.
준비1/Decode2에서는 Last4의 Token 우위와 Last8/12의 Layer 우위가 관측됐다.
Output Length 자체의 배치 일치 Crossover는 Output16/64 범위에서 확정하지 못했다.
Output32/128/200의 새 배치 일치 검증은 하지 않았다.

초기 Output200 Motivation은 QPS2/4/6/8/10의 실측으로 다시 제시했다.
QPS3/5/7/9는 보류 상태이므로 점을 만들지 않았다.
B512/S16은 새 Figure에 포함하지 않았다.
초기 실제 Geometry가 다른 결과를 새 배치 일치 비교로 표시하지 않았다.
기존 Native B/S만의 효과는 Bridge 미실행으로 `NOT_MEASURED`이다.

향후 Axis Selection은 Cut 크기 하나보다 실제 준비/Decode Geometry를 함께 봐야 한다.
P 준비 비용, D 준비 비용, Pending Queue와 Decode Residency도 후보 관측값이다.
이 관측값으로 동작하는 Dynamic Policy는 이번 캠페인에서 구현하거나 검증하지 않았다.

## 최종 자원과 보존

GPU 실험은 종료했다. 실제 배치 성능 140회를 완료했다.
390개 적격 성능 Start의 45,000개 요청을 검증했다.
17개 실패 실행과 초기 Prototype 실패를 삭제하지 않았다.
일부 실패는 선택적 Profiler와 종료 전에 차단된 정합성 Prototype이다.
이를 Clean 성능에 포함하지 않았다.

완료 모니터의 Burner 복원 호출이 누락돼 GPU가 빈 상태로 남았다.
2026-11-18 23:05 KST 점검에서 이를 확인했다.
승인된 Facade로 GPU0–7 Burner를 복원했다.
소유 세션과 실제 양의 사용률을 3회 연속 검사했다.
완료 경로의 회귀 검사도 실패 상태에서 확인한 뒤 수정했다.
이 수정은 실험 결과나 Model Forward를 변경하지 않았다.
최종 Burner는 계속 실행한다.

실행별 Source ZIP과 원시 결과는 검증된 무손실 Archive로 보존한다.
사용자 Index를 사용하지 않는 별도 Index로 Commit/Push한다.
Remote SHA 확인은 Commit 이후의 로컬 Publication Receipt에 저장한다.
이 보고서 작성 시점의 Parent SHA를 Push 완료 SHA로 대신 표시하지 않는다.
''')
    report('dense_ar_matched_axis_small_start_ci_sensitivity.md', '독립 Start CI와 분위수 민감도', common+'\n'+ratios+'\n\n'+safety+'''

## 경계 판정

Batch1 Output64 Last4의 TTFT Type7 CI 상한은 1 미만이다.
하지만 역CDF CI 상한은 1.000251040918798이다.
따라서 이 조건의 TTFT 우위를 확정적으로 표현하지 않는다.
Type7과 역CDF의 원래 값을 모두 유지했다.
원하는 결과를 얻기 위해 CI 정의를 교체하지 않았다.
3개 Start에서 가능한 Bootstrap Draw는 27개이다.
이 분포의 모든 Draw를 열거해도 모집단 신뢰도를 보장하지 않는다.
''')
    report('dense_ar_unified_budget_duetserve_motivation_results.md', '통일 Budget Motivation Figure의 범위', common+'''

## Output200 Figure

초기 공통 Admission8의 Output200 실측을 사용했다.
Agg DP2, Native PD, Static Layer, Static Token을 비교했다.
네 방식 모두 Worker B8192/S1024이다.
동일 Prompt, Arrival Seed2026111780과 N144를 사용했다.
Agg는 이 조건에서 새로 실행한 적격 측정이다.
이전 다른 Seed의 주황 곡선을 Paired 결과로 재사용하지 않았다.

실측 QPS는 2/4/6/8/10이다.
QPS3/5/7/9는 사용자 범위 변경으로 `NOT_RUN`이다.
Figure의 X축은 QPS2–10이다. 미측정 점은 비워 둔다.
Last4/Token≈889, Last8/Token≈1778, Last12/Token≈2667을 각각 같은 수준으로 제시한다.
Main 예시는 사전 요청에 따라 Last12이다. 결과를 보고 유리한 Pair만 고르지 않았다.
B512/S16은 제거했다.

이 Figure는 실제 배치 일치 Figure가 아니다.
초기 Layer의 실제 준비/Decode Group은 1이었다.
Token은 Budget과 Ready 상태에 따라 여러 요청을 묶을 수 있었다.
공통 Admission 상한과 실제 Group 크기를 구분한다.
이 차이를 숨기고 Layer-axis 자체의 성능 원인이라고 주장하지 않는다.

## 새 실제 Batch 일치 Figure

별도의 Output16/64 Batch1 Figure와 Output16 Decode2 Figure를 생성했다.
이 Figure에는 적격한 Native, Layer4/8/12와 Token4/8/12만 표시했다.
Agg는 이 실제 배치 통제 Protocol로 실행하지 않았으므로 곡선을 만들지 않았다.
Discovery N72와 Confirmation N144는 같은 Curve에 합치지 않았다.
확인된 QPS6의 독립 Start 3회 결과는 별도 Figure에 CI를 표시했다.
Output200 새 실제 Batch 일치 Motivation은 `NOT_RUN`이다.
''')
    figures = draw(data, sensitivity, root)
    figure_lines = '\n'.join('- ['+p['path'].split('/')[-1]+']('+p['path'].removeprefix('poc_flashvep/reports/')+') — '+p['evidence'] for p in figures if p['path'].endswith('.png') and '/reports/' in p['path'])
    report('dense_ar_matched_axis_output_sweep_figures.md', 'Matched-axis Figure와 재현 위치', common+'\n## Figure 목록\n\n'+figure_lines+'''

PNG와 PDF를 함께 저장했다.
주요 Figure는 사용자 요청 경로인 `poc_flashvep/`에도 저장했다.
실제 측정 점만 Marker로 표시했다.
단일 Start의 점에는 독립 Start CI가 없다.
Ratio Figure의 넓은 회색 범위는 역CDF 민감도 CI이다.
색 범위는 원래 Type7 CI이다.
Host 평균 구간의 누적 그림은 GPU Kernel Timeline이 아니다.
고립 GEMM 확장 모델은 Online Serving 실측과 분리했다.
기존 계측 Gate 실패 Timeline을 성공한 Physical Idle 그림으로 바꾸지 않았다.
''')
    write(ROOT/'analysis/scoped_final_report_manifest.json', dict(epoch=time.time(), documents=documents,
        statistics_SHA256=digest(data_path), final_integrity_SHA256=digest(ROOT/'analysis/final_integrity_validation.json'),
        scope='Final reports for the user-revised actual-batch protocols. Preserve deferred original matrices and existing frozen reports.'))
    preserve()
    print('SCOPED_FINAL_REPORTS', len(documents), 'FIGURE_FILES', len(figures), flush=True)


def draw(data, sensitivity, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figures = []
    colors = {0:'black',4:'#0072B2',8:'#D55E00',12:'#009E73'}
    arms = [('Native',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]
    def save(fig, stem, evidence, source_names):
        fig.tight_layout(rect=(0,.055,1,.91))
        for ext in ('png','pdf'):
            for parent in (ROOT/'figures',REPO/'poc_flashvep'):
                path = parent/(stem+'.'+ext)
                fig.savefig(path,dpi=170)
                figures.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path),
                    evidence=evidence,source_starts=source_names))
        plt.close(fig)
    panels = [('Joint SLO attainment (%)',lambda r:100*r['regimes']['Common_Balanced_OutputSweep']['Joint_attainment']),
              ('Mean TTFT (s)',lambda r:r['mean_TTFT_ms']/1000),
              ('Mean TBT (ms)',lambda r:r['mean_TBT_ms']),
              ('Generated output tokens/s',lambda r:r['output_tokens_s'])]
    # Use separate starts and counts, not a union of discovery and confirmation.
    rows = [r for r in data['starts'] if r['stage']=='cohort2_discovery']
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for a,m in arms:
        values=sorted([r for r in rows if r['method']==a and r['layers']==m],key=lambda r:r['QPS'])
        label=('Native' if a=='Native' else f'Layer Last{m}' if a=='Layer' else f'Token ~{ {4:889,8:1778,12:2667}[m]}')
        for ax,(ylabel,get) in zip(axes.flat,panels):
            ax.plot([r['QPS'] for r in values],[get(r) for r in values],marker='o',
                color=colors[m],linestyle='--' if a=='Token' else '-',label=label)
            ax.set(xlabel='Offered QPS (requests/s)',ylabel=ylabel,xticks=range(2,11))
            ax.grid(alpha=.2)
    axes[0,0].set_ylim(-2,102);axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Qwen3-8B | 2 x H100 | Input8000 / Output16 | Poisson\nActual P1 / D-prefill1 / Decode2 | B8192/S1024')
    fig.text(.5,.015,'MEASURED: N72, one independent start per point. Markers only at QPS2/6/10. Lines guide the eye. Finite-cohort drain rates.',ha='center',fontsize=8)
    save(fig,'dense_ar_cohort2_batch_matched_four_metrics_o16','MEASURED',[r['name'] for r in rows])
    groups=[('batch1_confirmation',16,'Decode1 / O16','#0072B2'),
            ('batch1_confirmation',64,'Decode1 / O64','#D55E00'),
            ('cohort2_confirmation',16,'Decode2 / O16','#009E73')]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    ratio_keys=[('TTFT_p95_ms','Layer / Token TTFT p95'),('output_tokens_s','Layer / Token raw token/s'),('SLO','Layer / Token SLO token/s')]
    used=[]
    for offset,(stage,cap,label,color) in zip((-.13,0,.13),groups):
        values=sorted([r for r in data['comparisons'] if r['stage']==stage and r['cap']==cap and r['reference']=='Token'],key=lambda r:r['layers'])
        for ax,(metric,ylabel) in zip(axes,ratio_keys):
            for r in values:
                value=r['metrics'][metric] if metric!='SLO' else r['regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput']
                audit=sensitivity[key(r)]
                bounds=(audit['metrics'][metric]['empirical_CDF_CI95'] if metric!='SLO' else audit['SLO_goodput_ratios']['Common_Balanced_OutputSweep']['empirical_CDF_CI95'])
                x=r['layers']+offset;y=value['ratio']
                if bounds:
                    ax.errorbar(x,y,yerr=[[y-bounds[0]],[bounds[1]-y]],color='#aaaaaa',capsize=5,lw=4,zorder=1)
                ci=value['ratio_CI95']
                ax.errorbar(x,y,yerr=[[y-ci[0]],[ci[1]-y]],color=color,marker='o',capsize=3,
                    label=label if r['layers']==4 else None,zorder=2)
            ax.axhline(1,color='black',lw=.8,ls='--');ax.set(xlabel='Fixed D suffix layers',ylabel=ylabel,xticks=(4,8,12));ax.grid(alpha=.2)
        used.extend(n for r in values for a in data['aggregate']
                    if a['stage']==stage and a['cap']==cap and a['method'] in ('Layer','Token')
                    and a['layers']==r['layers'] for n in a['names'])
    axes[0].legend(fontsize=8)
    fig.suptitle('Matched preparation batches | QPS6 | N144 x 3 independent starts\nTTFT lower favors Layer; raw/SLO throughput higher favors Layer')
    fig.text(.5,.015,'MEASURED finite cohorts. Colored: original Type7 pointwise 95% CI. Gray: inverse-CDF sensitivity. No sustainable-capacity claim.',ha='center',fontsize=8)
    save(fig,'dense_ar_actual_batch_matched_confirmation_ratios','MEASURED',used)
    selected=[r for r in data['aggregate'] if r['stage']=='cohort2_confirmation']
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    specs=[('TTFT_p95_ms','TTFT p95 (s)',.001),('ITL_p99_ms','ITL p99 (ms)',1),
           ('output_tokens_s','Raw generated output tokens/s',1),('SLO','SLO-qualified output tokens/s',1)]
    for ax,(metric,ylabel,scale) in zip(axes.flat,specs):
        for i,(a,m) in enumerate(arms):
            r=next(r for r in selected if r['method']==a and r['layers']==m)
            v=r['metrics'][metric] if metric!='SLO' else r['regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput']
            y=v['mean']*scale;ci=v['CI95']
            ax.bar(i,y,color=colors[m],alpha=.5 if a=='Token' else .9,hatch='//' if a=='Token' else '')
            ax.errorbar(i,y,yerr=[[max(0,y-ci[0]*scale)],[max(0,ci[1]*scale-y)]],color='black',capsize=3)
        ax.set_xticks(range(7),['Native','L4','T889','L8','T1778','L12','T2667']);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Qwen3-8B | Input8000 / Output16 | QPS6 | N144 x 3 starts\nActual P1 / D-prefill1 / Decode2 | B8192/S1024')
    fig.text(.5,.015,'MEASURED finite-cohort arrival-to-drain rates. Pointwise independent-start 95% CI. SLO: TTFT5000 / every ITL34 / TPOT33 ms.',ha='center',fontsize=8)
    save(fig,'dense_ar_cohort2_confirmation_metrics','MEASURED',[n for r in selected for n in r['names']])
    fig,ax=plt.subplots(figsize=(13,5.3))
    components=[('arrival_to_P_host_submit','Arrival to P submit','#a6cee3'),
                ('P_submit_to_source_notification','P submit to ready','#1f78b4'),
                ('source_notification_to_D_host_submit','Ready to D submit','#b2df8a'),
                ('D_submit_to_first_worker_notice','D submit to first worker','#6a3d9a'),
                ('first_worker_notice_to_client','Worker to client','#ff7f00')]
    bottom=np.zeros(7);used=[]
    for field,label,color in components:
        y=[]
        for a,m in arms:
            rows=[r for r in root['host_starts'] if r['stage']=='cohort2_confirmation' and r['method']==a and r['layers']==m]
            y.append(statistics.mean(r['host_path_mean_ms'][field] for r in rows)/1000)
            used.extend(r['name'] for r in rows)
        ax.bar(range(7),y,bottom=bottom,color=color,label=label);bottom+=np.array(y)
    ax.set_xticks(range(7),['Native','L4','T889','L8','T1778','L12','T2667'])
    ax.set_ylabel('Mean TTFT host-clock identity (s)');ax.legend(fontsize=8,ncol=2);ax.grid(axis='y',alpha=.2)
    fig.suptitle('TRACE_ANALYSIS | Actual preparation1 / Decode2 | Output16 / QPS6\nMean of three independent-start host identities, not exclusive GPU time')
    fig.text(.5,.015,'The five mean spans sum to mean TTFT. Component p95 values are not added. Worker-to-client includes paid first-token barrier waiting.',ha='center',fontsize=8)
    save(fig,'dense_ar_matched_cohort2_host_critical_path','TRACE_ANALYSIS',sorted(set(used)))
    operator_name='isolated_serving_projection_API_20261118_v1'
    operators=load(ROOT/'raw'/operator_name/'operator_geometry.json')['results']
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    names=sorted({r['projection'] for r in operators if r['kind']=='GEMM'})
    for ax,name in zip(axes.flat,names):
        for device,color in ((0,'#0072B2'),(1,'#D55E00')):
            rows=sorted([r for r in operators if r['kind']=='GEMM' and r['projection']==name and r['GPU']==device],key=lambda r:r['M'])
            ax.plot([r['M'] for r in rows],[r['enclosure_effective_TFLOPs_s'] for r in rows],marker='o',ms=3,color=color,label=f'GPU{device}')
        ax.set_xscale('log');ax.set(xlabel='Actual GEMM M rows (log scale)',ylabel='FLOPs / event enclosure (TFLOP/s)',title=name.split('layers.0.')[-1]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('MEASURED isolated original BF16 projections | Actual Serving linear_batch_invariant API\nOriginal layer0 weights and captured activation rows | Not online serving speedup')
    fig.text(.5,.015,'One process; GPU0/1 measured sequentially. Five warmups and 30 event samples per shape. No independent-start CI. No SM/HBM counter inference.',ha='center',fontsize=8)
    save(fig,'dense_ar_actual_serving_projection_shape_efficiency','MEASURED',[operator_name])
    # Keep the Output200 motivation separate from actual-batch controls.
    for m in (4,8,12):
        rows=[r for r in data['starts'] if r['stage']=='discovery' and r['cap']==200 and r['count']==144 and r['admission']==8 and (r['method'] in ('Agg','Native') or r['layers']==m)]
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for a,color in [('Agg','#d95f02'),('Native','#009e73'),('Layer','#0072b2'),('Token','#8c56a0')]:
            values={r['QPS']:r for r in rows if r['method']==a}
            label=a+(f' Last{m}' if a=='Layer' else f' ~{ {4:889,8:1778,12:2667}[m]}' if a=='Token' else '')
            for ax,(ylabel,get) in zip(axes.flat,panels):
                ax.plot(range(2,11),[get(values[q]) if q in values else math.nan for q in range(2,11)],marker='o',color=color,label=label)
                ax.set(xlabel='Offered QPS (requests/s)',ylabel=ylabel,xticks=range(2,11));ax.grid(alpha=.2)
        axes[0,0].set_ylim(-2,102);axes[0,0].legend(fontsize=8)
        fig.suptitle(f'Qwen3-8B | 2 x H100 | Input8000 / Output200 | Poisson Arrivals\nAll B8192/S1024 | Initial common PD cap8, actual batches UNMATCHED | Last{m}')
        fig.text(.5,.015,'MEASURED: N144, one start per point. QPS3/5/7/9 NOT_RUN. No fabricated interpolation. SLO5000/34/33ms. Finite-cohort rates.',ha='center',fontsize=8)
        save(fig,f'dense_ar_unified_four_metrics_o200_last{m}','MEASURED',[r['name'] for r in rows])
    # The requested main example remains Last12, not a result-selected pair.
    import shutil
    for ext in ('png','pdf'):
        src=REPO/'poc_flashvep'/f'dense_ar_unified_four_metrics_o200_last12.{ext}'
        target=REPO/'poc_flashvep'/f'dense_ar_unified_qps_four_metrics_o200.{ext}'
        shutil.copyfile(src,target)
        figures.append(dict(path=str(target.relative_to(REPO)),SHA256=digest(target),evidence='MEASURED',source='Prespecified Last12 main example'))
    write(ROOT/'analysis/scoped_final_figure_manifest.json',dict(epoch=time.time(),files=figures,
        statistics_SHA256=digest(ROOT/'analysis/independent_start_statistics.json'),
        scope='Keep batch1, preparation1/Decode2 and initial unmatched cap8 results separate. Measured points only. No CI at n1. Root copies are stored at the user-requested poc_flashvep location.'))
    return figures


if __name__ == '__main__':
    main()
