"""Prepare evidence-linked drafts while the frozen GPU matrix continues."""
import argparse
import datetime
import json
import math
import os
import statistics
from .common import ROOT, REPO, BASE, BRANCH, MODEL_REVISION, preserve


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def number(value, digits=3):
    if value is None or not math.isfinite(value):
        return 'NOT_MEASURED'
    return f'{value:.{digits}f}'


def interval(value):
    ci=value.get('ratio_CI95')
    return number(value.get('ratio'),6)+(' ['+', '.join(number(x,6) for x in ci)+']' if ci else ' [CI: NOT_MEASURED]')


def table(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |',
        '| '+' | '.join('---' for _ in columns)+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    preserve()
    from .quantile_audit import main as quantile_audit
    quantile_audit()
    data=read(ROOT/'analysis/independent_start_statistics.json',dict(starts=[],aggregate=[],comparisons=[]))
    sensitivity=read(ROOT/'analysis/CI_quantile_sensitivity.json')['comparisons']
    root=read(ROOT/'analysis/root_cause_components.json',dict(crossover=[]))
    if args.final:
        assert read(ROOT/'analysis/final_integrity_validation.json',{}).get('passed')
        assert read(ROOT/'state/CAMPAIGN_CLOSED.json',{}).get('status')=='CLOSED'
        assert read(ROOT/'state/final_GPU_resource_restoration.json',{}).get('passed')
    destination=REPO/'poc_flashvep/reports' if args.final else ROOT/'draft_reports'
    destination.mkdir(exist_ok=True)
    def link(path,label=None):
        assert path.exists(),path
        return f'[{label or path.name}]({os.path.relpath(path,destination)})'
    def document(name,title,body):
        path=destination/name
        status='최종 검증 후 생성한 보고서이다.' if args.final else '진행 중 초안이다. 완료하지 않은 조건과 최종 판정을 구분한다.'
        path.write_text('# '+title+'\n\n'+status+'\n\n'+body.strip()+'\n')
    aggregate=[r for r in data['aggregate'] if r['count']==144 and r['admission']==8]
    def selected(method,layers,cap,qps):
        rows=[r for r in aggregate if r['method']==method and r['layers']==layers and r['cap']==cap and r['QPS']==qps]
        confirmed=[r for r in rows if r['stage']=='confirmation' and r['starts']>=3]
        return (confirmed or [r for r in rows if r['stage']=='discovery'] or [None])[0]
    def latency_row(row):
        return [row['method']+(str(row['layers']) if row['layers'] else ''),row['cap'],row['QPS'],
            row['stage'],row['starts'],*[
                number(row['metrics'][k]['mean']) for k in ('mean_TTFT_ms','TTFT_p95_ms',
                    'mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')],
            number(100*row['regimes']['Common_Balanced_OutputSweep']['Joint_attainment']['mean']),
            number(row['regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput']['mean'])]
    columns=['Method','Output','QPS','Stage','Starts','Mean TTFT ms','TTFT p95 ms',
        'Mean TBT ms','ITL p99 ms','TPOT p95 ms','Output tokens/s','Joint SLO %','SLO tokens/s']
    primary=[r for r in data['starts'] if r['stage'] in ('discovery','motivation_odd','confirmation')]
    discovery=sum(r['stage']!='confirmation' for r in primary)
    confirmation=sum(r['stage']=='confirmation' for r in primary)
    common=f'''## 공통 실행 조건

Model: Qwen/Qwen3-8B. Revision: `{MODEL_REVISION}`.
Original BF16, Exact Causal Attention, Greedy, Thinking Disabled, Full36 Decode를 유지한다.
Input8000과 Output16/32/64/128/200을 사용한다. `ignore_eos=True`이다.
GPU0/1은 H100 SXM80GB이다. KV Budget은 GPU당42GiB이다.
GPU UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`와 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`이다.
Runtime은 vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1이다.
모든 Worker에 B8192/S1024를 적용한다. Prefix Cache는 비활성화한다.
PD의 공통 D Admission 상한은8이다. 실제 준비 Group과 Decode Cohort는 다를 수 있다.
실제 모델 Context Limit은40960이다. Worker의 실행 한도는 Input+Output에서 정한다.
따라서 Output16과 Output200의 Worker 한도는 각각8016과8200이다.
같은 Output에서는 모든 비교 방법의 한도가 같다.

Branch: `{BRANCH}`.
Publication Parent: `{BASE}`.
Parent SHA를 전체 실행 Source SHA라고 표시하지 않는다.
각 실행의 `command.json`에 실제 `source.zip` SHA256을 저장한다.
Worker별 실제 설정, Prompt Hash, Seed 및 설정 Hash도 원시 Evidence에 저장한다.
{link(ROOT/'state/environment.json')}에 GPU와 Model 설정을 보존한다.
{link(ROOT/'state/frozen_new_inference_sources_v3.json')}에 동결한 Inference Source Hash를 보존한다.

## 표본과 지표

성능 비교는 실제 문서144개를 사용한다. 동일 Prompt를 독립 Start로 세지 않는다.
기존 Held-out128개에 기존 Discovery의 다른 문서16개를 추가했다.
새로운 미관측 Prompt Pool이라고 주장하지 않는다.
Discovery Seed는2026111780이다.
QPS6 Confirmation Seed는2026111790/2026111791/2026111792이다.
Discovery와 Confirmation은 CI 계산에서 합치지 않는다.
CI의 재표본 단위는 독립 Server Start이다. n1에는 CI가 없다.
표의 p95/p99는 Start 안에서 계산한 분위수를 Start 사이에서 평균한 값이다.
기존 CI는 Bootstrap Draw 값의 Type7 선형 보간을 유지한다.
별도 역CDF 분위수 민감도 분석도 같은 독립 Start에 적용한다.
두 방식에서 결론이 바뀌면 확정적인 우위나 비열등성으로 판단하지 않는다.
표는 표시용으로 반올림한다. 반올림하지 않은 값과 CI는 JSON에 보존한다.
Raw Output tokens/s는 전체 유한 Cohort의 Arrival-to-drain 처리율이다.
이를 Maximum Sustainable Capacity라고 표시하지 않는다.
Client-visible ITL에는 첫 Token Interval을 포함한다.
Worker Notice 시간과 Client Delivery 시간은 별도로 보존한다.

현재 집계한 Discovery는{discovery}/232회이다.
현재 집계한 Confirmation은{confirmation}/120회이다.
표에 없는 예정 조건은 `NOT_RUN`이다.
상태 Matrix는 {link(ROOT/'analysis/full_execution_matrix.csv')}에서 확인한다.
개별 요청은 {link(ROOT/'analysis/per_request_metrics_and_work.csv')}에서 확인한다.
독립 Start 통계는 {link(ROOT/'analysis/independent_start_statistics.json')}에서 확인한다.

## SLO 계약

기존 Output별 Frozen SLO를 변경하지 않았다.
Output200 Balanced는 TTFT5000ms, 모든 ITL34ms, TPOT33ms이다.
Output Sweep의 공통 보조 SLO도5000/34/33ms이다.
기존 Output64 Balanced의 ITL35ms와 공통 보조34ms를 구분한다.
Goodput은 세 조건을 모두 통과한 요청의 Output Token 수를 분자로 사용한다.
Reference Goodput이0이면 비율은 정의하지 않는다.
Native-relative 보호는 ITL p99와 TPOT p95 비율의95% CI 상한을 검사한다.
두 상한이 모두1.05 이하여야 PASS이다. 최소3개 독립 Start가 필요하다.
Absolute SLO와 Relative 비열등성은 서로 다른 Gate이다.
{link(ROOT/'configs/frozen_SLO_regimes.json')}와 {link(ROOT/'configs/common_output_sweep_SLO.json')}을 보존한다.
'''
    recovery=ROOT/'state/daemon_recovery_20261118.json'
    if recovery.exists():
        common+='\n## 외부 중단과 재개\n\n'
        common+='제어 세션 재시작으로 중단한 실행1건은 FAILED로 보존한다.\n'
        common+='원래 Manifest Slot은 유지하고, name만 다른 대체 실행을 연결한다.\n'
        common+='중단한 부분 결과는 독립 표본이나 Clean 성능에 포함하지 않는다.\n'
        common+='완료한 기존 GPU 실행은 반복하지 않는다.\n'
        common+=link(recovery,'복구 기록')+'에 원래 Receipt Hash와 대체 이름을 저장한다.\n'
    q6=[selected(a,m,cap,6) for cap in (16,32,64,128,200)
        for a,m in [('Agg',0),('Native',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]]
    q6_table=table(columns,[latency_row(r) for r in q6 if r])
    ratios=[r for r in data['comparisons'] if r['reference']=='Token' and r['candidate']=='Layer' and r['count']==144 and r['admission']==8]
    ratio_table=table(['Stage','Last','Output','QPS','Starts','Layer/Token TTFT p95 [95% CI]',
        'Layer/Token common-SLO tokens/s [95% CI]'],[
        [r['stage'],r['layers'],r['cap'],r['QPS'],r['starts'],interval(r['metrics']['TTFT_p95_ms']),
         interval(r['regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput'])]
        for r in ratios])
    ni=table(['Stage','Candidate','Last','Output','QPS','Starts','ITL p99 ratio [95% CI]',
        'TPOT p95 ratio [95% CI]','Native-relative Gate'],[
        [r['stage'],r['candidate'],r['layers'],r['cap'],r['QPS'],r['starts'],
         interval(r['metrics']['ITL_p99_ms']),interval(r['metrics']['TPOT_p95_ms']),r['Native_relative_5pct']]
        for r in data['comparisons'] if r['reference']=='Native' and r['count']==144 and r['admission']==8])
    work=read(ROOT/'analysis/matched_axis_structural_work.json',dict(rows=[]))['rows']
    structural=table(['Last','Token Suffix Histogram (N144)','D GEMM Rows / method',
        'D Token/Layer causal Attention pairs','Extra Layer Hidden/Residual bytes'],[
        [r['layers'],json.dumps(r['token_suffix_histogram'],sort_keys=True),
         r['totals']['layer_D_GEMM_rows'],number(r['D_token_to_layer_attention_pair_ratio'],6),
         144*8000*4096*2*2] for r in work])
    geometry='''## 실행 Geometry와 해석 범위

B8192는 Iteration Token Budget이다. S1024는 Scheduler Sequence 상한이다.
둘 다 실제 GPU Batch 수를 뜻하지 않는다.
현재 D Runtime은 준비 Group과 이후 Decode Cohort를 결합한다.
새 준비 Group은 이전 Decode Cohort가 종료한 뒤 수용한다.
Layer의 Prompt Query8000개 때문에 준비 Group은1개 요청이다.
Token≈889/1778/2667의 준비 Group은 각각 최대8/4/3개 요청이다.
Native의 Cached Query는1개이며 공통 상한8개 요청을 사용할 수 있다.
각 Group의 실제 크기는 Ready-work와 종료 경계에 따라 더 작을 수 있다.
따라서 공통 Admission Policy를 사용했지만 실제 Geometry는 일치하지 않는다.
이 비교를 `GEOMETRY_MATCHED`라고 표시하지 않는다.

이 구조는 현재 구현의 제한이다.
Layer-axis가 원리적으로 Batch1만 사용할 수 있다는 뜻은 아니다.
여러 Iteration에서 Layer 요청을 준비한 뒤 큰 Decode Cohort로 합치는 경로는 구현하지 않았다.
이번 캠페인에서 이 실행 구조를 조용히 변경하지 않는다.
P의 외부 Queue 증가만으로 P Compute Saturation을 주장하지 않는다.
D Reservation, Source Window32 및 Cohort Backpressure가 P 제출을 제한할 수 있다.
'''
    document('dense_ar_matched_work_output_sweep_results.md','Matched-work Output Sweep 결과',common+'\n'+geometry+f'''
## QPS6 요약

{q6_table}

## 모든 완료 Pair 비교

{ratio_table}

## Native-relative Decode 보호

{ni}

독립 반복을 완료하지 않은 점의 우위를 확정하지 않는다.
Layer의 TTFT 우위와 Native-relative Decode 보호를 별도로 검사한다.
실행별 actual query coverage, full migration 및 KV reservation Gate를 확인한다.
한 요청의 Prompt Query가 누락되거나 중복되면 적격 결과에서 제외한다.
''')
    controls=[r for r in data['comparisons'] if r['stage']=='admission_control' and r['reference']=='Token' and r['candidate']=='Layer']
    control_table=table(['Last','Output','Admission cap','Starts','Layer/Token TTFT p95 [95% CI]'],[
        [r['layers'],r['cap'],r['admission'],r['starts'],interval(r['metrics']['TTFT_p95_ms'])] for r in controls])
    host_rows=[]
    for cap in (16,32,64,128,200):
        for a,m in [('Native',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]:
            items=[r for r in root.get('host_starts',[]) if r['method']==a and r['layers']==m and r['cap']==cap
                and r['QPS']==6 and r['count']==144 and r['admission']==8 and 'D_host_span_request_rate_ceiling' in r]
            confirmed=[r for r in items if r['stage']=='confirmation']
            items=confirmed if len(confirmed)>=3 else [r for r in items if r['stage']=='discovery']
            if not items:continue
            def avg(key):
                values=[r[key] for r in items]
                return statistics.mean(values) if all(v is not None for v in values) else None
            host_rows.append([a+str(m),cap,len(items),number(avg('summed_D_preparation_per_request_ms')),
                number(avg('summed_decode_residency_per_request_ms')),
                number(avg('D_host_span_request_rate_ceiling')),number(avg('actual_finite_cohort_completed_QPS'))])
    host_table=table(['Method','Output','Starts','D prep ms / request','Decode residency ms / request',
        'D Host-span rate ceiling req/s','Observed finite-cohort req/s'],host_rows)
    verdicts=table(['Stage','Last','QPS','Admission','판정','Layer favored Outputs','Token favored Outputs'],[
        [r['stage'],r['layers'],r['QPS'],r['admission'],r['verdict'],r['Layer_favored_outputs'],r['Token_favored_outputs']]
        for r in root['crossover']])
    document('dense_ar_layer_token_crossover_root_cause.md','Layer/Token Crossover 원인 분석',common+'\n'+geometry+f'''
## 같은 GEMM Rows와 다른 Attention 분배 — STRUCTURAL_ACCOUNTING

{structural}

전체 모델의 Causal Attention Pair 합은 두 방법에서 같다.
P와 D의 작업 분배는 다르다.
Token/Layer Attention Pair 비율을 GPU 시간 비율로 바꾸지 않는다.
Layer는 각 요청에 Hidden/Residual131072000bytes를 추가 전송한다.
정확한 N144 Matched Plan에서는 유효 KV Byte 합이 두 방법에서 같다.
이 값은 Source KV Credit 점유량이 같다는 뜻이 아니다.

## Admission 통제 — MEASURED

기존 공통 cap8 Matrix를 유지한다.
별도 cap1 실험에서 두 방법의 실제 Decode Cohort Geometry 차이를 줄인다.
Last4/8은 Output16/200에서 각1개 Start이다.
Last12는 사전 선정한 Seed2026111787/1788/1789에서 확인한다.
동일한 Seed의 Native cap1/cap8도 비교한다.
독립 반복을 끝내지 않은 결과에는 확정적 원인 판정을 적용하지 않는다.

{control_table}

## Host Critical Path — TRACE_ANALYSIS

Arrival → P Host Submit → Source Ready Notice → D Host Submit → First Worker Notice → First Client Output을 추적한다.
이 다섯 구간의 합은 해당 요청의 TTFT와 일치해야 한다.
평균 구간 차이는 Host Clock의 분해이다. Exclusive GPU Compute 원인으로 표시하지 않는다.
p95 구간을 더해서 TTFT p95를 설명하지 않는다.
자기 요청의 첫 Token 이후 Decode는 자기 TTFT에 포함되지 않는다.
이전 Cohort의 Decode Residency는 다음 요청의 D 대기를 늘릴 수 있다.
Output Length는 이 대기 증폭을 바꿀 수 있다.
{link(ROOT/'analysis/root_cause_components.json')}에 같은 Start의 구간을 저장한다.

{host_table}

D Host-span Rate Ceiling은 요청 수를 서로 겹치지 않는 D Host Span 합으로 나눈 값이다.
여기에는 Host와 Transfer 대기도 포함된다.
이는 해당 실행의 시간 회계 상한이다. 물리적 GPU 처리 용량이나 Sustainable Capacity가 아니다.
실제 Group 크기가 달라지면 요청당 Decode Residency의 분모도 달라진다.
Raw 처리율과 이 상한의 차이는 Idle, 경계 및 Host Gap도 포함한다.
이를 Exclusive GPU Compute 효율로 표시하지 않는다.

## GEMM과 Handoff의 증거 한계

고립 실험의 원래 Layer0 Projection Median 합은 구조적 성분 추정이다.
이를 실제 전체 Suffix 실행시간으로 표시하지 않는다.
별도 Last2/Last4 Serial Diagnostic은 실제 P→D Continuation의 CUDA Event Enclosure를 측정했다.
그 측정에는 Attention, Norm, Activation 및 Launch/Stream 간격도 포함된다.
그 측정을 순수 GEMM 시간이나 QPS Sweep의 TTFT 이득으로 대체하지 않는다.
NIXL Telemetry와 제출 Byte는 실제 기록이다.
동시 Transfer Duration을 더해서 요청별 Wire Time으로 표시하지 않는다.
Handoff, Admission 및 Host Scheduling 비용을 구분한다.

## Profiler와 Crossover 판정

온라인 Profiler의 Clean 대비 전체5% 간섭 Gate를 통과하지 못했다.
Raw Failure와 별도 사후 PID 매핑을 보존한다.
개선한 API 배치 Pilot도 Maximum ITL Screen을 통과하지 못했다.
따라서 Kernel Timeline은 `DIAGNOSTIC`이다.
신뢰 가능한 물리적 P Busy, D Idle 및 SM/HBM Counter는 `NOT_MEASURED`이다.
Host State를 물리적 GPU Idle Budget으로 표시하지 않는다.

{verdicts}

이 표의 TTFT 판정은 기존 Type7 CI 기준이다.
역CDF 민감도 분석과 Native-relative Decode 보호는 별도로 검사한다.
이 표만으로 연구 GO를 선언하지 않는다.

TTFT의 통계적 Crossover와 Decode-safe Serving 우위는 서로 다른 판정이다.
공통 cap8의 결론을 모든 Token Streaming Pipeline으로 확대하지 않는다.
''')
    motivation=[]
    for qps in range(2,11):
        for a,m in [('Agg',0),('Native',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]:
            row=selected(a,m,200,qps)
            if row:motivation.append(latency_row(row))
    document('dense_ar_unified_budget_duetserve_motivation_results.md','Unified-budget DuetServe Motivation 결과',common+f'''
## 네 Panel과 데이터 선택

X축은 Offered QPS2–10이다.
Panel은 Joint SLO Attainment, Mean TTFT, Mean TBT, Generated Output tokens/s이다.
Last4/8/12 Pair를 각각 같은 수준으로 제시한다.
기본 Last12 Pair는 결과를 본 뒤 유리한 Cut으로 선정한 것이 아니다.
B512/S16은 이 Figure에 포함하지 않는다.
Agg의 주황색 Curve도 새 Prompt/Arrival 계약으로 재측정한다.
기존 Historical Figure는 덮어쓰지 않는다.
QPS6에서 Confirmation3회가 완료되면 해당 평균과 CI를 사용한다.
다른 점은 같은 Discovery Seed의 n1이다.
따라서 모든 Curve 점을 동일 Seed의 단일 Sweep이라고 표시하지 않는다.
미측정 점은 비워 둔다. 실측 점처럼 보간하지 않는다.

## Output200 완료 조건

{table(columns,motivation)}

## 기존 Native와의 차이

새 Native는 실제 양 Worker의 B8192/S1024를 검사한다.
예전32000/4와128000/16 설정으로 실행한 시간을 새 결과에 합치지 않는다.
별도 Bridge는 같은 과거 Prompt와 Arrival Trace에서 새 Native를 재측정한다.
이 비교에는 B/S 외에 Admission, Source Packing, Reservation 및 측정 Epoch 차이도 있다.
따라서 Bridge를 B만의 인과 효과라고 주장하지 않는다.
DuetServe Figure2는 DP2이다. TP2 결과로 바꾸어 설명하지 않는다.
우리 vLLM/NIXL 경로가 원 논문의 Runtime과 완전히 같다고 주장하지 않는다.
원문은 [DuetServe](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)이다.
''')
    figure_rows=[]
    for name in ('figure_manifest.json','root_figure_manifest.json'):
        manifest=read(ROOT/'analysis'/name,dict(figures=[]))
        for row in manifest['figures']:
            stem=row['stem']
            png=ROOT/'figures'/(stem+'.png');pdf=ROOT/'figures'/(stem+'.pdf')
            if png.exists() and pdf.exists():
                figure_rows.append([stem,row['evidence'],link(png,'PNG'),link(pdf,'PDF')])
    document('dense_ar_matched_axis_output_sweep_figures.md','Matched-axis Figures와 통계',common+f'''
## Figure Manifest

{table(['Figure','Evidence','PNG','PDF'],figure_rows)}

기본 네 Panel의 파일은 Repository의 `poc_flashvep/`에도 저장한다.
파일명은 `dense_ar_unified_four_metrics_o200_last4/8/12.png`와 같은 규칙을 사용한다.
기본 Alias는 `dense_ar_unified_qps_four_metrics_o200.png`이다.
각 PNG의 PDF도 생성한다.
원시 Figure Manifest는 분석 구간, Seed, Start 수와 Evidence를 보존한다.
고립 Operator Figure는 Online Serving 우위가 아니다.
GPU Kernel Timeline과 Host Enclosure를 같은 물리적 계측으로 표시하지 않는다.
실측, 구조적 계산, Diagnostic 및 미측정 결과를 구분한다.
''')
    confirmed=[r for r in ratios if r['stage']=='confirmation' and r['starts']>=3]
    lwin=[r for r in confirmed if r['metrics']['TTFT_p95_ms']['ratio_CI95'] and r['metrics']['TTFT_p95_ms']['ratio_CI95'][1]<1]
    twin=[r for r in confirmed if r['metrics']['TTFT_p95_ms']['ratio_CI95'] and r['metrics']['TTFT_p95_ms']['ratio_CI95'][0]>1]
    qualified=[];qualified_goodput=[]
    for r in lwin:
        safety=[n for n in data['comparisons'] if n['stage']=='confirmation' and n['reference']=='Native' and n['candidate']=='Layer'
            and all(n[k]==r[k] for k in ('layers','cap','QPS','count','admission'))]
        def same(row):
            return all(row[k]==r[k] for k in ('stage','layers','cap','QPS','count','admission'))
        cdf=[n for n in sensitivity if same(n) and n['reference']=='Token' and n['candidate']=='Layer']
        native_cdf=[n for n in sensitivity if same(n) and n['reference']=='Native' and n['candidate']=='Layer']
        robust=(len(cdf)==1 and cdf[0]['metrics']['TTFT_p95_ms']['empirical_CDF_CI95'] is not None
            and cdf[0]['metrics']['TTFT_p95_ms']['empirical_CDF_CI95'][1]<1 and len(native_cdf)==1
            and native_cdf[0]['Native_relative_empirical_CDF_5pct']=='PASS')
        if len(safety)==1 and safety[0]['Native_relative_5pct']=='PASS' and robust:qualified.append(r)
    for r in sensitivity:
        if r['stage']!='confirmation' or r['reference']!='Token' or r['candidate']!='Layer' or r['starts']<3:continue
        g=r['SLO_goodput_ratios']['Common_Balanced_OutputSweep']
        if not(g['linear_CI95'] and g['empirical_CDF_CI95'] and g['linear_CI95'][0]>1 and g['empirical_CDF_CI95'][0]>1):continue
        safety=[n for n in sensitivity if n['reference']=='Native' and n['candidate']=='Layer'
            and all(n[k]==r[k] for k in ('stage','layers','cap','QPS','count','admission'))]
        if len(safety)==1 and safety[0]['original_Native_relative_5pct']=='PASS' and safety[0]['Native_relative_empirical_CDF_5pct']=='PASS':qualified_goodput.append(r)
    verdict='INCONCLUSIVE_PENDING_CONFIRMATION' if confirmation<120 else 'LAYER_FAVORED_ON_QUALIFIED_CONFIRMED_POINTS' if qualified or qualified_goodput else 'NO_QUALIFIED_LAYER_ADVANTAGE_ON_CONFIRMED_POINTS'
    summary=f'''## 판정

현재 판정은 `{verdict}`이다.
완료한 Confirmation에서 Type7 TTFT CI가 Layer를 지지하는 점은{len(lwin)}개이다.
Type7 TTFT CI가 Token을 지지하는 점은{len(twin)}개이다.
Layer TTFT 우위와 Native-relative Decode 보호를 함께 통과한 점은{len(qualified)}개이다.
공통 SLO Goodput 우위와 Native-relative Decode 보호를 함께 통과한 점은{len(qualified_goodput)}개이다.
두 통과 수에는 역CDF 민감도 점검도 적용한다.
이 수는 독립적인 전체 Workload 개수를 뜻하지 않는다.
조건별 CI는 Pointwise이며 다중 비교를 보정하지 않았다.
Absolute SLO와 Goodput 비율은 별도 표에서 확인한다.
모든 조건의 전역적 축 우위나 Sustainable Capacity를 주장하지 않는다.

## QPS6 전체 Output 비교

{q6_table}

## 동일 작업량과 실행 구조

{structural}

{geometry}

## 성공·실패·미실행의 구분

Full-state 정합성은 N8→N32→N128과 Output200 N8에서 검사했다.
Full Prompt KV, Final Hidden, Greedy Output, Layer 방문, Full36 Decode, ACK와 정상 Retirement를 검사했다.
성능 Start마다 실제 B/S, Output, Query Coverage, Matched Rows와 Credit Lifetime도 검사한다.
정상 종료만으로 계측 간섭 Gate를 통과했다고 표시하지 않는다.
실패한 Profiler와 Optional Projection Observer 기록은 보존한다.
정합성 실패 결과를 적격 성능에 합치지 않는다.
Dynamic Scheduler 개선, TP2 Strict Parity 및 새 Streaming Token Pipeline은 `NOT_RUN`이다.
Last2 Full Sweep은 사용자가 취소했다. 해당 추가 GPU 실행은0회이다.
별도 완료한 Serial Last2/4 Diagnostic은 원래 Sweep과 합치지 않는다.

## 다음 연구에서 필요한 검증

큰 D Decode Cohort를 유지하면서 Layer 준비를 여러 Iteration으로 분리할 수 있는지 검토할 수 있다.
이는 이번 구현에서 측정한 결과가 아니다.
그 구조를 구현하면 Token도 같은 Runtime과 Admission에서 다시 비교해야 한다.
현재 결과만으로 Layer×Token Dynamic Selection의 Serving 이득을 주장하지 않는다.

## 보존과 전달

이 보고서는 기존 사용자 Checkout과 Index를 바꾸지 않는 별도 Branch에 저장한다.
Per-job Command, Source ZIP, Hash, Worker Log, Prompt/Arrival 및 실패 기록을 보존한다.
큰 Evidence는 원본 SHA256과 압축 해제 SHA256을 검사한 Archive로 전달한다.
최종 Remote SHA와 GPU 복원 Receipt는 별도로 검사한다.
GPU0–7의 승인 Burner가 실제 실행 중인지 확인한 뒤 종료한다.
'''
    summary+='''
## 요청한15개 질문의 답

1. B8192/S1024: 완료된 Start의 실제 양 Worker 설정을 검사한다. 미실행 조건까지 통과했다고 표시하지 않는다.
2. D Admission: 공통 상한은8이다. 실제 Layer 준비 Group1과 Token 최대8/4/3은 같지 않다.
3. 기존 Native와의 차이: 별도 Historical Bridge로 비교한다. B/S만의 단독 인과 효과는 아니다.
4. Last4/Token≈889: 아래 Pair별 TTFT 비율과 CI를 사용한다.
5. Last8/Token≈1778: 아래 Pair별 TTFT 비율과 CI를 사용한다.
6. Last12/Token≈2667: 아래 Pair별 TTFT 비율과 CI를 사용한다.
7. Output Sweep: Output16/32/64/128/200의 모든 완료 결과를 같은 고정 Partition에서 비교한다.
8. Crossover: 독립 Start의 TTFT CI가 양 방향 우위를 각각 지지해야 통계적 역전을 확인한다.
9. 원인 분리: 같은 B/S와 cap8은 실제 Cohort Geometry를 일치시키지 못한다. cap1 통제를 별도로 검사한다.
10. GEMM/Attention/Transfer/Queue: 구조적 작업량과 Host 경로를 구분한다. Exclusive GPU 성분별 원인 비율은 확인하지 못했다.
11. 같은 SLO의 Goodput: Common SLO 비율과 CI를 사용한다. Reference0인 비율을 개선0%로 처리하지 않는다.
12. Output200 Motivation: 새 Agg와 Native를 같은 Trace에서 측정한다. 아직 측정하지 않은 QPS는 비워 둔다.
13. 새 네 Panel: Last4/8/12의 모든 고정 Pair를 각각 제시한다. 기본 Alias는 Last12이다.
14. 과거 Output16 Last12 우위: 이번 통일 조건의 비율로 다시 판단한다. 과거13.95%를 새 측정값으로 사용하지 않는다.
15. 관측값: 실제 D Cohort 크기, Residency, Ready-to-admit 대기, Source Window/KV Credit과 Suffix 비용을 함께 봐야 한다.

이 관측값만으로 새로운 Axis Predictor가 검증됐다는 뜻은 아니다.
물리적 GPU Idle과 SM/HBM Counter가 없으면 해당 항목은 `NOT_MEASURED`로 남긴다.

## Pair별 상세 비율

'''+ratio_table+'\n\n## Native-relative 상세 판정\n\n'+ni
    document('dense_ar_matched_axis_crossover_final_summary.md','Dense AR Matched-axis Crossover 최종 요약',common+'\n'+summary)
    sensitive=[]
    for r in sensitivity:
        if r['starts']<3:continue
        metric=r['metrics']['TTFT_p95_ms'];a=metric['linear_CI95'];b=metric['empirical_CDF_CI95']
        changed=(a is not None and b is not None and ((a[1]<1)!=(b[1]<1) or (a[0]>1)!=(b[0]>1)))
        if changed or r['original_Native_relative_5pct']!=r['Native_relative_empirical_CDF_5pct'] and r['reference']=='Native':
            sensitive.append([r['stage'],r['reference']+'→'+r['candidate'],r['layers'],r['cap'],r['admission'],
                r['starts'],str(a),str(b),r['original_Native_relative_5pct'],r['Native_relative_empirical_CDF_5pct']])
    document('dense_ar_matched_axis_small_start_ci_sensitivity.md','소수 독립 Start의 CI 분위수 민감도',common+f'''
## 정의와 원인

기존 코드는 Type7 선형 보간 분위수를 계산한다. 산술 오류로 판정하지 않았다.
역CDF는 관측된 이산 Bootstrap 분포의 누적 확률에서 분위수를 정한다.
두 방법은 서로 다른 정의이다. [NumPy quantile 문서](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)가 두 정의를 구분한다.
3개 Start의 정확한 Bootstrap Draw는27개이다.
각 Draw를 반복해서 늘려도 이산 확률분포는 바뀌지 않는다.
그러나 Type7 보간의 끝점은 표현된 Sample 수에 따라 바뀔 수 있다.
역CDF 분석은 이 표현 변경에 영향을 받지 않는다.

발견한 N36/Last12/Output16/cap1 예에서는 기존 TTFT 비율 CI 상한이0.9995468303958746이다.
역CDF 방식의 상한은1.0109602351406097이다.
따라서 이 경계 결과를 강한 Layer 우위로 채택하지 않는다.
원시 Metric, 기존 CI 및 Source는 바꾸지 않는다.
새 점검의 시점과 원래 통계 Snapshot Hash를 별도로 보존한다.
이 점검은 Control 결과를 본 뒤 추가했다. Primary Confirmation 시작 전이다.
모든 방법과 조건에 같은 점검을 적용한다.
SLO와 비열등성 Threshold1.05는 변경하지 않는다.

## 경계 결론이 바뀐 조건

{table(['Stage','비교','Last','Output','cap','Starts','원래 TTFT CI','역CDF TTFT CI','원래 NI','역CDF NI'],sensitive)}

Start가 적다는 한계는 어느 정의를 사용해도 남는다.
정확한 Draw 열거가 모집단95% Coverage나 WCET 보장을 뜻하지 않는다.
Within-start Latency Percentile의 정의는 변경하지 않는다.
전체 민감도 기록은 {link(ROOT/'analysis/CI_quantile_sensitivity.json')}에 보존한다.
''')
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('REPORT_DRAFTS_READY' if not args.final else 'FINAL_REPORTS_READY',stamp,str(destination),flush=True)


if __name__=='__main__':
    main()
