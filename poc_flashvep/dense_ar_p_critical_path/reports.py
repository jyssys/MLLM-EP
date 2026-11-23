"""Write scoped Korean reports from current evidence,not expected outcomes."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from .common import ROOT,BASE,BRANCH,MODEL_REVISION,START,write,digest

NAMES=('dense_ar_2d_p_queue_root_cause.md','dense_ar_2d_p_gpu_operator_breakdown.md',
    'dense_ar_2d_p_admission_intervention.md','dense_ar_2d_hybrid_pipeline_optimization.md',
    'dense_ar_2d_optimized_c3_confirmation.md','dense_ar_2d_p_critical_path_final_summary.md')


def read(name):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else None


def number(value,digits=3):return 'NOT_RUN / 진행 중' if value is None else f'{value:.{digits}f}'


def estimate(value):
    if not value:return 'NOT_RUN / 진행 중'
    result=number(value.get('estimate',value.get('mean')),4);ci=value.get('CI95')
    return result+(' ['+', '.join(number(v,4) for v in ci)+']' if ci is not None else ' [CI 미확정]')+f', N={value.get("starts",0)}'


def table(headers,rows):
    if not rows:return '해당 적격 결과는 아직 없습니다. 미실행과 실행 중 상태를 구분해서 최종 검증합니다.\n'
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(str(v).replace('|',' / ').replace('\n',' ') for v in row)+' |' for row in rows])+'\n'


def link(path,folder):
    target=ROOT/path
    return '['+str(path)+']('+os.path.relpath(target,folder)+')'


def body_common(final):
    return (f'상태: {"FINAL" if final else "DRAFT — 연구 진행 중"}.\n\n'
        f'작성 시각: {datetime.now(timezone.utc).isoformat()}.\n\n'
        f'기준 Commit: `{BASE}`. 연구 Branch: `{BRANCH}`.\n\n'
        f'Model: Qwen/Qwen3-8B, Revision `{MODEL_REVISION}`.\n'
        'BI0, Original BF16, Exact Causal Attention, Greedy, Thinking Disabled를 유지합니다.\n'
        'Full36 Decode와 Prefix Cache Disabled를 유지합니다.\n'
        'B8192/S1024, GPU당 KV42GiB, KV Block128을 유지합니다.\n\n'
        'GPU Pair는 H100 SXM80GB×2입니다. 실제 GPU UUID는 Environment와 각 Raw Runtime 파일에 보존합니다.\n'
        'Primary QPS는2입니다. C3의 Actual Batch는 동적입니다. B/S를 Batch 수로 표현하지 않습니다.\n'
        'Client 시각은 Coordinator 전달 시각입니다. HTTP Serving Latency를 측정한 값은 아닙니다.\n\n'
        'LongBench400-v2는 변경하지 않습니다. 성능 확인은 동결한 Heldout192를 사용합니다.\n'
        f'Full400 SHA256: `{digest(ROOT/"data/longbench_balanced400_v2.json")}`.\n'
        f'Heldout192 SHA256: `{digest(ROOT/"data/balanced400_v2_heldout.json")}`.\n'
        'Discovery208과 Heldout192를 구분합니다. Natural400 정합성 검사는 성능 측정이 아닙니다.\n'
        'Natural EOS와 요청별 독립 C1 길이를 사용한 Equal-output Replay를 구분합니다.\n'
        'Fixed-output 표의 Quality는 적격 Natural400 구성 검사를 계승합니다. EOS 이후 강제 출력은 새 Natural 정답으로 평가하지 않습니다.\n'
        '독립 단위는 Server Start입니다. Request와 Token을 독립 Start로 사용하지 않습니다.\n\n'
        '동결 SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, TPOT≤33ms/token입니다.\n'
        'NativeStrong과 Common-D-policy Native에 대한 ITL p99·TPOT p95의 상대5% Gate를 별도로 검사합니다.\n'
        '각 상대 비율의 Paired95% CI 상한은1.05 이하여야 합니다. Absolute Decode Attainment는 각 Start에서95% 이상이어야 합니다.\n')


def comparison_rows(families):
    rows=[]
    for family,value in families:
        if not value:continue
        for row in value['comparisons']:
            if row['right'] not in ('StockToken','OriginalLayer','OriginalHybrid','StockCommon16','HybridGCControl'):continue
            rows.append([family,row['mode'],row['left']+'/'+row['right'],row['paired_starts'],
                estimate(row['ratios']['TTFT_p95_ms']),estimate(row['P_critical_path_ratio']),
                estimate(row['ratios']['SLO_qualified_output_tokens_per_second']),row['quality_passed_all_paired_starts']])
    return rows


def write_reports(final=False):
    destination=ROOT.parent if final else ROOT/'report_drafts';destination.mkdir(exist_ok=True)
    if final:
        audit=read('state/science_completion_audit.json')
        assert audit and audit['passed'],'Do not label incomplete evidence FINAL'
    causal=read('analysis/p_critical_causal_summary.json') or dict(observations=[],comparisons=[])
    serial=read('analysis/serial_serving_operator_analysis.json') or dict(results=[])
    families=[('Credit Retirement',read('analysis/c3_credit_confirmation_analysis.json')),
        ('Partition Geometry',read('analysis/p_geometry_confirmation_analysis.json')),
        ('D Cap4',read('analysis/p_geometry_D_cap4_confirmation_analysis.json')),
        ('Common Coordinator',read('analysis/coordinator_hygiene_confirmation_analysis.json')),
        ('Stable-source Stock Coordinator',read('analysis/coordinator_hygiene_stable_stock_confirmation_analysis.json')),
        ('Fixed-port Coordinator',read('analysis/coordinator_hygiene_fixed_ports_confirmation_analysis.json')),
        ('Warm-regime Coordinator',read('analysis/coordinator_hygiene_warm_regime_confirmation_analysis.json')),
        ('Output192 D cap2→4',read('analysis/output192_D_cap_intervention_analysis.json')),
        ('Fixed-output Stress',read('analysis/p_output_stress_analysis.json'))]
    common=body_common(final)
    refs='\n## 근거 파일\n\n'+ '\n\n'.join(link(path,destination) for path in (
        Path('configs/research_contract.json'),Path('analysis/p_critical_causal_summary.json'),
        Path('analysis/c3_credit_confirmation_analysis.json'),Path('analysis/p_geometry_confirmation_analysis.json'),
        Path('analysis/serial_serving_operator_analysis.json'),Path('analysis/native_relative_accuracy_analysis.json'),
        Path('analysis/coordinator_stock_compiler_artifact_audit.json'),Path('analysis/figure_manifest.json'),
        Path('analysis/historical_stage_recalculation.json'),Path('references/notion_archive_audit.json')))+'\n'
    kernel_rows=[]
    for row in causal['observations']:
        if row['family']!='PUBLISHED_TRACE_REPRODUCTION_SAME_PAIR':continue
        kernel_rows.append([row['label'],row['start'],row['evidence'],
            number(row['Arrival_to_P_ready_mean_ms']),number(row['P_kernel_overlapping_path_mean_ms']),
            number(row['no_observed_P_kernel_path_mean_ms']),number(row['P_kernel_resource_demand_ms_per_request']),
            number(100*row['P_kernel_busy_fraction'])])
    p_queue='# P Queue와 GPU·Host 경로의 원인 분석\n\n'+common+'''
## 측정 범위

기존 약1.76초는 과거 N3의 Mean Host 경로 차이입니다.
이 값을 새 Capture의 다른 시각·Pair 결과로 소급 분해하지 않습니다.
새 계측은 같은 Request, 과거 Arrival Seed3개, 같은 Output Work와 GPU0/1을 사용합니다.
Start별 Clean/Profile/Clean을 따로 검사합니다. 모든5개 지연 지표의 차이는5% 이하여야 합니다.

다음 표는 Arrival→P-ready를 P Kernel 실행과 겹친 시간과 나머지 시간으로 나눕니다.
P Kernel 실행에는 앞선 Request와 Shared Batch의 계산도 포함됩니다.
따라서 Kernel과 겹친 대기시간을 해당 Request 자체의 GPU 계산시간으로 표현하지 않습니다.
또한 이 비율은 GPU 계산이 원인인 지연의 인과적 비중이 아닙니다.

''' +table(['Axis','Start','Evidence','Arrival→P-ready ms','P Kernel과 겹침 ms','P Kernel 미관측 ms','Kernel Union/N ms/request','P Kernel Busy %'],kernel_rows)+'''
P Kernel Busy의 분모는 해당 Cohort 관측 구간입니다. SM Utilization이 아닙니다.
P Queue가 길어도 P Kernel이 관측 구간 전체에서 계속 실행된 것은 아닙니다.
이 사실만으로 즉시 실행 가능한 Decode-safe Headroom이 있다고 주장하지 않습니다.

## Host 경계와 Credit

Arrival→P Submit, Submit→Accept, Accept→Prepare, Prepare→Ready를 따로 보존합니다.
Prepare는 Host 경계입니다. 실제 GPU Kernel 시작 시각이 아닙니다.
Continuous Source에는 Pending Credit Wait의 직접 계측이 없습니다.
빈 `P_wait_segments.jsonl`을 대기시간0으로 처리하지 않습니다.
Arrival→Dispatch와 Full Source Window의 중첩은 관측된 공동 상태입니다.
이 중첩은 독립적인 Credit 지연의 인과적 추정값이 아닙니다.

P Kernel이 없는 구간도 전체 GPU Physical Idle을 뜻하지 않습니다.
DMA는 계측하지 않았습니다. Event Clock 정렬은 Endpoint Anchor와 Drift의 불확실성을 포함합니다.
Single-request Forward Enclosure의 Kernel은 해당 호출과 연관된 Activity입니다.
다른 Stream의 동시 Kernel을 Correlation ID로 제외하지는 못했습니다.
Shared Batch의 Kernel 비용을 Request별로 임의 배분하지 않습니다.
''' +refs
    operator_rows=[]
    for row in causal['observations']:
        if row['family']!='PUBLISHED_TRACE_REPRODUCTION_SAME_PAIR':continue
        costs=row['category_resource_demand_ms_per_request']
        operator_rows.append([row['label'],row['start'],row['evidence'],*[number(costs.get(k)) for k in
            ('GEMM','ATTENTION','RMSNORM','FUSED_RMSNORM_RESIDUAL','ROPE','UNKNOWN')]])
    serial_rows=[]
    for row in serial['results']:
        for phase in row.get('phase_means',[]):
            serial_rows.append([row.get('family','SERIAL_CONTROL'),row['label'],row['start'],row['evidence'],phase['phase'],
                number(phase['mean_host_ms']),number(phase['mean_GPU_kernel_union_ms']),number(phase['mean_no_observed_kernel_ms'])])
    operators='# P GPU Operator와 Serial P/D Service 분석\n\n'+common+'''
## 실제 C3 P Kernel

아래 값의 분모는 각 Capture의 Request 수입니다.
Kernel Union을 Request 수로 나눈 평균 Resource Demand입니다.
Request의 경과시간이나 Host Queue의 가법 분해가 아닙니다.
GEMM 이름만으로 Q/K/V·O·MLP Projection을 구분하지 못했습니다.
Fused Norm/Residual을 두 비용으로 나누지 않았습니다. Unknown은 그대로 남깁니다.
SM Utilization과 HBM 사용률은 NOT_MEASURED입니다.

''' +table(['Axis','Start','Evidence','GEMM ms/req','Attention ms/req','RMSNorm ms/req','Fused Norm/Residual ms/req','RoPE ms/req','Unknown ms/req'],operator_rows)+'''
## Serial Serving Control

실제 Serving API를 사용합니다. 동시에 살아 있는 Request는1개입니다.
P Prefill, D Suffix, D Decode의 실제 Batch1을 검사합니다.
Prompt는 동결한32개입니다. Output은 정확히16개입니다.
이는 Natural EOS나 Poisson Online Capacity 결과가 아닙니다.
Start별 CPC Gate를 실패한 Capture는 DIAGNOSTIC입니다.

''' +table(['Family','Axis','Start','Evidence','Phase','Host ms','Kernel Union ms','Kernel 미관측 ms'],serial_rows)+'''
Natural C3의 D Suffix Operator별 기여율은 Serial 결과로 대체하지 않습니다.
직접 계측하지 않은 Natural C3 D Operator 비중은 NOT_MEASURED입니다.

## P RMSNorm Primitive Probe

기존 P Prefix Segment는 CUDA 우선순위를 별도 Scope로 강제합니다.
Runtime-global Native IR 설정만으로 이 임시 Scope를 확인할 수는 없습니다.
새 Probe는 P Scope의 RMSNorm Priority만 바꿉니다. D Code와 작업량은 유지합니다.
Provider별 Cache Directory를 분리합니다. Control Artifact를 Candidate에 재사용하지 않습니다.
Strict8 실패는 보존합니다. 해당 Same-family Replacement를 Primary 성능으로 사용하지 않습니다.
이 검사는 Native 구현의 수학적 타당성을 부정하는 검사가 아닙니다.
''' +refs
    comparisons=comparison_rows(families)
    diagnostic_rows=[]
    for path in ('analysis/coordinator_delivery_diagnostic.json','analysis/coordinator_GC_diagnostic.json'):
        value=read(path) or dict(cases=[])
        for row in value['cases']:
            diagnostic_rows.append([row['name'],row['deferred'],row.get('cyclic_GC_deferred',False),
                number(row['client_ITL_ms'].get('maximum')),number(row['worker_emission_gap_ms'].get('maximum')),
                number(row['GC_ms'].get('maximum'))])
    admission='# P Admission·Source Credit의 Controlled Intervention\n\n'+common+'''
## Safe Retirement

원래 Credit은 첫 Client Token에서 반환합니다.
새 경로는 Native Request 제거, 필요한 GPU Fence, Hidden READ ACK와 Owner 해제를 확인합니다.
조건이 충족되면 Logical Source Credit을 먼저 반환합니다.
첫 Client Token의 기존 반환 경로도 유지합니다. 반환은 Idempotent입니다.
KV Block을 수동 해제하지 않습니다. Native Allocator가 물리적 재사용을 관리합니다.
Credit 반환 전에 모든 물리적 KV Block이 Free Pool에 들어갔다고 주장하지 않습니다.

같은 Capsule에서 False/True Switch만 바꿉니다.
Source Credit16/8과 D Cap2는 유지합니다. Actual Batch는 별도로 기록합니다.
Queue가 Coordinator에서 P Worker로 이동한 것만으로 성공을 판정하지 않습니다.
Primary P 지표는 Arrival→P-ready입니다. 사전 목표는5% 감소입니다.
Normal Completion을 검증합니다. Cancellation과 Crash Recovery는 NOT_RUN입니다.

''' +table(['Family','Workload','비교','Paired Starts','TTFT p95 Ratio [95% CI]','P Path Ratio [95% CI]','Goodput Ratio [95% CI]','Quality'],comparisons)+'''
## Coordinator 전달 지연

Client ITL은 Worker Emission Gap과 Delivery Delay 변화로 나눕니다.
기존1.85초 구간의 정확한 Host 원인은 계측하지 못했습니다.
별도 Diagnostic에서 파일 쓰기와 Python GC가 Token 전달을 지연한 구간을 관측했습니다.
Host Enclosure가 겹치므로 각 시간을 더해서 독점 원인 비중으로 계산하지 않습니다.
파일 쓰기 지연과 GC 지연은 별도의 One-factor 진단으로 검사합니다.
Worker Emission Gap이 긴 경우를 Coordinator 문제로 바꾸어 설명하지 않습니다.

''' +table(['Diagnostic Run','Write Deferred','Cyclic GC Deferred','Client Max ITL ms','Worker Max Gap ms','GC Max ms'],diagnostic_rows)+'''
공통 개선은 모든 비교군에 동일하게 적용합니다. Client Timestamp는 보정하지 않습니다.
JSON 직렬화는 원래 호출 시점에 유지합니다. 파일 쓰기만 측정 이후로 옮깁니다.
Coordinator의 Cyclic GC 상태는 측정 후와 예외 종료 시 복원합니다.
Reference Counting과 Worker GC는 변경하지 않습니다.
Hard Kill이 `finally`보다 먼저 발생하면 Buffered Evidence를 잃을 수 있습니다.
이 개선은 계측 경로의 개선입니다. Layer-specific GPU 효율로 주장하지 않습니다.
''' +refs
    hybrid='# Hybrid Pipeline과 P/D 작업 재배치 PoC\n\n'+common+'''
## 기존 DAG

기존 Long-input 경로는 Prefix Upper Layers 후 Suffix Full36을 실행합니다.
이미 적격인 Short-input Fusion은 Suffix Lower Layers 후 전체 Upper Layers를 실행합니다.
이 Fusion은 L≤8192에만 적용합니다. 기존 개선을 새 기여로 계산하지 않습니다.
긴 입력에 Fusion을 강제로 적용해서 B8192를 위반하지 않습니다.
새 Active-decode Insertion은 실행하지 않습니다.

## STRUCTURAL_ACCOUNTING

전체 Layer 수는36입니다. 입력 길이는 L입니다. Upper Layer 수는 m입니다. Token Suffix는 s입니다.
P GEMM Rows는 `(L−s)(36−m)`입니다. D GEMM Rows는 `36L−P Rows`입니다.
P Attention Pair는 `(36−m)(L−s)(L−s+1)/2`입니다.
D Attention Pair는 전체 `36L(L+1)/2`에서 P Pair를 뺀 값입니다.
이는 Work Accounting입니다. GPU 시간이나 측정된 Speedup이 아닙니다.

같은 P GEMM Rows에서 Token-only는 P Attention Pair가 더 작을 수 있습니다.
반대로 Layer는 D Attention과 작은-query Launch 비용을 줄일 수 있습니다.
어느 효과가 TTFT를 지배하는지는 P/D Queue와 Decode Residency에도 의존합니다.

## 새 Geometry Control

첫 후보는 Layer Last2→Last8입니다. Source Credit16과 D Cap2는 유지합니다.
둘째 후보는 Hybrid Last2의 Suffix1/8→3/17입니다. Source Credit8과 D Cap2는 유지합니다.
두 후보는 P 작업을 D로 더 옮깁니다. 동일 작업량의 Kernel 최적화로 표현하지 않습니다.
Hybrid3/17과 Layer8은 명목상 P28L/D8L Rows입니다.
실제16-token Alignment와 추가 실행 Rows는 Raw Work Audit을 사용합니다.

Discovery208의 사전 Screen을 적용한 뒤 Candidate를 동결했습니다.
P 경로5% 감소 또는 Stock 대비 TTFT Point Ratio≤1.05는 후속 확인의 Screen입니다.
이 Screen은 성공 기준이 아닙니다. 최종 우위는 독립 N3 CI와 Decode Gate로 판정합니다.
''' +refs+'\n'+link(Path('analysis/p_service_balance_discovery.json'),destination)+'\n'
    safety_rows=[]
    for family,value in families:
        if not value:continue
        for row in value['cells']:
            gate=row['decode_gate'];f=row['fields']
            safety_rows.append([family,row['mode'],row['label'],row['independent_starts'],
                estimate(f['ITL_p99_ms']),estimate(f['TPOT_p95_ms']),
                gate.get('relative_5percent_pass'),gate.get('absolute_decode_attainment_95percent_pass'),
                row['quality_passed_all_starts'],gate.get('per_start_max_ITL_ms')])
    confirmation='# Optimized C3의 독립 확인과 Decode 보호\n\n'+common+'''
## 비교 계약

Strong Stock Token은 원래 Compiled Full-model 경로를 유지합니다.
Credit0은 무제한 Coordinator Window입니다. GPU Memory가 무제한이라는 뜻이 아닙니다.
Common16 Control은 Logical Source Window16과 D Cap2를 사용합니다.
Strong Stock을 이 Control로 대체하지 않습니다.
같은 정책이 같은 Actual GPU Batch나 Hidden Buffer 소유권을 뜻하지 않습니다.

Geometry, Credit Retirement와 Common Coordinator 실험은 서로 다른 Family입니다.
서로 다른 Family의 Start를 합쳐 N을 늘리지 않습니다.
Natural EOS의 실제 Output Work가 다른 경우를 기록합니다.
Equal-output Replay에서는 Request별 Output Token 수를 정확히 맞춥니다.
Natural400 품질 통과만으로 Heldout 성능 통과를 주장하지 않습니다.

''' +table(['Family','Workload','비교','Paired Starts','TTFT p95 Ratio [95% CI]','P Path Ratio [95% CI]','Goodput Ratio [95% CI]','Quality'],comparisons)+'''
## Decode Gate

''' +table(['Family','Workload','Arm','Starts','ITL p99 ms [CI]','TPOT p95 ms [CI]','Relative5%','Absolute95%','Quality','Start별 Max ITL ms'],safety_rows)+'''
Output16/64/128/192 Stress는 원문을 바꾸지 않습니다.
실제 Natural Output을 미리 안다고 가정하지 않습니다. Fixed-output 결과는 별도 Control입니다.
CI가 없는 Point를 반복 재현된 우위로 표현하지 않습니다.
실측하지 않은 QPS와 Output Cell은 보간하지 않습니다.
''' +refs
    accuracy=read('analysis/native_relative_accuracy_analysis.json') or dict(starts=[])
    accuracy_rows=[[row['family'],row['start'],row['label'],row['requests'],row['exact_output_requests'],
        row['lower_score_requests'],row['higher_score_requests'],number(row['task_macro_delta_pp'],4),
        str(row['context_cluster_CI95_pp']),row['frozen_C1_quality_passed']] for row in accuracy['starts']]
    quality_text='''
## 출력 동일성과 Task 품질

Token과 Layer 모두 Native와 다른 답변을 생성할 수 있습니다.
분할 자체가 필요한 계산을 생략하는 근사 기법이라는 뜻은 아닙니다.
BI0의 Batch·Shape·Compiler 변화가 수치 차이를 만들 수 있습니다.
그러나 확인되지 않은 구현 오류를 수치 차이로 단정하지 않습니다.

아래 표의 NativeStrong은 정답 Teacher가 아니라 같은 Arrival Start의 비교군입니다.
Output Token 전체 일치와 Task 점수를 별도로 검사합니다.
Task 점수 하락은 개별 요청에서 측정된 품질 저하입니다.
전체 평균 저하는 별도의 CI로 판단합니다. 낮아진 요청 수만으로 평균 저하를 판정하지 않습니다.
이 표의 CI는 Context-cluster Bootstrap입니다. 독립 Server Start CI가 아닙니다.
기존 Compiled C1 기준과2pp 비열등성 Gate는 변경하지 않습니다.
품질 Gate 통과는 모든 답변이 정답이거나 모든 토큰이 동일하다는 뜻이 아닙니다.

'''+table(['Family','Start','Arm','Requests','Native와 토큰 전체 일치','Task 점수 하락','Task 점수 상승',
        'Task-macro Δpp','Context95% CI pp','Frozen C1 Gate'],accuracy_rows)+'''
같은 Numerical Geometry의 Strict KV·Hidden·Greedy 검사가 실패한 새 구성은 Primary에서 제외합니다.
분산 Teacher를 단일 GPU Schema로 읽은 Missing-file 오류는 Reader 수정으로 분리했습니다.
Reader 수정 후 실제 Strict 불일치가 남으면 실패를 그대로 유지합니다.
Artifact의 Reduction 설정 차이는 관측 근거입니다. 이것만으로 원인을 확정하지 않습니다.
공유 Source 경로의 통제 실험은 기존 실패를 삭제하거나 정확도 기준을 완화하지 않습니다.
'''
    confirmation+=quality_text
    verdict=read('analysis/scientific_verdict.json') or dict(outcomes=[])
    repeated=[row for row in verdict['outcomes'] if row['paired_starts']>=3]
    wins=sum(row['serving_GO'] for row in repeated)
    decision=('GO_IN_SPECIFIC_TESTED_CELLS' if wins else 'NO_GO_ON_TESTED_GRID') if final else 'DRAFT_PENDING_FINAL_AUDIT'
    headline=(f'과학적 성능 판정: `{decision}`. 독립 N3 이상 비교의 Serving GO는 {wins}/{len(repeated)}개입니다.\n\n'
        'P Critical Path의 원인은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`입니다.\n'
        'GPU 계산 증가와 Queue 누적을 관측했습니다. 과거1.76초의 독점적 인과 비중은 완전히 식별하지 못했습니다.\n'
        'Credit·Geometry·D Admission·Coordinator 개입을 각각 검증했습니다. 개입의 성공 여부는 아래 표에서 구분합니다.\n'
        '미래 Decode-safe Window와 Online Dynamic Insertion은 `NOT_RUN`입니다.\n\n')
    final_text='# P-side Critical Path Root Cause & Optimization — 종합\n\n'+headline+common+'''
## 질문별 판정 기준

1. 과거1.76초의 가법 Host 차이와 새 GPU Capture를 구분합니다.
   새 계측은 Kernel Resource Demand와 요청 경로의 Kernel 중첩을 따로 제공합니다.
   GPU 계산·Credit·Queue의 독점적인 인과 비중을 완전히 분리했다고 주장하지 않습니다.
2. 제거 가능한 Logical Credit 지연과 Coordinator 전달 지연을 각각 검사합니다.
   Attention Pair의 차이는 Scheduling만으로 제거하지 못합니다.
3. P 지연 감소는 Arrival→P-ready로 판정합니다.
   앞단 Queue의 위치만 바뀐 결과를 개선으로 계산하지 않습니다.
4. D의 이점은 같은 Geometry A/B와 별도 Serial Iso-work에서 검사합니다.
   Last2와 Stock WorkCut8의 D 시간 차이를 동일 GEMM 효율로 표현하지 않습니다.
5. C3 성공에는 Strong Stock 대비 TTFT CI 상한<1과 Decode 보호가 필요합니다.
   SLO-goodput의 분모와 CI도 보존합니다.
6. 긴 Output의 Cut은 동일 Input·Arrival을 사용한 Fixed-output Stress로 검사합니다.
   단일 Start로 보편적인 안전 Cut을 확정하지 않습니다.
7. Hybrid는 Attention·KV 의존성, Source Buffer와 D Residency의 제약을 받습니다.
   작은 D 비용이 작은 P Queue나 높은 SLO-goodput을 보장하지 않습니다.

## 현재 확인 표

''' +table(['Family','Workload','비교','Paired Starts','TTFT p95 Ratio [95% CI]','P Path Ratio [95% CI]','Goodput Ratio [95% CI]','Quality'],comparisons)+'''
## 재현성과 제한

원래 Checkout과 Staged Index를 보존합니다. 실패한 실험은 삭제하지 않습니다.
기존 Guard의 조기 Drain은 별도로 기록했습니다. 실행 중 GPU 작업은 재시작하지 않았습니다.
GPU 시작 전 차단된 동일 Config만 별도 Recovery Plan에서 재개했습니다.
Native 검사에서 누락된 State Capture Flag는 새 검사 Version으로 수정했습니다.
기존 Missing-file 실패를 Native 계산 오류나 성공으로 바꾸지 않습니다.

실제 Decode-safe Window와 미래 Deadline Predictor는 아직 별도 Gate가 필요합니다.
Kernel Inactivity만으로 안전한 Offloading Window를 주장하지 않습니다.
검증되지 않은 Dynamic Insertion은 NOT_RUN입니다.
Science 완료, Raw Metric 재계산, Figure 검사, Resource Closeout과 Remote SHA 검증은 별도로 확인합니다.
''' +refs+quality_text
    from .report_sections import sections
    extra=sections(table,estimate,number)
    p_queue+=extra['arrivals']
    confirmation+=extra['runtime']
    final_text+=extra['answers']+extra['arrivals']
    figure_manifest=read('analysis/figure_manifest.json') or dict(figures=[])
    main_stems={'p_queue_kernel_overlap','p_request_critical_path_gantt','p_gpu_operator_resource_demand',
        'p_input_output_bottleneck_map','fixed_output_residency_host_paths','credit_retirement_equalreplay_pd_host_stages','credit_retirement_p_intervention',
        'credit_retirement_equalreplay_ttft_goodput','credit_retirement_equalreplay_stock_ratios',
        'credit_retirement_equalreplay_decode_protection','coordinator_hygiene_warm_regime_equalreplay_ttft_goodput',
        'coordinator_hygiene_warm_regime_equalreplay_decode_protection','output192_d_cap_intervention_fixed192_ttft_goodput'}
    final_text+='\n## Figure와 대응 수치\n\n'+table(['Figure','PNG','PDF','CSV'],[
        [row['stem'],*[link(Path(row['files'][kind]['path']),destination) for kind in ('png','pdf','csv')]]
        for row in figure_manifest['figures'] if row['stem'] in main_stems])
    texts=(p_queue,operators,admission,hybrid,confirmation,final_text)
    for name,text in zip(NAMES,texts):(destination/name).write_text(text)
    write(ROOT/'analysis/report_draft_manifest.json',dict(time=datetime.now(timezone.utc).isoformat(),final=final,
        files={name:dict(path=str((destination/name).relative_to(ROOT.parent)),SHA256=digest(destination/name)) for name in NAMES}))
    print('P_CRITICAL_REPORTS',str(destination),len(texts),'FINAL' if final else 'DRAFT',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();write_reports(args.final)
