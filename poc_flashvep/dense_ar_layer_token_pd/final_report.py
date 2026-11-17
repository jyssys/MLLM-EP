"""Render the final Korean summary from independent-start evidence."""
import collections
import json
import statistics
from datetime import datetime,timezone
from .common import ROOT,REPO,BASE,BRANCH


def load(name,default=None):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else default


def number(value,digits=3):
    return 'NOT_MEASURED' if value is None else f'{value:.{digits}f}'


def ratio(value):
    if not value or value.get('ratio') is None:
        return 'UNDEFINED_ZERO_REFERENCE_MEAN' if value and value.get('reference_mean')==0 else 'NOT_MEASURED'
    ci=value.get('ratio_CI95')
    return number(value['ratio'],6)+(f' [{ci[0]:.6f}, {ci[1]:.6f}]' if ci else ' [n<2, CI 없음]')


def main():
    data=load('analysis/heldout_start_statistics.json')
    assert data and load('state/CAMPAIGN_CLOSED.json'),'Close admissions before the final summary'
    blocks=[]
    def add(text=''):
        blocks.append(text.strip())
    add('''# Dense AR TP2·Layer/Token·Dynamic 심층 PoC 최종 요약

## 최종 판정

TP2의 최초 수치 차이는 식별했다.
그러나 TP1 Strict Parity를 회복하지 못했다.
따라서 TP2는 Primary 성능 기준선에서 제외했다.

실제 2GPU Token-axis Continuation은 구현과 State 정합성을 통과했다.
같은 이전 작업량의 Last12 Control에서는 Layer의 조건부 이점을 확인했다.
이 결과를 모든 Token Split이나 모든 Workload에 대한 우위로 해석하지 않는다.
새 문서의 Best-vs-Best 결과와 Dynamic의 독립 기여는 아래 Table에서 별도로 판단한다.

이 캠페인은 Sustainable Capacity 또는 논문 Novelty를 입증하지 않았다.
N128/N144 완료율은 Finite-cohort Drain Rate다.
물리적인 Kernel Busy/Idle, SM/HBM Counter는 `NOT_MEASURED`다.''')
    add('''최종 판정은 부분 성공이다.

| 연구 목표 | 판정 | 범위 |
|---|---|---|
| TP2 수치 원인 | NUMERICAL_NON_EQUIVALENCE_IDENTIFIED | 수집한 최초 경계와 같은 Operand Replay |
| TP1 Strict Parity의 TP2 Baseline | BLOCKED_CORRECTNESS | 전체 Within-TP2 State도 별도 미측정 |
| 실제 Token-axis Continuation | CORRECTNESS_ONLY: PASS | N8·N32와 선정 후보 N128 Prompt State |
| Layer-axis Serving 이점 | MEASURED: 조건부 확인 | Output16의 동결 후보와 같은 작업량 Control |
| Strong Static을 넘는 Dynamic Primary 기여 | NO_GO_ON_TESTED_GRID | SLO Rate CI와 Native-relative 보호의 동시 우위 미확보 |
| Sustainable Capacity·전역 Novelty | NOT_MEASURED / NOT_IDENTIFIED | 이번 결과의 주장 범위 밖 |

NO-GO는 이번 모델·Runtime·탐색 Grid의 증거 부족을 뜻한다.
모든 Dynamic 정책의 효과가0이라는 증명은 아니다.''')
    closed=load('state/science_closeout.json')
    if closed:
        add('탐색 시작은 '+datetime.fromtimestamp(closed['start_epoch'],timezone.utc).isoformat(timespec='seconds')+'다.\n'+
            'GPU 실험 Admission 종료는 '+datetime.fromtimestamp(closed['time'],timezone.utc).isoformat(timespec='seconds')+'다.\n'+
            f"이 구간의 Wall-clock 연구 시간은{closed['science_wall_seconds']/3600:.3f}시간이다.\n"+
            'GPU 시작·정합성 Capture·CPU 분석·Lease 대기를 포함한다.\nExclusive GPU Compute Time이 아니다.\n이후 Archive·보고·Push 시간은 별도다.')
    primary=data['comparisons']
    gains=[]
    for row in primary:
        if (row['reference'],row['candidate'])!=('Token','Static') or len(row['paired_seeds'])<3:
            continue
        ci=row['metrics']['TTFT_p95_ms']['ratio_CI95']
        native=next((r for r in primary if r['cap']==row['cap'] and r['QPS']==row['QPS'] and
            (r['reference'],r['candidate'])==('Native','Static')),None)
        if ci and ci[1]<1 and native and native['Decode_5pct_noninferiority']:
            gains.append(row)
    if gains:
        lines=['새 문서의 독립3회에서도 다음 조건부 TTFT 우위를 확인했다.','',
            '| 조건 | Static Layer/Token TTFT p95, 95% CI | TTFT 감소 | Native 대비 Decode5% |',
            '|---|---|---:|---|']
        for row in gains:
            lines.append(f"| Output{row['cap']}, QPS{row['QPS']:g} | "+ratio(row['metrics']['TTFT_p95_ms'])+
                f" | {100*(1-row['metrics']['TTFT_p95_ms']['ratio']):.3f}% | PASS |")
        add('\n'.join(lines))
        add('이 우위는 동결한 Completed-prefix Token 후보와 Static 후보의 비교다.\nStreaming Token Pipeline이나 전역 최적 Token Scheduler를 이긴 결과가 아니다.')
    losses=[r for r in primary if (r['reference'],r['candidate'])==('Token','Static') and
        len(r['paired_seeds'])>=3 and r['metrics']['TTFT_p95_ms']['ratio_CI95'] and
        r['metrics']['TTFT_p95_ms']['ratio_CI95'][0]>1]
    if losses:
        lines=['반대로 Token이 더 빠른 조건도 확인했다.','',
            '| 조건 | Static Layer/Token TTFT p95, 95% CI |','|---|---|']
        for row in losses:
            lines.append(f"| Output{row['cap']}, QPS{row['QPS']:g} | "+ratio(row['metrics']['TTFT_p95_ms'])+' |')
        add('\n'.join(lines))
    dynamic_wins=[]
    for row in primary:
        if (row['reference'],row['candidate'])!=('Static','Dynamic') or len(row['paired_seeds'])<3:
            continue
        ci=row['metrics']['Output_token_SLO_goodput']['ratio_CI95']
        native=next((r for r in primary if r['cap']==row['cap'] and r['QPS']==row['QPS'] and
            (r['reference'],r['candidate'])==('Native','Dynamic')),None)
        if ci and ci[0]>1 and native and native['Decode_5pct_noninferiority']:
            dynamic_wins.append(row)
    add('Dynamic의 Strong Static 대비 SLO Rate 우위와 Native-relative Decode 보호를 함께 통과한 Primary 조건 수는 '+str(len(dynamic_wins))+'개다.\n이 수치만으로 Matched Geometry의 독립 정책 기여까지 통과했다고 판정하지 않는다.')
    add(f'''## 환경·보존 계약

- Model: `Qwen/Qwen3-8B`.
- Revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Original BF16, Exact Causal Attention, Full36 Decode, Greedy와 Thinking Disabled.
- H100 SXM 80GB GPU0/1의 같은 2GPU.
- GPU0 UUID: `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`.
- GPU1 UUID: `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`.
- vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1.
- GPU당 KV42GiB. Prefix Cache 비활성화.
- Publication Parent: `{BASE}`.
- 새 Branch: `{BRANCH}`.

기존 User Checkout과 Staged Index를 사용해 Commit하지 않았다.
별도 Index와 Commit Tree로 새 Branch를 공개한다.
이전 Source 435개와 외부 Runtime의 보존 검사를 수행한다.
추가 Worker 의존 Source140개도 Publication Parent와 Byte Equality를 확인했다.
원래 Weight Shard 5개의 SHA256은 Content-addressed 원본과 일치했다.
실험별 Config, Command, Model Revision, GPU Map, Source ZIP과 Hash를 보존한다.
실패한 실행도 삭제하지 않는다.''')
    add('P와 D는 모두 전체 Model Weight를 보유한다.\nLayer Weight를 영구 분할하는 PP2가 아니다.\nD는 모든 방법에서 정상적인 Full36 Decode를 수행한다.')
    add('''## 데이터·SLO·통계

Primary Input은 8000 Tokens다.
Output16/64는 `ignore_eos=True`로 고정했다.
새 LongBench 원문128개는 기존 Discovery 원문 ID와 첫7900-token Hash가 겹치지 않는다.
Discovery의 최적 설정을 고른 뒤 Candidate 평가 전에 동결했다.
같은 Output/QPS/Seed에서는 Prompt와 Arrival Trace를 모두 맞췄다.
Seeds는 2026111740/2026111741/2026111742다.
한 번만 실행한 탐색 QPS에는 독립 반복 CI를 만들지 않는다.

Mean TBT는 Coordinator가 전달한 연속 Token 간격의 평균이다.
TPOT는 Request별 `(last−first)/(Output−1)`이다.
First ITL을 포함한다.
요청의 TTFT·TPOT·모든 ITL이 통과해야 SLO Token에 포함한다.
Worker 간격이 정상이어도 Client SLO 실패를 통과로 바꾸지 않는다.

| Output | Regime | TTFT, ms | ITL, ms | TPOT, ms |
|---|---|---:|---:|---:|
| 16 | TTFT-priority | 2000 | 54 | 52 |
| 16 | Balanced | 5000 | 34 | 33 |
| 16 | Decode-priority | 10000 | 30 | 29 |
| 64 | TTFT-priority | 2000 | 55 | 53 |
| 64 | Balanced | 5000 | 35 | 33 |
| 64 | Decode-priority | 10000 | 31 | 29 |

이 Threshold는 기존 Frozen SLO다.
이번 결과를 보고 변경하지 않았다.
Primary 선정은 Decode-priority의 SLO-qualified Output Token Rate를 사용했다.
다른 두 Regime도 원자료와 통계 파일에 함께 보존했다.

아래 값은 Start별 Metric의 평균이다.
Tail을 모든 Token에 Pool한 값이 아니다.
95% CI는 Arrival Seed를 짝지은 독립 Server Start Bootstrap이다.
n=3은 작은 표본이다.
CI는 Pointwise이며 다중 비교 보정은 적용하지 않았다.
Native-relative ITL p99와 TPOT p95의 CI 상한이 모두1.05 이하여야 보호를 통과한다.
Absolute SLO와 Relative Gate는 별도 판정이다.''')
    add('## 새 문서 Best-vs-Best — MEASURED')
    for cap,qps in ((16,6.4),(64,4.8),(64,6.4)):
        rows=[r for r in data['aggregate'] if r['cap']==cap and r['QPS']==qps]
        if not rows:
            continue
        add(f'### Input8000/Output{cap}, Offered QPS{qps:g}')
        lines=['| 방법 | 독립 Start | Mean TTFT, ms | TTFT p95, ms | Mean TBT, ms | ITL p99, ms | TPOT p95, ms | Raw Output tokens/s | SLO Output tokens/s | Joint, % | 최대 ITL, ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method in ('Native','Colocated','Token','Static','Dynamic'):
            row=next((r for r in rows if r['method']==method),None)
            if row is None:
                lines.append(f'| {method} | NOT_RUN | — | — | — | — | — | — | — | — | — |');continue
            values=row['metrics']
            keys=('mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms',
                  'output_tokens_s','Output_token_SLO_goodput')
            lines.append('| '+method+' | '+str(row['starts'])+' | '+
                ' | '.join(number(values[key]['mean']) for key in keys)+
                ' | '+number(100*values['Joint_attainment']['mean'],4)+
                ' | '+number(max(r['maximum_ITL_ms'] for r in data['rows'] if
                    r['cap']==cap and r['QPS']==qps and r['method']==method))+' |')
        add('\n'.join(lines))
        paired=[r for r in data['comparisons'] if r['cap']==cap and r['QPS']==qps and
                (r['reference'],r['candidate']) in (('Native','Token'),('Native','Static'),
                ('Native','Dynamic'),('Token','Static'),('Static','Dynamic'))]
        lines=['비율의 방향은 Candidate/Reference다. 대괄호는 95% CI다.','',
            '| Reference → Candidate | n | TTFT p95 비율 | ITL p99 비율 | TPOT p95 비율 | SLO Output 비율 | Native 대비 Decode5% Gate |',
            '|---|---:|---|---|---|---|---|']
        for row in paired:
            gate=('PASS' if row['Decode_5pct_noninferiority'] else 'NOT_ESTABLISHED') if row['reference']=='Native' else '해당 없음'
            lines.append('| '+row['reference']+' → '+row['candidate']+' | '+str(len(row['paired_seeds']))+
                ' | '+' | '.join(ratio(row['metrics'][key]) for key in
                    ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','Output_token_SLO_goodput'))+' | '+gate+' |')
        add('\n'.join(lines))
        actual=[r['scheduled_trace_QPS'] for r in data['rows'] if r['cap']==cap and r['QPS']==qps and r['method']=='Native']
        if actual:
            add('Native와 같은 Arrival Trace의 실제 유한 표본 QPS는 '+
                '/'.join(number(x,6) for x in actual)+'다.\nOffered Rate와 유한 표본 Rate를 구분한다.')
    add('### 선정된 실행 설정')
    for cap in (16,64):
        freeze=load(f'state/candidate_confirmation_freeze_O{cap}.json')
        if not freeze:
            add(f'Output{cap} 후보 동결은 `NOT_RUN`이다.');continue
        lines=[f'Output{cap}의 설정은 다음과 같다.','',
            '| 방법 | Discovery에서 동결한 설정 |', '|---|---|']
        for method in ('Token','Static','Dynamic'):
            cfg=freeze['contract']['selected'][method]['representative']
            detail=[cfg['name'],f"P slots{cfg.get('P_slots')}",f"D admission{cfg.get('D_admission_batch')}"]
            if method=='Token':
                detail.append('Suffix'+str(cfg['token_suffix']))
            elif method=='Static':
                detail.extend(('Last'+str(cfg['static_layers']),'Fraction'+str(cfg.get('migration_fraction',1))))
            else:
                detail.append('Horizon'+str(cfg.get('flow_horizon_groups')))
            lines.append('| '+method+' | '+' / '.join(detail)+' |')
        add('\n'.join(lines))
    add('''Source Group은 Coordinator Macro다.
P slots4/Token Budget32000에서 Source Macro8을 한 GPU Batch8로 표시하지 않는다.
실제 P/D Shape는 각 Run의 `measured_geometry_*.json`에 기록했다.
Colocated는 사용자 지정 B8192/S1024의 Standard Continuous DP2다.
이를 모든 SLO에 대한 전역 최적 Colocated라고 주장하지 않는다.
이번 캠페인은 중단한 Aggregated B512/S16 탐색을 재개하지 않았다.
Token Arm의 D-side B512 검사는 다른 실험이다.''')
    add('''## 같은 작업량 Control — 독립3회

N144, Output16, QPS6.4에서 같은64개 Discovery 원문을 선언된 주기로 반복했다.
새 원문 Holdout은 아니다.
Seeds는 2026111760/1761/1762다.
LayerLast12와 Token2656/2672는 실제 P/D GEMM Rows와 유효 KV Bytes를 맞췄다.
실제 Source/Destination Group도 일치했다.
Layer의 Hidden/Residual과 Source Credit 점유는 같지 않다.

| Metric | Layer/Token | 독립 Start 기반95% CI |
|---|---:|---|
| TTFT p95 | 0.860530 | [0.791981, 0.927848] |
| ITL p99 | 1.009008 | [0.999586, 1.019340] |
| TPOT p95 | 1.001546 | [1.000400, 1.002979] |
| Raw Output tokens/s | 1.044749 | [1.023927, 1.064201] |
| SLO Output tokens/s | 1.080545 | [1.026571, 1.144348] |

TTFT p95는13.947% 감소했다.
SLO Output은8.054% 증가했다.
Raw Output은4.475% 증가했다.
Raw 개선이5% 이상이라고 입증한 것은 아니다.
Layer의 Native-relative Decode5% Gate도 통과했다.
작은 Start 수와 반복 Prompt Pool의 한계를 유지한다.
Last8의 초기 n=1 Control에서는 반대로 Token이 더 빨랐다.
따라서 축의 보편적 우위가 아니라 조건부 이점이다.''')
    add('''## Layer/Token의 원인·정합성

Token Arm은 P의 완성된 Prefix KV를 D로 전송한다.
D는 원래 Position의 나머지 Token을 Full36으로 계산한다.
이전 Prefix를 다시 계산하거나 두 Chunk의 독립성을 가정하지 않는다.
Layer별 KV Streaming을 사용한 Token Pipeline은 `NOT_RUN`이다.
P-side Chunk512 PD는 N32 State Gate와 N64 성능 Pilot을 수행했다.
다른 P-side Chunk의 광범위한 튜닝과 새 N128 독립3회는 `NOT_RUN`이다.
Suffix Layerℓ는 Prefix KVℓ와 Suffix Hiddenℓ−1에 의존한다.
따라서 Prefix의 다음 Layer와 겹치는 별도 구현은 가능할 수 있다.
현재 NIXL의 Layer-load Wait는 No-op이며 이 Pipeline을 제공하지 않는다.
이번 우위는 검증한 Completed-prefix Token 구현과 탐색 Grid에 한정한다.
Layer Arm은 Partial KV와 Hidden/Residual을 전달한다.
D는 잔여 Layer 뒤에 정상 Full36 Decode를 수행한다.

Suffix512/1024/2048/4096과 인접 경계 Plan은 N8→N32 State Gate를 통과했다.
새 Prompt128의 주요 Token Suffix와 모든 Layer Cut도 전체 Prompt State를 통과했다.
최종 후보의 추가 N128 Gate는 별도 파일에 기록한다.
Source Macro8/P slots4와 실제 D Prefill Batch8의 N32 State 검사도 별도로 수행했다.
Clean 성능 실행마다 Output Gate를 적용했다.
성능 Run마다 전체 Hidden을 복사했다는 뜻은 아니다.
Held-out PD의 KV·ACK·사용한 Reservation/Hidden Pool 복원도 검사했다.
정상 종료 검사이며 Cancellation이나 GPU Failure 복구의 보장은 아니다.

같은 GEMM Rows의 Last12 Control에서 Token의 D Attention Pair는 Layer의1.666580677배다.
이는 늦은 Query가 더 긴 Causal Prefix를 참조하기 때문이다.
전체 모델 계산을 추가하거나 생략한 것이 아니다.
이는 `STRUCTURAL_ACCOUNTING`이며 GPU 시간 비율이 아니다.
Token 경로는 P Prefix의 불필요한 Sample을 버리지만 현재 Engine은 이를 계산한다.
Layer 경로도 사용하지 않는 P Sample 경로를 유지한다.
이를 Token만의 고유한 비용으로 집계하지 않는다.
해당 공통 LM-head/Sampling을 제거한 최적 경로는 별도로 검증하지 않았다.
같은 Token-layer Rows는 모든 Operator의 동일 작업량을 뜻하지 않는다.
Token Arm의 Last0 Histogram은 Layer Cut을 하지 않는다는 뜻이다.
실제 Token Suffix 이전은 별도 D Rows와 Migrated Request Count로 확인한다.

고립된 Projection의 GPU1 합산 추정은 Layer206.02829ms, Token212.32665ms였다.
Token/Layer는1.030570배였다.
이는 실제 Layer0 Weight의 Shape별 Event 중앙값을 반복한 추정이다.
36개 Layer의 Exclusive Kernel Time이 아니다.
새 GEMM Kernel로 일부 차이를 줄일 수 있다.
Attention Work의 축별 분배는 GEMM 변경만으로 같아지지 않는다.
그러나 이 사실만으로 Serving 우위가 보장되지는 않는다.

주요 비용의 배타적 기여율은 `NOT_IDENTIFIED`다.
GEMM·Attention·Queue·Transfer의 중앙값을 더해 TTFT 개선을 분해하지 않았다.
Layer의 추가 Hidden/Residual은 Request별131,072,000Bytes다.
유효 KV Bytes가 같아도 현재 Layer Cache의 Source Block Credit은 더 크다.
이 불리한 차이도 그대로 유지했다.''')
    add('''## TP2 — 최초 차이 식별, Strict Baseline 차단

기존 N8 중 가장 이른 Output 분기는 Request4의 Position6이었다.
새 Eager 순차 Capture에서는 같은 Prefix의 Layer0 `self_attn.o_proj`가 최초 차이였다.
수집한 앞선 Q/K/V·RoPE·Attention Rows는 일치했다.
Prefill 마지막8 Rows의 Max Absolute Error는0.015625였다.
RMS Error는0.00020561653946060687였다.
Relative L2는0.003151108743622899였다.

실제 Operand와 Weight Shard를 합치면 TP1과 정확히 일치했다.
Full-K BF16 GEMM은 실제 TP1을 재현했다.
K/2의 BF16 부분합 두 개와 합산은 실제 TP2를 재현했다.
Collective 전의 Local BF16 Projection 반올림이 이 경계의 차이를 설명한다.
실제 Collective Backend는 PYNCCL이었다.
Custom All-reduce는 비활성화됐다.
PyTorch의 `BF16_reduced_precision_reduction` 옵션도 거짓이었다.
NCCL 내부 Precision이나 Ring/Tree Kernel 선택까지 계측한 것은 아니다.
잘못된 Weight Shard나 누락된 Rank 합산으로 설명되지 않는다.
이 결론은 모든 TP2 Runtime 오류를 배제하는 증명은 아니다.

Step6의 TP1 Logit은 Token1986=28.125, Token785=28.0이었다.
TP2는 둘 다27.875였고 실제 Argmax는785였다.
FP32 Replay는 진단용이다.
FP32나 Token 덮어쓰기를 BF16 성능 경로에 넣지 않았다.

판정은 `NUMERICAL_NON_EQUIVALENCE_IDENTIFIED`다.
TP1 Strict Parity는 `FAILED`다.
전체 TP2 KV/Hidden의 Within-TP2 검증은 `NOT_MEASURED`다.
통일된 Client Timestamp의 적격 TP2 성능 비교는 `NOT_RUN`이다.
TP2를 Main Table에 넣지 않는다.''')
    add('## Dynamic의 독립 기여와 Matched Geometry')
    controls=load('analysis/heldout_policy_matched_actual_geometry_statistics.json',{})
    add(f"실제 Histogram/Geometry Control은 {controls.get('qualified_pairs',0)}개 독립 Pair가 Matching을 통과했다.")
    if controls.get('metrics'):
        lines=['| Metric | Dynamic/Matched Control, 95% CI |','|---|---|']
        for key in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s'):
            lines.append('| '+key+' | '+ratio(controls['metrics'].get(key))+' |')
        values=controls.get('SLO_regimes',{}).get('Decode_priority',{})
        lines.append('| Decode-priority SLO Output tokens/s | '+ratio(values.get('Output_token_SLO_goodput'))+' |')
        add('\n'.join(lines))
    for qps in (4.8,6.4):
        extra=load(f'analysis/heldout_O64_q{qps:g}_policy_matched_actual_geometry_statistics.json')
        admission=load(f'analysis/O64_q{qps:g}_conditional_matched_admission.json')
        if extra:
            add(f"Output64/QPS{qps:g}의 추가 Matching 통과 Pair는{extra['qualified_pairs']}개다.\n이 추가 Control은 Primary 결과가 유망할 때만 수행하는 사후 Diagnostic이다.")
            lines=['| Metric | Dynamic/추가 Matched Control, 95% CI |','|---|---|']
            for key in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s'):
                lines.append('| '+key+' | '+ratio(extra['metrics'].get(key))+' |')
            lines.append('| SLO Output tokens/s | '+ratio(extra.get('SLO_regimes',{}).get('Decode_priority',{}).get('Output_token_SLO_goodput'))+' |')
            add('\n'.join(lines))
            add('유망한 Primary 조건을 본 뒤 실행한 Control이다.\n미리 고정한 새 독립 Workload의 최종 Confirmation을 대신하지 않는다.')
        elif admission:
            add(f"Output64/QPS{qps:g}의 추가 Matched Control Admission은{bool(admission['admit'])}였다.\nNative-relative Gate와 SLO Rate의 반복 우위가 함께 성립하지 않으면 추가 GPU Control을 실행하지 않았다.")
    add('''이 Control은 실제 Cut Histogram·완료 Work·Source/Destination 구성원을 맞춘다.
Role별 실제 GPU Shape Histogram도 별도 검사한다.
Cut의 시간 배치는 의도적으로 다르다.
Control은 사후 Aggregate Cut Count를 사용한다.
따라서 배포 가능한 Best Static을 대신하지 않는다.
Cache 상태와 Retention Duration까지 같다는 뜻도 아니다.

정책은 이미 도착한 FIFO 작업만 사용한다.
P Prefix 제출 전에 Last0/2/4/8/12를 평가한다.
현재 Group의 KV·Hidden Reservation과 유효 Deadline Conflict를 검사한다.
기존 Cohort가 끝나기 전에는 새 Suffix를 Admission하지 않는다.
미래 Arrival 또는 미래 실측 완료시각은 Online 입력이 아니다.
110ms 추가 Margin과 실측 Tail은 WCET가 아니다.

실제 Source 가변 Group Discovery에서 다섯 Cut을 모두 선택했다.
선택 횟수를 늘리기 위해 작은 Cut을 강제하지 않았다.
그러나 Cut 다양성만으로 Strong Static보다 좋은 정책이라고 판정하지 않는다.
새 문서의 Static/Dynamic 비율과 Native-relative CI를 함께 검사해야 한다.''')
    predictions=load('analysis/online_prediction_and_deadline_validation.json',{}).get('runs',[])
    selected=[r for r in predictions if r.get('phase')=='FROZEN_HELDOUT_CANDIDATE' and
        any(r['name']==v['name'] and v['method']=='Dynamic' and
            (v['cap'],v['QPS']) in ((16,6.4),(64,4.8),(64,6.4)) for v in data['rows'])]
    if selected:
        lines=['### 실제 Online Predictor 검증 — TRACE_ANALYSIS','',
            '아래 표본은 같은 Run 안의 상관된 Source Group이다.',
            'Group 수를 독립 Server Start 수로 사용하지 않는다.',
            'First-output 예측은 GPU Kernel 완료가 아니라 Client 첫 출력 경계를 대상으로 한다.','',
            '| Run | Group 수 | P-ready MAE, ms | First-output MAE, ms | First-output 절대오차 p95, ms | 과소예측, % | Upper-bound Coverage, % | Controller p95, ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in selected:
            lines.append('| '+row['name']+' | '+str(row['controller_host_ms']['count'])+' | '+
                ' | '.join(number(value) for value in (row['source_ready_MAE_ms'],row['first_output_MAE_ms'],
                    row['first_output_absolute_error_ms'].get('p95'),
                    100*row['completion_underprediction_fraction'] if row['completion_underprediction_fraction'] is not None else None,
                    100*row['upper_bound_coverage'] if row['upper_bound_coverage'] is not None else None,
                    row['controller_host_ms'].get('p95')))+' |')
        add('\n'.join(lines))
    add('''## Oracle와 변동성 — OFFLINE_ORACLE와 TRACE_ANALYSIS

상태 갱신 Replay는 P/D 완료시간과 Credit Lifetime을 갱신했다.
모델은 실제 Kernel Trace가 아니라 Host Service Enclosure를 사용했다.
유한 Beam은 전역 최적해나 성능 상한이 아니다.

같은 Mixed Output 모델에서 Joint95% Static은231.036054 SLO tokens/s였다.
Beam1024는235.761290이었다.
추가 Headroom은2.045%였다.
이는 유망 기준5%보다 작았다.
이 Grid의 대규모 Mixed Sweep은 진행하지 않았다.

실제 Mixed Output n=1 Pilot에서 Online H2는230.700 SLO tokens/s였다.
Fraction Static은230.853이었다.
두 방법의 Joint Attainment는 각각90.625%와88.28125%였다.
둘 다Joint95% Gate를 통과하지 못했다.
미래 정보로 선정한 Beam Plan의 GPU Diagnostic은 Online 성과가 아니다.
모델의 작은 Headroom을 실측 Dynamic 이득으로 바꾸어 보고하지 않는다.

Host 예산의 변동성은 기록했다.
이를 물리적인 Decode-safe GPU Idle로 해석하지 않는다.
실제 Mixed-input, Natural EOS와 Bursty Serving의 폭넓은 검증은 `NOT_RUN`이다.
Input8000 외에는 현재 정책이 보수적으로 Native를 선택한다.
모든 Heterogeneous Workload에서 Dynamic이 불가능하다고 주장하지 않는다.''')
    add('''## 계측·실패·Runtime 비용

CUDA Event A/B/A에서 Layer Capture의 네 Metric 변화는5% 이내였다.
Token Capture의 TTFT p95 비율은1.081923배로 간섭 기준에 실패했다.
Token Event Capture는 Diagnostic으로만 사용한다.
Event Enclosure를 Exclusive Kernel Time이나 Kernel Union으로 표시하지 않는다.
두 Run의 시간을 합쳐 하나의 실제 Timeline으로 만들지 않는다.

별도 CPU Credit Lane은 RPC 지연을 줄였다.
그러나 전체 처리량 개선은 재현하지 못했다.
이 옵션은 최종 Main 정책에서 비활성화했다.
이를 Dynamic의 기여로 포함하지 않는다.

Mixed Static의 진행 정지 한 건은 실패로 보존했다.
D Free16613 Blocks 중 미래 예약16152 Blocks가 남았다.
유효461 Blocks는 Native Prompt 한 개의501 Blocks보다 작았다.
전체 Cohort Import Barrier와 미래 Reservation이 순환 대기를 만들었다.
마지막 Callback 순서는 `NOT_IDENTIFIED`다.
예약 상한32→16의 별도 진단은 Output128개와 Credit 복원을 통과했다.
실제 Cut이 달라졌으므로 성능 비교로 사용하지 않았다.

Mixed H2의 Client Joint 실패12개 중8개는 ITL 실패였다.
그8개는 Worker 간격이 기준 이내였다.
Client 간격과 Worker 간격의 차이는 Delivery Lag 변화로 확인됐다.
Client SLO를 Worker 경계로 바꾸지 않았다.
특정 함수의 배타적 원인은 `NOT_IDENTIFIED`다.

새 Output64/QPS4.8의 Seed2026111741에서도 Client ITL 실패가8 Requests에서 발생했다.
TTFT와 TPOT 실패는 없었다.
Worker 간격은 모두31ms 이내였다.
q56의 Client42.641804ms는 Worker24.019083ms와 Delivery Lag 변화18.622721ms로 분리됐다.
이8개 Request의512 Output Tokens는 원래 SLO 집계에서 제외했다.
이 지연을 제거한 Counterfactual Goodput을 Main 결과로 사용하지 않았다.

일부 UCX Context Query Code4는 Coordinator COMPLETE 뒤에 발생했다.
이는 Serving 중의 CUDA Launch Failure와 별도로 분류했다.
프로세스 전체가 Error-free였다고 주장하지 않는다.
Warmup 없는 State Capture와 측정 전 JIT 경고도 Clean 성능과 분리했다.
초기 Hook·검증기·Warmup·분석 Schema 실패를 모두 보존했다.''')
    add('''## 최종 질문에 대한 답

1. TP2의 첫 유효 Tensor 차이는 수집한 Layer0 Output Projection이다.
2. 같은 Operand Replay가 BF16 Local Projection 반올림의 수치 차이를 재현했다.
3. Strict TP1 Parity를 가진 TP2 성능 Baseline은 확보하지 못했다.
4. 실제 Dense AR Token-axis P→D 계산 이전은 구현했다.
5. Causal Prefix KV·전체 Prompt KV·Hidden·Work·ACK Gate를 별도 State 실행에서 검사했다.
6. 같은 Last12 작업량에서는 Layer가 유리했다. Last8 초기 Control은 반대였다.
7. 같은 SLO의 우위 범위는 위 독립 Start Table에 한정한다.
8. Attention 분배와 일부 GEMM Shape 차이는 확인했다. 배타적 기여율은 미식별이다.
9. Host-state 예산과 실제 Cut의 변화는 관측했다. 물리적 Safe Idle은 미측정이다.
10. 강한 Joint95% Static 대비 근사 Beam의 모델 Headroom은2.045%였다.
11. Mixed Pilot의 Dynamic SLO Rate는 Static보다 높지 않았다. Primary 세 조건에서도 SLO Rate의 반복 우위를 입증하지 못했다.
12. Token Suffix·Chunk·Group과 Static Cut·Fraction·Group을 Discovery에서 탐색했다.
13. 실제 Histogram·Work·Group·GPU Shape Matching은 별도 파일에 기록한다.
14. Mean TTFT·Mean TBT·ITL·Raw/SLO Token Rate는 위 Table과 Per-start CSV에 보존했다.
15. 구현·State Gate의 통과와 Performance·Dynamic 독립 기여의 통과를 구분한다.

같은 작업량의 조건부 Layer 이점은 조건과 제한을 명시한 실험 결과로 기록할 수 있다.
전역 최적 Token Baseline에 대한 우위나 Dynamic의 독립 Systems Contribution을 확정하지 않는다.
전역 우위·Sustainable Capacity·미실행 Workload의 성공을 주장하지 않는다.''')
    inventory=load('analysis/experiment_inventory.json',{})
    tests=load('analysis/final_CPU_verification.json',{})
    restored=load('state/final_GPU_resource_restoration.json',{})
    archive=load('archives/large_evidence_manifest.json',{})
    add('## 최종 검증과 Resource 복원')
    add('새 문서의 Clean Serving 평가74개 Start를 완료했다.\n'+
        '이 중 세 Primary 조건은 각각 Method5개×독립 Start3개다.\n'+
        '추가 QPS의 Candidate1회 결과는 탐색으로만 표시한다.\n'+
        f"실험 Inventory의 상태 수는{inventory.get('counts',{})}다.\n"+
        '이 수는 정합성 Capture와 진단도 포함한다.\n모든 MEASURED가 Primary 적격 성능을 뜻하지 않는다.\n'+
        'Raw Directory가 있지만 완료 Receipt가 없는 초기 시도도 실패로 보존했다.\n'+
        f"CPU 계약·분석 검사는24개를 통과했다. Return Code는{tests.get('returncode')}다.\n"+
        f"GPU0–7의 승인된 Burner 복원 검사는{restored.get('passed')}다.\n"+
        '실제 양의 NVML 사용률과 소유한 Worker를3회 연속 확인했다.\n'+
        '실험의 Quiet Lane에서는 GPU0–7을 예약해 승인 Burner를 중단했다.\n'+
        'CPU-only 구간과 최종 종료 뒤에는 승인된 Burner를 복원했다.\n'+
        '다른 사용자의 GPU Process는 종료하지 않았다.')
    if archive:
        add(f"Archive는 원본{archive['original_bytes']:,}Bytes를 보존한다.\n"+
            f"압축 크기는{archive['compressed_bytes']:,}Bytes이며48MiB Part는{len(archive['parts'])}개다.\n"+
            f"전체 Member의 압축 해제 후 SHA256 검사는{archive['verified_lossless']}다.\n"+
            '로컬 원본은 그대로 남겼다.')
    add('''## Figure·근거·재현

- [정합성과 실행 계약](dense_ar_layer_vs_token_correctness_and_execution.md).
- [Layer/Token 성능과 원인](dense_ar_layer_vs_token_online_results.md).
- [TP2 수치 원인](dense_ar_tp2_numerical_correctness_root_cause.md).
- [Dynamic/Static와 Oracle](dense_ar_dynamic_vs_static_oracle_and_online_results.md).
- [Host 예산의 변동성](dense_ar_safe_budget_variance_analysis.md).
- [Figure13개와 Host/Stream Timeline](dense_ar_layer_token_dynamic_main_figures.md).
- [전체 독립 Start와 CI](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_start_statistics.json).
- [Per-start CSV](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_per_start_metrics.csv).
- [실제 Matched GPU Geometry 통계](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_policy_matched_actual_geometry_statistics.json).
- [정상 KV·ACK·Credit 복원](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_normal_lifetime_credit_audit.json).
- [같은 작업량 독립3회](dense_ar_layer_token_dynamic_headroom_20261117/analysis/matched_Last12_independent_start_statistics.json).
- [실험 Inventory와 실패 상태](dense_ar_layer_token_dynamic_headroom_20261117/analysis/experiment_inventory.json).
- [CPU 검사와 이전 Source 보존](dense_ar_layer_token_dynamic_headroom_20261117/analysis/final_CPU_verification.json).
- [Weight SHA256 검사](dense_ar_layer_token_dynamic_headroom_20261117/state/model_weights_SHA256_verification.json).
- [GPU0–7 Burner 복원](dense_ar_layer_token_dynamic_headroom_20261117/state/final_GPU_resource_restoration.json).
- [손실 없는 Archive Manifest](dense_ar_layer_token_dynamic_headroom_20261117/archives/large_evidence_manifest.json).
- [공개 파일 SHA256 Manifest](dense_ar_layer_token_dynamic_headroom_20261117/state/publication_SHA256_manifest.json).
- [Evidence 복원·분석 안내](dense_ar_layer_token_dynamic_headroom_20261117/README.md).

큰 Evidence는 손실 없는 분할 Archive로 공개한다.
원본별 SHA256과 압축 해제 후의 SHA256을 비교한다.
로컬 원본은 삭제하지 않는다.
복원 명령과 Publication 검증은 Evidence Manifest와 Local Receipt에 기록한다.''')
    target=REPO/'poc_flashvep/reports/dense_ar_layer_token_dynamic_headroom_final_summary.md'
    target.write_text('\n\n'.join(blocks)+'\n')
    print('FINAL_SUMMARY_RENDERED',target,flush=True)


if __name__=='__main__':
    main()
