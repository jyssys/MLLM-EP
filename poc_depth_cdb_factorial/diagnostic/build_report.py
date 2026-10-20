"""Write the Korean diagnostic report from derived evidence."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'diagnostic/results'


def read(name):
    return json.loads((P / name).read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    cohort = read('cohort_statistics.json')
    rank = list(csv.DictReader((P / 'rank_clean.csv').open()))
    balance = list(csv.DictReader((P / 'balance.csv').open()))
    traces = [r for r in read('trace_summary.json') if r['C'] == 64]
    times = read('timing_comparison.json')
    parity = read('parity_summary.json')
    contents = ['''# A1 vs A2: Hindsight 추가 이득이 작은 원인

## 결론

주원인은 `DEPTH_COST_MISMATCH`다.
구체적인 실행 원인은 `BATCH_EXECUTION_DOMINANT`다.
보조 원인은 작은 초기 불균형과 `OUTPUT_WORKLOAD_CONFOUNDED`다.
`HINDSIGHT_ASSIGNMENT_WEAK`를 주원인으로 확정할 근거는 없다.

Hindsight는 실제로 Depth work를 균등하게 만든다.
하지만 최적화 대상은 **decode row별 Depth 합**이다.
실제 no-refill 실행은 **배치별 최대 종료 Depth와 반복 호출 수**에도 의존한다.
같은 Work를 줄여도 GPU 실행시간이 같은 비율로 줄지 않는다.

C64에서 Work CV는 4.634%→2.133%로 감소했다. 상대 감소는 53.963%다.
반면 최대 recurrent batch 호출 수는 982→987로 증가했다.
Makespan 중앙값은 7.304→7.323초다. A1/A2는 0.997×다.
5회 paired bootstrap 95% CI는 [0.997, 1.038]×다.
따라서 18.679ms 차이를 확정적인 성능 저하로 해석하지 않는다.

기존 GPU 기록을 CPU에서 재분석했다. 신규 GPU 추론은 하지 않았다.
모델, gate, scheduler, 기존 보고서와 원시 결과를 변경하지 않았다.
기존 `QUALITY_CONFOUNDED` 판정도 변경하지 않았다.

## 증거와 분모

| 표기 | 의미 |
| --- | --- |
| LIVE_MEASURED | 보존된 실제 GPU 실행 기록의 재분석. 이번에 새로 측정한 GPU 결과는 아님 |
| TRACE_REPLAY | 같은 실제 출력·Depth trace를 고정한 반사실적 재배치와 실행 횟수 계산 |
| SIMULATED | trace 비용 회귀 또는 명시된 단위비용 가정의 추정 |
| NOT_AVAILABLE | 필요한 측정값이 없거나 해당 counterfactual을 실제로 실행하지 않음 |

속도 cohort는 64개 요청이다. GSM8K/MATH-500/Alpaca/ShareGPT가 각각 16개다.
각 C는 이 순서의 앞 C개다. Cap은 256토큰이며 EOS를 허용한다.
별도 품질 cohort는 GSM8K 128개와 MATH-500 128개다. Cap은 8192다.
이 두 cohort를 합쳐 하나의 모집단으로 취급하지 않는다.

Depth trace는 decode 토큰만 포함한다.
Full-R4 prefill이 만든 첫 output token은 Depth 분모에서 제외한다.
따라서 요청당 `decode_count = generated_count − 1`이다.
분산은 모집단 분산(ddof=0)이다. CV는 표준편차/평균이다.
Token과 rank를 독립 serving sample로 bootstrap하지 않았다.

## 1. 구현 감사

| 항목 | 확인 결과 | 구현 위치 |
| --- | --- | --- |
| A1 배치 | 입력 순서의 `i % 4`. 실질적으로 round-robin | `scripts/freeze.py:81` |
| A2 정보 | 과거 v3의 요청별 실제 decode Depth 합. 길이와 평균 Depth가 함께 들어감 | `scripts/freeze.py:45`, `artifacts/speed_reference_depth_work.json` |
| A2 알고리즘 | Work 내림차순 LPT. 누적 Work, 요청 수, rank 순으로 tie break | `contract.py:17` |
| 비용에서 빠진 항목 | Prefill, batch별 최대 Depth, graph bucket, KV copy, 실제 GPU 시간 | `contract.py:23` |
| 변경 범위 | 최초 request→GPU 매핑. 이후 KV migration 없음 | `scripts/run.py:72` |
| Local admission | 전송 순서는 원래 요청 순서. 매핑이 바뀌면 rank별 부분순서·구성도 바뀜 | `scripts/run.py:72` |
| 내부 scheduler | 두 arm 모두 같은 no-refill 경로 | `scripts/worker.py:197`, `contract.py:11` |
| 공통 설정 | BF16, q=0.5, R2–R4, last_exited, local cap16, decode graph ON | `runtime.py:28`, `artifacts/arm_manifest.json` |
| 실제 topology | GPU4/5/6/7의 독립 TP1/EP1/PP1 replica. EP A2A 없음 | `scripts/run.py:56` |

경로의 기준은 `poc_depth_cdb_factorial/`다.
`runtime.py`의 전체 경로는 `poc_decode_first_depth_aware_dp/runtime.py`다.
모델은 Base Ouro-1.4B다. Revision은 `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`이다.
CDB revision은 `abdfae0adbdd12be8c6123ccb34eee98a38afb39`이다.
요청 ID→physical GPU는 rank 0/1/2/3→GPU4/5/6/7이다.
각 프로세스의 local CUDA device는 0이다.

모든 C·repeat·rank에서 A1/A2 config가 동일함을 검사했다.
다만 실제 graph capture 이력까지 동일한 것은 아니다.
C64 r0의 recurrent graph capture 수는 A1=[0,2,1,2], A2=[2,1,1,0]이다.
r1/r3에서는 두 arm 모두 0이다. r2/r4는 r0 패턴을 반복한다.
Engine reset은 counter를 지우지만 기존 graph 객체는 보존한다.
Lazy capture의 별도 소요시간은 NOT_AVAILABLE이다.
이를 제거한 가상의 warm-cache 성능을 실측이라고 주장하지 않는다.

LPT는 greedy 근사다. 이 결과를 Perfect Oracle이라고 부르면 안 된다.
역사적 Work CV도 C64에서 4.669%→1.608%로 감소했다.
따라서 LPT가 자신의 scalar objective조차 개선하지 못했다는 설명은 맞지 않는다.

## 2. Table 1 — Depth와 Output Length

모든 행은 LIVE_MEASURED다. Speed 행은 A1 C64 r0다.
Quality 행은 별도 A1 품질 cohort다.
''']
    rows = []
    for name, domains in [('speed_C64_A1_r0', ['all', 'gsm8k', 'math500']), ('quality_A1', ['all', 'gsm8k', 'math500'])]:
        for domain in domains:
            r = next(s for s in cohort if s['cohort'] == name and s['dataset'] == domain)
            d, l = r['token_depth'], r['request_output_length']
            rows.append([('Speed' if name.startswith('speed') else 'Quality') + '/' + domain,
                         r['requests'], r['recorded_decode_tokens'],
                         f"{d['mean']:.4f}/{d['variance']:.4f}/{d['std']:.4f}",
                         f"{100*d['CV']:.2f}%", '/'.join(f'{100*r[k]:.2f}%' for k in ['R2_fraction','R3_fraction','R4_fraction']),
                         f"{l['mean']:.2f}/{l['P50']:.1f}/{l['P90']:.1f}"])
    contents.append(table(['집합', '요청 N', 'Decode N', 'Depth 평균/분산/SD', 'Token CV', 'R2/R3/R4', '출력길이 평균/P50/P90'], rows))
    contents.append('\n\nTable 1b는 같은 집합의 request-level 분포다. 요청 평균은 token-weighted 평균과 다르다.\n')
    rows = []
    for name, domains in [('speed_C64_A1_r0', ['all', 'gsm8k', 'math500']), ('quality_A1', ['all', 'gsm8k', 'math500'])]:
        for domain in domains:
            r = next(s for s in cohort if s['cohort'] == name and s['dataset'] == domain)
            d, w = r['request_mean_depth'], r['request_total_work']
            rows.append([('Speed' if name.startswith('speed') else 'Quality') + '/' + domain,
                         f"{d['mean']:.4f}/{d['variance']:.5f}/{100*d['CV']:.3f}%",
                         f"{w['mean']:.1f}/{w['variance']:.1f}/{100*w['CV']:.3f}%",
                         f"{100*r['request_output_length']['CV']:.3f}%"])
    contents.append(table(['집합','요청 평균 Depth: 평균/분산/CV','요청 Work: 평균/분산/CV','Output Length CV'], rows))
    contents.append('''

Speed의 Depth P10/P50/P90은 2/3/4다. Quality 전체는 2/3/3이다.
Speed 요청 평균 Depth는 3.0184다. 요청 간 분산은 0.029109, CV는 5.652%다.
요청별 Work 평균은 752.984다. 분산은 9076.328, CV는 12.652%다.
출력길이 CV는 11.407%다. Length–Work Pearson r=0.892, Mean-depth–Work r=0.333이다.

Quality 요청 평균 Depth CV는 3.950%다. 요청별 Work CV는 196.432%다.
Length–Work r=0.9983이다. Mean-depth–Work r=−0.6336이다.
Quality의 큰 Work 차이는 주로 출력길이 차이와 연관된다.
음의 상관은 깊이를 낮추면 시간이 늘어난다는 인과 증거가 아니다.
긴 출력이 token-weighted Depth 평균에도 더 큰 가중치를 준다.

길이가 정확히 256인 Speed 요청 60개만 비교했다.
이때 Mean-depth CV와 Work CV는 모두 5.792%다.
길이를 고정해도 Depth 차이는 남는다. 다만 큰 request-level Work heterogeneity는 아니다.
GSM8K 16개의 평균 Depth는 3.1225, MATH-500 16개는 2.9998이다.
전체 token 분산만으로 요청 간 이질성을 과장하면 안 된다.

이전 v3의 별도 128요청 trace도 검산했다.
Decode 31,946토큰의 평균 Depth는 2.9978이다.
R2/R3/R4는 11.861%/76.498%/11.642%다.
요청 Mean-depth CV는 6.465%, Work CV는 13.120%다.

요청별 값, dataset별 분산, 상관계수, log-variance identity는 아래 파일에 있다.
`Work = decode_length × mean_depth`를 그대로 분해했다. 이를 인과적 기여도로 해석하지 않는다.

- [request_statistics.csv](../diagnostic/results/request_statistics.csv)
- [cohort_statistics.json](../diagnostic/results/cohort_statistics.json)

## 3. Table 2 — C64 실제 GPU별 배치와 Work

Work·Depth·출력은 clean r0 값이다. 시간은 해당 rank의 5회 중앙값이다.
Rank finish는 첫 submit→rank의 마지막 요청 완료다. Exclusive GPU service가 아니다.
GPU별 중앙값을 합성해 cohort makespan을 다시 만들지 않는다.
모든 행은 LIVE_MEASURED다. 두 arm 모두 GPU마다 요청 16개다.
''')
    rows = []
    for arm in ['A1', 'A2']:
        for g in range(4, 8):
            rs = [r for r in rank if r['C'] == '64' and r['arm'] == arm and r['GPU'] == str(g)]
            r = next(s for s in rs if s['repeat'] == '0')
            med = lambda k: statistics.median(float(x[k]) for x in rs)
            rows.append([arm, g, r['output_tokens'], r['total_loop_work'],
                         f"{float(r['depth_mean']):.4f}/{float(r['depth_variance']):.4f}",
                         '/'.join(f"{100*float(r[k]):.2f}" for k in ['R2_fraction','R3_fraction','R4_fraction']),
                         f"{med('rank_finish_s'):.3f}", f"{med('host_decode_span_s'):.3f}",
                         f"{med('host_queue_mean_ms'):.2f}"])
    contents.append(table(['Arm', 'GPU', '출력 N', 'Decode Work', 'Depth 평균/분산', 'R2/R3/R4 %', 'Rank finish s', 'Host decode span s', 'Host queue 평균 ms'], rows))
    contents.append('''

A1은 GPU4=GSM8K, GPU5=MATH-500, GPU6=Alpaca, GPU7=ShareGPT다.
원래 목록이 네 domain을 반복하므로 round-robin이 domain을 GPU별로 분리한다.
A2는 domain을 섞는다. 예를 들어 GPU4는 GSM8K 3, MATH-500 3, Alpaca 5, ShareGPT 5개다.
전체 요청 ID, 요청별 Work 평균/분산, domain 구성은 [rank_clean.csv](../diagnostic/results/rank_clean.csv)에 보존했다.

다음 값은 5회 중앙값이다. Request count CV는 모든 C에서 두 arm 모두 0이다.
''')
    rows = []
    for C in [8, 16, 32, 64]:
        values = []
        for arm in ['A1', 'A2']:
            rs = [r for r in balance if r['C'] == str(C) and r['arm'] == arm]
            values.append({k: statistics.median(float(x[k]) for x in rs) for k in ['work_CV', 'rank_finish_CV', 'finish_skew_ms', 'max_core_calls']})
        t = next(t for t in times if t['C'] == C)
        rows.append([C, f"{100*values[0]['work_CV']:.3f}→{100*values[1]['work_CV']:.3f}%",
            f"{100*values[0]['rank_finish_CV']:.3f}→{100*values[1]['rank_finish_CV']:.3f}%",
            f"{values[0]['finish_skew_ms']:.1f}→{values[1]['finish_skew_ms']:.1f}",
            f"{values[0]['max_core_calls']:.0f}→{values[1]['max_core_calls']:.0f}",
            f"{t['speedup']:.3f}× [{t['CI95'][0]:.3f}, {t['CI95'][1]:.3f}]"])
    contents.append(table(['C', 'Work CV A1→A2', 'Finish CV A1→A2', 'Finish skew ms', '최대 core 호출', 'A1/A2 makespan ratio [95% CI]'], rows))
    contents.append('''

C64의 r0 Work Max/Mean은 1.0575→1.0197이다.
초기 불균형은 작지만 0은 아니다. 균형 개선도 실제로 발생했다.
동일 A1 Work를 완벽하게 4등분하는 선형 모형의 최대 speedup은 1.057×다.
이는 row-work 모형에만 적용하는 상한이다. 실제 GPU 시간의 상한은 아니다.

GPU별 output 수 CV, host decode span CV, finish Max/Mean은 [balance.csv](../diagnostic/results/balance.csv)에 있다.
Tail은 `최종 rank finish − 해당 rank finish`로 계산했다.
이는 요청을 먼저 끝낸 rank의 꼬리 구간이다. 전체 GPU idle 측정값이 아니다.
C64 r0의 A1 tail은 [254.0, 844.1, 1187.2, 0.0]ms다.
A2는 [0.0, 780.6, 870.3, 331.7]ms다.
Clean exclusive GPU decode/prefill time과 whole-GPU busy/idle은 NOT_AVAILABLE이다.
Queue 시간은 host의 첫 scheduled 시점이다. GPU service 시작과 동일시하지 않는다.

Prefill도 균일하지 않다.
A1의 GPU별 prompt token은 [5180,1206,480,857]이다.
A2는 [2007,2091,1837,1788]이다.
Full-R4 prefill row-work는 각각 prompt token의 4배다.
총 prompt token은 같지만 prefill packing과 비용은 달라진다.
A1 GPU4는 prefill batch가 2개다. A2는 각 GPU에서 1개다.
따라서 decode Depth 합만으로 E2E를 예측할 수 없다.

## 4. Table 3 — 실제 Batch Depth 구성과 비용

이 표는 별도 trace lane의 LIVE_MEASURED 값이다.
GPU ms는 CUDA event의 stream bracket이다. Clean makespan의 분해가 아니다.
''')
    rows = []
    for r in traces:
        b = r['active_batch']; m = r['measured_ms']
        rows.append([r['arm'], r['GPU'], '/'.join(map(str, r['by_depth_calls'])),
            f"{b['mean']:.2f}/{b['P50']:.0f}/{b['P90']:.0f}",
            f"{r['final_exit_variance_batch_mean']:.4f}", f"{r['graph_padding_amp']:.3f}×",
            f"{m['compute_recurrent_batch']:.1f}", f"{m['exit_KV_copy']:.1f}", f"{m['stage_prefill_batch']:.1f}"])
    contents.append(table(['Arm','GPU','R1/R2/R3/R4 호출','Active batch 평균/P50/P90','배치 내 최종 Exit 분산 평균','Padding amp','Core ms','Exit KV ms','Prefill ms'], rows))
    contents.append('''

R1–R4는 현재 실행하는 recurrent step이다. 종료 Depth는 R2–R4다.
각 core batch의 현재 step은 동일하다. 현재-step variance는 0이다.
그 batch 안에서 토큰들이 최종 종료하는 Depth는 서로 다를 수 있다.
위 분산은 실제 최종 exit trace와 `(request ID, position)`을 join한 사후 진단값이다.
이 미래 정보를 실제 scheduler가 사용했다고 주장하지 않는다.

No-refill은 한 wave의 exited token을 보충하지 않는다.
같은 wave에 R4 토큰이 하나라도 남으면 R4 호출을 실행한다.
각 rank에서 R1/R2/R3은 모두 255번이다. LPT가 이를 줄이지 못했다.
R4의 평균 활성 batch는 약 1.90–2.97이다. 최대 capacity는 16이다.
실제 graph bucket은 [1,2,4,8,12,16]이다.
따라서 유효 row 감소, padded row 감소, graph 호출 감소는 서로 다른 값이다.
Core 호출 하나를 CUDA kernel 하나라고 세지 않았다. 실제 kernel 수는 NOT_AVAILABLE이다.

같은 wave t에 대해 `batch_rows(s,t)=count(depth_i,t ≥ s)`다.
`row_work=sum(batch_rows)`인 반면 `core_calls=sum(batch_rows > 0)`다.
32개 rank trace에서 이 identity와 per-depth 호출 수가 모두 실제 기록과 일치했다.
이것이 단순 Work balance가 실행시간 balance와 다른 직접적인 근거다.

예를 들어 GPU4의 Work는 12740→12201로 줄었다. Core 호출은 967로 같다.
Trace core bracket은 5854.5→6094.5ms다. KV-copy bracket은 487.1→701.6ms다.
이를 18.679ms clean 차이의 정확한 인과 분해로 사용하지 않는다.
하지만 Work 감소가 core 시간 감소를 보장하지 않는 실제 반례다.

last_exited는 R2 종료 KV를 두 deeper slot에, R3 종료 KV를 한 slot에 복사한다.
이 복사는 다음 deeper 실행 전에 완료해야 한다.
GPU4의 R2-copy flush는 83→207회다. R3-copy flush는 255→255회다.
작은 Depth가 항상 물리 비용을 줄이는 것은 아니다.
이는 depth-indexed storage 정책의 비용이다. EP 통신 비용이 아니다.

구현 근거는 pinned CDB의 아래 위치다.

- `continuous_depth_batching/continuous_api.py:593`: no-refill 정의
- `continuous_depth_batching/continuous_api.py:850`: survivors의 순차 recurrent 실행
- `continuous_depth_batching/continuous_api.py:910`: 다음 실행 전 exit KV flush
- `kv_cache_policy.py:236`: deeper-slot copy 목록
- `continuous_depth_batching/cache.py:105`: gather와 scatter
- `continuous_depth_batching/model_runner.py:899`: bucket/slot별 graph key와 lazy capture

표는 관측한 상관관계와 실행 구조를 보여준다.
GPU clock, host dispatch, graph capture의 개별 시간은 분리하지 못했다.
동일 Work를 물리적으로 강제한 GPU 실험은 하지 않았다.
동일 Work의 batch 조합에 따른 호출 수 차이는 replay와 unit test로만 확인했다.

## 5. Table 4 — Frozen trace의 정보별 Hindsight 비교

같은 A1 r0의 64요청, output length, token별 Depth를 고정했다.
GPU당 요청 16개도 유지했다. 새 배치가 실제로 같은 출력을 낸다는 뜻은 아니다.
각 행은 TRACE_REPLAY다. 마지막 추정 열만 SIMULATED다.
''')
    rows = []
    for r in read('frozen_replay.json'):
        rows.append([r['policy'], f"{100*r['work_CV']:.3f}%", r['max_work'], r['max_core_calls'],
                     f"{r['additive_work_speedup']:.3f}×", f"{r['unit_core_call_speedup']:.3f}×",
                     f"{r['SIMULATED_core_speedup']:.3f}×"])
    contents.append(table(['배치 정책','Work CV','Max Work','Max core 호출','선형 Work speedup','단위 호출 speedup','Core bracket 모형 speedup'], rows))
    contents.append('''

Length-only는 실제 미래 decode token 수만 사용한다.
Depth-only는 길이를 정규화한 요청 평균 Depth만 사용한다.
Depth+Length는 같은 frozen trace의 실제 Depth 합을 사용한다.
Legacy는 원래 A2의 과거 v3 scalar와 매핑을 그대로 사용한다.
모든 LPT는 greedy다. 최적해 또는 Perfect Oracle이 아니다.

Core 비용 모형은 `a×core_calls + b×real_rows + c×padded_rows`의 nonnegative 회귀다.
계수는 [4.8060, 0.09007, 0.0000]ms다.
이 회귀계수를 실제 fixed launch latency나 kernel bandwidth로 해석하지 않는다.
32개 rank trace를 사용했다. Leave-one-C-out MAPE는 2.19–4.01%다.
C64의 MAE는 234.97ms다. Nested cohort이므로 independent request CV는 아니다.
Prefill/copy/host를 제외한다. Clean E2E 비용 모형은 NOT_AVAILABLE이다.

Measured-cost 행은 이 제한된 모형으로 1500회 swap을 시도한 heuristic이다.
예상 core speedup 1.014×는 검증 오차보다 작다. 유효한 E2E 이득으로 주장하지 않는다.
즉 더 좋은 scalar balance나 단순 비용 회귀만으로 큰 이득이 확인되지는 않았다.

특히 Legacy frozen replay는 Work CV를 줄여도 최대 core 호출이 982→992로 증가한다.
선형 Work 모형은 1.037×를 예상한다. 단위 호출 모형은 0.990×다.
실제 A2 출력이 달라지기 전에도 objective mismatch가 존재한다.

선형 완전균형 모형의 상한은 1.057×다.
단위 core-call 모형의 느슨한 상한은 1.127×다.
둘은 서로 다른 가정의 bounds다. 어느 것도 실제 achievable E2E oracle이 아니다.

## 6. Table 5 — 불균형 메커니즘 검증

Gate와 weights는 바꾸지 않았다. 실제 기록된 token Depth만 사용했다.
High-depth-variance는 Quality A1 중 decode가 128개 이상인 요청에서 만들었다.
각 요청의 첫 128개 실제 position을 사용했다.
그 평균 Depth의 하위 16개와 상위 16개를 선택했다.
따라서 이 행은 동일 길이의 사후 선택 진단이며 새로운 자연 workload 결과가 아니다.

High-variance의 요청 평균 Depth는 2.609–3.391이다. CV는 6.739%다.
Adversarial은 heavy 요청을 같은 GPU에 몰되 요청 수는 같게 유지했다.
목적은 메커니즘 확인이다. 이를 성능 주장의 baseline으로 쓰지 않는다.
''')
    rows = []
    for r in read('mechanism_cohorts.json'):
        rows.append([r['cohort'], r['policy'], f"{100*r['work_CV']:.3f}%", r['max_work'], r['max_core_calls'],
                     f"{r['SIMULATED_core_ms']:.1f}", f"{r['SIMULATED_core_speedup']:.3f}×"])
    contents.append(table(['기록 집합','Placement','Work CV','Max Work','Max core 호출','추정 core ms','각 RR 대비 추정 speedup'], rows))
    contents.append('''

앞의 Work/호출 수는 TRACE_REPLAY다. ms와 speedup은 SIMULATED다.
모든 counterfactual의 LIVE E2E는 NOT_AVAILABLE이다.
High-variance adversarial→balanced의 단위 호출 speedup은 506/485=1.043×다.
Core bracket 모형은 1.047×를 예상한다.
Natural adversarial→balanced는 약 1.037× 예상이다.
큰 live 이득은 입증되지 않았다. Depth 변동만으로 충분하지 않을 수 있다.
이 예상 이득은 비용 모형의 오차와 비슷하므로 추가 GPU 실행을 확대하지 않았다.

## 7. A1/A2 출력 공정성 직접 검사

아래 결과는 A3/A4에서 가져온 값이 아니다.
''')
    rows = []
    for label in ['speed_C8_r0','speed_C16_r0','speed_C32_r0','speed_C64_r0','quality_256_A1_A2']:
        p = next(p for p in parity if p['label'] == label)
        rows.append([label,p['N'],p['token_exact'],p['depth_exact'],p['length_exact'],p['work_exact'],p['reason_exact']])
    contents.append(table(['직접 비교','N','Token exact','Depth exact','Length exact','Work exact','Stop reason exact'], rows))
    contents.append('''

C64의 두 arm 모두 output 16043개다. 요청별 길이와 종료 사유는 64/64로 같다.
하지만 token은 44/64, Depth trace는 43/64만 같다.
총 decode Work도 48191→48130으로 0.127% 다르다.
따라서 A1/A2 live 비교는 placement-only 통제 실험이 아니다.
다만 위 frozen replay에서 objective mismatch가 유지되므로 출력 차이만이 원인은 아니다.

Full-answer Quality에서는 140/256 token exact, 69/256 depth exact다.
출력길이는 150/256, Work는 74/256만 같다.
총 output은 210877→233808개다. Decode Work는 599878→655857이다.
이 cohort의 live 시간을 순수한 Depth 배치 효과로 해석할 수 없다.
출력 차이의 정확한 수치 원인은 NOT_AVAILABLE이다.
Batch/graph 차이가 관찰됐지만 logits를 고정한 원인 분리 실험은 없다.

기존 evaluator의 A1/A2 직접 비교는 다음과 같다.
GSM8K는 97/128→99/128이다. 차이는 +1.563pp, paired 95% CI [−1.563,4.688]pp다.
MATH-500은 80/128→74/128이다. 차이는 −4.688pp, CI [−10.156,0.000]pp다.
이는 기존 `answer_matches` 집계다. 잘린 답의 parsed 정답도 포함하므로 품질 동등성을 입증하지 않는다.
원래 truncation은 GSM8K 0→1, MATH-500 11→15개다.

## 8. Timing validity와 남은 한계

Clean timing은 5개 paired whole-cohort repeat로 bootstrap 2000회를 수행했다.
CI는 실행 반복의 불확실성이다. 전체 benchmark 모집단의 CI가 아니다.
Trace의 token/depth는 각 arm의 clean r0와 120/120 요청에서 일치했다.
그러나 trace/clean wall ratio는 0.930–1.125다.
C64는 A1 0.993×, A2 1.042×다.
각 한 쌍뿐이므로 observer overhead와 run-to-run 변동을 분리할 수 없다.
Trace의 wall time을 headline latency로 사용하지 않았다.
각 arm 내부의 5개 clean repeat는 요청별 output·Depth trace가 일치했다.
따라서 이 반복 간 시간 변동은 서로 다른 생성 길이 때문이 아니다.
GPU timestamp anchor uncertainty는 기록돼 있지만 정밀한 cross-rank kernel critical path는 NOT_AVAILABLE이다.
Stage stream bracket을 exclusive GPU busy time으로 바꾸지 않았다.

부족한 정보는 전체 kernel 수, clocks, exact GPU idle, 독립적인 lazy-capture 시간이다.
이 한계 때문에 18.679ms의 항목별 정확한 인과 배분은 하지 않는다.
추가 GPU 재실험 없이도 scalar objective와 실제 batch 실행의 불일치는 확인된다.

## 9. 네 질문에 대한 답

1. **A1이 이미 균형적인가?** 비교적 그렇다. Count CV는 0, Work CV는 4.634%다. 그러나 완전균형은 아니었다.
2. **A2가 얼마나 개선했는가?** 실제 Work CV를 53.963% 줄였다. Max Work는 12740→12270으로 3.689% 줄였다.
3. **왜 makespan이 줄지 않았는가?** No-refill wave의 최대 Depth·반복 호출·KV copy·graph 이력이 scalar Work와 다르다. 최대 core 호출도 줄지 않았다. 출력 차이와 측정 변동도 남는다.
4. **불균형이 크면 유용한가?** 기록 기반 adversarial replay에서는 일부 회복 가능하다. 하지만 1.043–1.047× core 추정일 뿐이다. 큰 live E2E 개선은 아직 입증되지 않았다.

최종 판정은 **DEPTH_COST_MISMATCH**다.
실행 메커니즘은 **BATCH_EXECUTION_DOMINANT**다.
**ALREADY_BALANCED**는 제한된 초기 headroom을 설명한다.
**OUTPUT_WORKLOAD_CONFOUNDED**는 live 비교의 보조 제한이다.

이번 자료는 단순 요청 평균 Depth 예측이나 RL 학습을 바로 시작할 근거가 아니다.
후속 실험이 필요하다면, 같은 frozen trace의 wave 최대 Depth와 KV-copy 비용을 포함하는 모델을 먼저 검증해야 한다.
현재 요청 범위를 넘어 scheduler나 새로운 architecture를 구현하지 않았다.

## 산출물

- [GPU별 Work](../diagnostic/results/gpu_work.png)
- [Depth histogram](../diagnostic/results/depth_histogram.png)
- [Work/Depth variance와 core 시간](../diagnostic/results/work_depth_vs_core_time.png)
- [Frozen replay](../diagnostic/results/frozen_replay.json)
- [Batch별 trace CSV](../diagnostic/results/trace_events.csv.gz)
- [입력 hash manifest](../diagnostic/results/input_manifest.json)
- [구현·artifact 존재 감사](../diagnostic/results/source_audit.json)
- [재현 절차와 실패 기록](../diagnostic/reproduce.md)

기준 factorial commit은 `9726dc30729f160f7772ac484bc5ee01d596351c`다.
기준 v3 commit은 `0eb9c04d1eef6af33c30a0ca2c0244fb26e41950`다.
분석 branch는 `flashvep/hindsight-depth-diagnostic`다.
새 commit SHA는 보고서와 함께 제출한 Git commit을 따른다.
''')
    path = ROOT / 'reports/hindsight_depth_diagnostic_ko.md'
    path.write_text('\n'.join(contents))
    print(path)


if __name__ == '__main__':
    main()
