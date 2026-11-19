"""Summarize historical accounting and prespecify discriminating tests."""
import json
import time
from .common import PREVIOUS,REPO,ROOT,BASE,write


def main():
    doc=json.loads((ROOT/'analysis/historical_guard_decomposition.json').read_text())
    selected=[r for r in doc['aggregate'] if r['stage']=='cohort2_confirmation' and r['method']!='Native']
    header='| Axis | Last | Ready→D ms | Prior Guard overlap ms | Overlap % | P effective stage ms | D slot/request ms | D/P ratio | Reservation rejects |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'
    rows=[]
    for r in selected:
        m=r['metrics'];wait=m['mean_Ready_to_D_ms'];guard=m['prior_guard_overlap_ms_per_request']
        rows.append('| '+ ' | '.join(map(str,[r['method'],r['layers'],f'{wait:.6f}',f'{guard:.6f}',
            f'{100*guard/wait:.6f}',f'{m["P_submit_to_notification_mean_ms"]:.6f}',
            f'{m["D_cohort_host_occupancy_per_request_ms"]:.6f}',f'{m["effective_stage_balance_D_over_P"]:.6f}',
            r['reservation_rejections']]))+' |')
    table=header+'\n'+'\n'.join(rows)
    text=f'''# 기존 Layer/Token Evidence의 대기 원인 감사

## 범위

Source Commit은 `{BASE}`이다.
이 보고서는 기존 실제 Batch 일치 140개 Start를 재분석한다.
새 GPU 성능 측정으로 표시하지 않는다.
새 Evidence Namespace는 `dense_ar_axis_sweetspot_knob_discovery_20261119/`이다.
기존 Source, Raw Evidence와 보고서는 변경하지 않았다.

## 재현

기존 Output16/QPS6/준비1·Decode2의 독립 Start 3회를 사용했다.
Last8 Layer의 Ready→D 평균은 391.853506ms이다.
Token의 같은 평균은 1595.630584ms이다.
차이는 1203.777077ms이다.
원시 `native_handoffs.jsonl`과 `D_actual_submissions.jsonl`에서 다시 계산했다.
Token의 추가 대기가 1000ms 미만이라는 검사는 이 데이터에서 실패했다.
이는 미해결 현상의 재현이다. 새 GPU 실행 실패가 아니다.

## Host Guard 구간 회계 — TRACE_ANALYSIS

{table}

각 수치는 Start 안의 평균을 독립 Start 3개 사이에서 평균한 값이다.
새 CI를 계산한 표가 아니다. 기존 성능 CI와 분리한다.
P effective stage는 P Host Submit→Source Ready Notification이다.
D slot은 Cohort Host Submit→마지막 Client Output이다.
두 값 모두 Host·Transport·실행 비용을 포함한다.
Exclusive GPU Service Time이 아니다.

기존 Coordinator는 이전 Cohort가 남아 있으면 다음 Cohort를 수용하지 않는다.
각 Request의 Ready→D 구간을 실제 이전 Cohort 구간과 대조했다.
현재 요청의 Partner Ready 대기와 겹치는 구간도 따로 계산했다.
겹친 시간을 두 번 더하지 않았다.
나머지 FIFO Partner 대기, Collection/Host Gap과 미계측 Residual을 분리했다.
각 Request에서 성분의 합은 실제 Ready→D와 일치한다.

Last8/12의 큰 대기는 대부분 이전 Cohort Guard 구간과 겹친다.
이 구간에 GPU가 계속 계산했다고 확인한 것은 아니다.
GPU Idle, Exclusive Kernel Busy와 DMA 시간은 이 분석에서 `NOT_INSTRUMENTED`이다.

## 경쟁 가설

1. P/D 처리율 차이가 Cohort Queue를 증폭한다.
   Last4에서는 양쪽 D/P 비율이 1 미만이다.
   Last8/12에서는 1보다 크다.
   이 비율은 유효 Host Stage 회계이다. 물리적 GPU 포화 판정이 아니다.
2. KV Credit이 D 수락을 직접 막는다.
   위 18개 Start의 예약 거절과 공통 Credit 재시도는 모두 0건이다.
   기존 코드는 P 제출 전에 D 예약을 협상한다.
   따라서 이 데이터의 긴 Ready→D를 새 예약 거절로 설명할 근거는 없다.
   Credit 점유가 P 제출과 전체 Pipeline에 미치는 간접 영향은 별도 검사한다.
3. Partner Formation과 First-token Barrier가 지연을 만든다.
   Last4는 이전 Guard와 겹치지 않는 대기의 비중도 크다.
   준비 배치와 Decode 배치가 같아도 Partner Ready 시점은 달라질 수 있다.
4. Handoff/Import가 직접 원인이다.
   실제 Import/D Forward의 세부 경계는 기존 Trace만으로 완전히 분리되지 않는다.
   이전 Cohort의 전송 비용은 Guard 점유를 통해 다음 요청에 간접 영향을 줄 수 있다.
   Ready→D 전체를 DMA 시간이라고 표시하지 않는다.

## 선택한 반증 실험

먼저 실제 배치1의 Last2를 새 N8→N32 State Gate로 검사한다.
원래 BF16, Full36, KV/Hidden, Layer 방문과 ACK 계약을 유지한다.
성능은 9의 배수 요청 수로 정확한 GEMM Rows를 맞춘다.
N8/N32 정합성은 비정수 작업량의 완전 일치 비교가 아니다.

Low-load에서 P/D 유효 Stage Demand를 측정한다.
Kernel 계측은 별도의 5% Observer Gate로 검사한다.
호출 API는 기존 Serving의 Batch-invariant Linear를 유지한다.

가장 먼저 바꿀 Counterfactual은 Source Outstanding Window32→4이다.
같은 Axis·Cut·Arrival·Output·실제 Batch를 유지한다.
Layer와 Token 양쪽에 동일하게 적용한다.
Ready→D만 줄고 전체 TTFT가 비슷하면 대기 위치의 이동을 지지한다.
Service Cost 또는 전체 처리율이 크게 바뀌면 그 차이도 기록한다.
Window4가 Pipeline을 굶길 수 있으므로 결과를 미리 가정하지 않는다.

## 초기 Knob 후보 — 아직 선택하지 않음

첫 후보는 P/D 유효 Service Demand의 비율이다.
포화 경로의 예측 Cycle은 `max(P_effective, D_slot_per_request)`로 검사한다.
이 모델은 Queueing Theory의 해석용 가설이다.
M/M/1 또는 M/G/1의 가정을 확인했다고 주장하지 않는다.

두 번째 후보는 도착·Backlog에 따른 부하 수준이다.
Queue가 거의 없을 때는 개별 Critical Path를 별도로 검사한다.
실시간 Feature에는 미래 완료시간이나 미래 Arrival을 넣지 않는다.
고정 비용 Lookup과 실시간 관측값의 수를 구분한다.

Held-out Seed, QPS, Output 및 실제 Decode Batch에서 반례를 검사한다.
성능 차이가 불확실한 Cell을 확정적 승자로 만들지 않는다.
Oracle은 Queue·Credit·Cohort 상태를 갱신한 Replay에서만 검사한다.
모델 오차가 크면 Headroom은 검증되지 않은 추정으로 남긴다.

## 현재 판정

판정은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`이다.
직접적인 Host Guard 차단 구간과 예약 거절 부재는 확인했다.
왜 Guard의 Service Demand가 다른지는 새 GPU 계측과 Counterfactual이 필요하다.
하나 또는 두 Knob의 Held-out 설명력은 아직 `NOT_MEASURED`이다.

근거는 `analysis/historical_guard_decomposition.json`에 있다.
이 파일은 입력 Trace Hash와 Request별 구간을 포함한다.
'''
    (REPO/'poc_flashvep/reports/dense_ar_axis_existing_evidence_root_cause_audit.md').write_text(text)
    decisions=[dict(id='A0001',epoch=time.time(),phase='OBSERVE',
        hypothesis='Source/Destination effective service imbalance is amplified by the whole-cohort guard.',
        competing=['KV reservation blocking','Partner barrier','Transfer/import timing'],
        observation='Last8 Token minus Layer mean Ready-to-D1203.777077ms, with no reservation rejections in the18 baseline starts.',
        evidence='TRACE_ANALYSIS',status='PARTIALLY_SUPPORTED_NOT_GPU_COMPUTE_PROOF'),
        dict(id='A0002',epoch=time.time(),phase='DESIGN',
        test='Source Outstanding Window32 versus4 at fixed P1/D-preparation1/Decode2, Last8, Output16.',
        prediction='A queue-location effect reduces Ready-to-D while moving waiting before P; total TTFT need not improve.',
        next='Run exact Last2 state qualification and low-load action calibration before selecting new held-out cells.')]
    path=ROOT/'state/research_decisions.jsonl'
    if not path.exists():
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in decisions))
    write(ROOT/'state/phase_A_complete.json',dict(epoch=time.time(),historical_starts=140,
        new_GPU_measurement=False,verdict='PARTIAL_ROOT_CAUSE_IDENTIFICATION'))
    print('PHASE_A_REPORT_WRITTEN',flush=True)


if __name__=='__main__':
    main()
