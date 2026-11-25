"""Close the frozen E/D idle addendum without closing the entire campaign."""
import json
import time

from .common import ROOT,digest,preserve,write
from .token_discovery_analysis import table


def P_busy_states(roi):
    p=roi['roles']['prefill']['kernel_busy_fraction']
    both=roi['E_D_simultaneous_idle_during_P_busy_fraction_total']
    rows=[('P_ACTIVE_E_IDLE',roi['P_busy_E_idle_fraction_total'],'MARGINAL_OVERLAPS_OTHER_ROWS'),
        ('P_ACTIVE_D_IDLE',roi['P_busy_D_idle_fraction_total'],'MARGINAL_OVERLAPS_OTHER_ROWS'),
        ('P_ACTIVE_E_IDLE_D_IDLE',both,'DISJOINT_PARTITION'),
        ('P_ACTIVE_E_ACTIVE_D_IDLE',roi['P_busy_D_idle_fraction_total']-both,'DISJOINT_PARTITION'),
        ('P_ACTIVE_E_IDLE_D_ACTIVE',roi['P_busy_E_idle_fraction_total']-both,'DISJOINT_PARTITION'),
        ('E_P_D_ALL_ACTIVE',roi['all_kernel_active_fraction'],'DISJOINT_PARTITION')]
    assert all(value>=-1e-12 for _,value,_ in rows)
    assert abs(sum(value for _,value,kind in rows if kind=='DISJOINT_PARTITION')-p)<1e-9
    return [dict(state=state,fraction_total_ROI=max(0,value),fraction_of_P_busy=max(0,value)/p,
        ROI_seconds=roi['duration_seconds'],seconds=max(0,value)*roi['duration_seconds'],
        accounting=kind,evidence='TRACE_ANALYSIS',scope='Physical kernel state. Not P-queue presence or actionable slack.') for state,value,kind in rows]


def main():
    preserve();paths={}
    def load(name):
        path=ROOT/'analysis'/name;paths[name]=digest(path);return json.loads(path.read_text())
    low=load('low_load_C25_evidence_v0.json');headroom=load('dual_side_headroom_synthesis_v0.json')
    probe=load('interference_PROBE_C40R4_audit_v0.json')
    suffix=load('suffix_QUANTA_C41R3_report_audit_v0.json')
    assert low['observer_gate']=='PASS' and len(probe['decisions'])==8
    assert all(not item['observed_HTTP_protection_gate'] for item in probe['decisions'])
    assert headroom['dual_side_research_decision']=='HOLD'
    roi=low['central_ROI'];states=P_busy_states(roi);stages=[];thresholds=[]
    for role,value in roi['roles'].items():
        stats=value['idle_windows']
        stages.append(dict(role=role,ROI_seconds=roi['duration_seconds'],QPS=.5,
            kernel_busy_percent=100*value['kernel_busy_fraction'],kernel_idle_percent=100*value['kernel_idle_fraction'],
            no_pending_HTTP_kernel_idle_percent=100*value['no_pending_HTTP_kernel_idle_fraction'],
            pending_HTTP_kernel_idle_percent=100*value['pending_HTTP_kernel_idle_fraction'],
            observed_waiting_mean=value['scheduler_waiting_mean_per_observation'],waiting_max=value['scheduler_waiting_max'],
            **{key:stats[key] for key in ['count','total_seconds','mean_ms','p50_ms','p90_ms','p95_ms','p99_ms','max_ms']},
            independent_initializations=1,evidence='TRACE_ANALYSIS_LOW_LOAD_REPLAY',
            scope='Qualified physical kernel gaps and observation-weighted scheduler queue, not safe windows.'))
        if role in ['encoder','decode']:
            thresholds.extend(dict(role=role,threshold_ms=row['threshold_ms'],
                count_coverage_percent=100*row['count_fraction'],duration_coverage_percent=100*row['duration_fraction'],
                evidence='TRACE_ANALYSIS_LOW_LOAD_REPLAY') for row in stats['threshold_coverage'])
    for name,rows in [('stages',stages),('P_busy_states',states),('thresholds',thresholds)]:
        table(ROOT/'analysis'/f'idle_addendum_final_{name}_v0.csv',rows)
    decision=dict(E_idle='E_IDLE_CONFIRMED',simultaneous_idle='E_D_SIMULTANEOUS_IDLE_CONFIRMED',
        physical_idle_scope='Qualified C25 QPS0.5 replay only, N1, not P-prefill-heavy',
        prefill_heavy='NOT_CONFIRMED',encode_safe_headroom='ENCODE_SAFE_HEADROOM_NOT_CONFIRMED',
        encode_exclusive_service_gate='NOT_MEASURED',decode_safe_headroom='NOT_CONFIRMED_ON_TESTED_PROTOCOL',
        followup='HOLD',E_D_same_imported_KV_safe_quantum_comparison='NOT_MEASURED',
        encoder_maximum_batch_memory_capacity='NOT_MEASURED',main_goal_complete=False)
    write(ROOT/'analysis/idle_addendum_final_decision_v0.json',dict(time=time.time(),sources=paths,
        decision=decision,suffix_cost_evidence=suffix['evidence'],
        scope='Scientific close of the frozen C25/C40/C41 E/D addendum. Remaining Native-policy/counter studies and overall resource/publication closeout are separate.'))
    report=ROOT.parent/'mllm_epd_encoder_decoder_idle_final_summary.md';assert not report.exists()
    lines=['# MLLM EPD: Encoder/Decoder Idle 추가 연구 종합','',
        '후속 연구 판정은 `HOLD`다. 저부하의 물리적 Idle은 확인했다. 안전한 추가 Prefill 실행은 확인하지 못했다.',
        '이 보고서는 완료된 C25·C40·C41 추가 연구의 결과다. 전체 캠페인의 종료나 Serving GO를 선언하지 않는다.','',
        '## 첫 페이지: 연구 질문 8개','',
        '| 질문 | 답과 증거 수준 |', '|---|---|',
        '| 1. E에도 D와 비슷한 Idle이 있는가 | 저부하 C25에서 E87.2674%, D85.6284%의 Kernel Idle 관측. TRACE_ANALYSIS |',
        '| 2. P가 포화되고 E/D가 동시에 Idle인가 | 동시 Idle은 관측. 그러나 적격 Trace의 P Busy는14.0682%다. P 포화와의 동시 조건은 미확인 |',
        '| 3. 어느 쪽 Window가 긴가 | 같은 저부하 Trace의 E Gap Mean·p50·p99가 D보다 길었다. 실제 Safe Budget의 비교는 아님 |',
        '| 4. 안전한 Quantum은 얼마나 다른가 | NOT_MEASURED. E/D128·512 Prefix Probe의 N3 전체 보호 Gate는 모두 미통과 |',
        '| 5. E의 추가 LLM 메모리는 가능한가 | 추가 Full Model·중복 Vision·KV8GiB의 실제 공존 적재와 Main100 실행 완료. 최대 Batch Capacity는 미측정 |',
        '| 6. Online Intervention은 안전한가 | 검증한 조건에서 전체 Encode/Decode/E2E 보호를 입증하지 못함. 실제 요청 Offload 이득도 아님 |',
        '| 7. 다음 구조의 우선순위는 무엇인가 | 먼저 Runtime·고부하 병목·현재 상태 Admission을 규명. 그 후 Request-level부터 검토. Pipeline·Wavefront·Elastic CP는 NOT_RUN |',
        '| 8. EAServe/HydraInfer와 차별화한 근거인가 | Image-level 전달 단위와 Exact Visual-span Continuation을 관측. 이 관측만으로 새로운 기여나 우위를 확정하지 않음 |','',
        'E Idle 활용 자체는 새로운 기여로 주장하지 않는다. 선행연구와 구현은 별도 Related-work 감사에 있다.',
        '원래 Image-only Main400은 보존했다. 사용자 변경 후 후속 실험은 동결 Main100을 사용했다.','',
        '## 독립 판정','',
        '- E_IDLE_CONFIRMED: 적격 저부하의 물리적 Kernel Idle에 한정한다.',
        '- E_D_SIMULTANEOUS_IDLE_CONFIRMED: 같은 저부하 Trace에 한정한다.',
        '- ENCODE_SAFE_HEADROOM_NOT_CONFIRMED: 실제 Probe의 전체 보호 Gate를 통과하지 못했다.',
        '- Encoder-exclusive Service Gate와 E/D 같은 Suffix의 Safe Quantum 비교는 NOT_MEASURED다.',
        '- Prefill-heavy 및 D의 추가 Prefill 보호는 검사한 전체 계약에서 미확인이다.',
        '- 후속 Dual-side 연구는 HOLD다. 모든 Workload에서 물리적으로 불가능하다는 판정은 아니다.','',
        '## 동일 구간의 Kernel·Queue 상태','',
        f'C25는 QPS0.5, Main100 Replay5269 Output Tokens, 독립 초기화1회다. 중앙 구간은 {roi["duration_seconds"]:.6f}초다.',
        '예정 Arrival p10–p90를 사용했다. 이를 Sustainable Steady State라고 주장하지 않는다.','',
        '| Stage | Busy % | Idle % | No HTTP Pending + Idle % | HTTP Pending + Idle % | Waiting Mean/Observation | Waiting Max |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in stages:
        lines.append(f'| {row["role"]} | '+
            ' | '.join(f'{row[key]:.6f}' for key in ['kernel_busy_percent','kernel_idle_percent',
                'no_pending_HTTP_kernel_idle_percent','pending_HTTP_kernel_idle_percent','observed_waiting_mean'])+
            f' | {row["waiting_max"]} |')
    lines+=['','Waiting Mean은 Schedule 관측별 평균이다. 시간 가중 Queue가 아니다.',
        'Kernel Busy는 SM Active·Tensor Core Activity·HBM Bandwidth가 아니다. 해당 Counter는 NOT_MEASURED다.',
        'Kernel Gap이 있고 HTTP Pending이 없더라도 GPU Stream의 완전한 Quiescence를 보장하지 않는다.',
        'TRUE_ENCODE_IDLE·HOST_SCHEDULER_WAIT·INPUT_OR_TRANSFER_WAIT·INTRA_ENCODE_KERNEL_GAP·DOWNSTREAM_BACKPRESSURE의 완전한 배타적 분해는 미완료다.',
        '미확인 상태를 유리한 Idle Category로 채우지 않았다.','',
        '## E/D Kernel Gap의 분포','',
        '| Stage | Count | Total Idle s | Mean ms | p50 ms | p90 ms | p95 ms | p99 ms | Max ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in stages:
        if row['role'] not in ['encoder','decode']:continue
        lines.append(f'| {row["role"]} | {row["count"]} | '+
            ' | '.join(f'{row[key]:.9f}' for key in ['total_seconds','mean_ms','p50_ms','p90_ms','p95_ms','p99_ms','max_ms'])+' |')
    lines+=['','Count CDF와 Idle Duration 가중 Coverage는 다르다. 매우 짧은 Gap이 Count를 지배한다.',
        '공통 Monotonic Clock으로 변환했다. 짧은 GPU 간 선후관계에는 원시 Clock Anchor 오차가 있다.',
        '미래 Window의 길이를 알고 실행할 수 있다고 가정하지 않았다.','',
        '| Threshold ms | E Window Count % | E Idle Duration % | D Window Count % | D Idle Duration % |',
        '|---:|---:|---:|---:|---:|']
    for threshold in [.1,1,5,10,20,50,100]:
        E=next(r for r in thresholds if (r['role'],r['threshold_ms'])==('encoder',threshold))
        D=next(r for r in thresholds if (r['role'],r['threshold_ms'])==('decode',threshold))
        lines.append(f'| {threshold:g} | {E["count_coverage_percent"]:.6f} | {E["duration_coverage_percent"]:.6f} | '+
            f'{D["count_coverage_percent"]:.6f} | {D["duration_coverage_percent"]:.6f} |')
    lines+=['','## P Busy와 동시 Idle','',
        '| Physical State | 전체 ROI 대비 % | P Busy 대비 % | Accounting |', '|---|---:|---:|---|']
    for row in states:
        lines.append(f'| {row["state"]} | {100*row["fraction_total_ROI"]:.6f} | {100*row["fraction_of_P_busy"]:.6f} | {row["accounting"]} |')
    lines+=['','첫 두 Marginal 행은 다른 행과 겹친다. Disjoint 네 행만 P Busy를 분할한다.',
        'P Busy와 P Queue 존재는 다른 조건이다. 위 표는 지속적인 P Queue 시간의 분해가 아니다.',
        'Idle Time Overlap을 Actionable Slack Overlap 또는 동일 요청의 3GPU 병렬화 가능성으로 해석하지 않는다.',
        '고부하 C11·C27·문서 C39의 Observer Gate 실패는 DIAGNOSTIC으로 유지했다.',
        'Runtime Source가 다른 점이나 Short ROI를 하나의 적격 QPS 곡선으로 연결하지 않았다.','',
        '## 실제 추가 Prefill과 메모리','',
        'C40은 실제 Online E/P/D Main100에 별도의 실제 Prefix Probe를 추가했다.',
        'E/D128·512를 Natural과 Replay에서 각각 독립 N3로 실행했다. 새 모델 초기화와 실제 추가 작업을 확인했다.',
        '같은 Block 안의 조건은 초기화를 공유했다. 조건별 독립 초기화 수는 3개다.',
        '추가 작업은 613회,195200 Prefix Tokens,5465600 Token-layer Rows다. 버린 Sample은 Client Throughput에 포함하지 않았다.',
        '8개 조건의 전체 보호 Gate는 모두 미통과다. 일부 Metric의 개선과 모든 부정적인 결과를 함께 보존했다.',
        '추가 Probe가 없는 Control에서도 실패 Start가 있었다. 모든 Gap을 Probe의 GPU 간섭으로 단정하지 않는다.',
        'E의 HTTP p95는 Encoder-exclusive GPU Service가 아니다. 원래 서비스 Gate를 대신하지 않았다.',
        'E는 추가 Full Model과 중복 Vision·KV8GiB를 실제 적재했다. Main100 후 Final Device Free는 19,053,608,960–21,605,842,944 bytes였다.',
        'Device 표본은 Process Allocator와 구분했다. 최대 Encoder Batch나 OOM 경계는 측정하지 않았다.','',
        'C41은 실제 Prefix KV Import가 있는 D Suffix128/256/512/1024/2048을 검사했다.',
        '15개 Input×Quantum의 정합성과 작업량 보존은 통과했다. 전체 비용 Observer Gate는 실패했다.',
        '비용은 DIAGNOSTIC이다. GPU-exclusive Kernel·DMA 또는 실제 Safe Budget으로 사용하지 않았다.',
        '미래 Physical Gap과 비용을 비교한 Fit은 OFFLINE_ORACLE이다. Online Speedup이나 보호 성공이 아니다.','',
        '## 재현 자료와 전체 캠페인의 남은 작업','',
        '모델은 Original BF16 Qwen2.5-VL-7B-Instruct, Full28, 원래 Vision·Processor·MRoPE다.',
        'Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. GPU4/5/6의 E/P/D를 사용했다.',
        'P/D B8192/S1024, KV42GiB, Block128, Prefix Cache Off다. E의 B114688은 별도 Encoder Budget이다.',
        'C25 Stock Compiled와 C40/C41 Common Compiled BI0의 Source·Compiler 계약을 섞지 않았다.',
        'Main100 Hash는 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1이다.',
        '분포·원문·이미지·문서 Split은 그대로 유지했다. Full Task Quality는 PENDING이다.',
        'analysis/idle_addendum_final_*_v0에 표·Source SHA·독립 판정을 저장했다.',
        '상세 N3 Gate·메모리·비용은 mllm_epd_dual_side_prefill_headroom_report.md에 있다.',
        '대응 Figure와 CSV는 idle_scope_*·low_load_C25_*·interference_PROBE_C40R4_*·suffix_QUANTA_C41R3_*에 있다.',
        '후속 Native Transport·Actual Batch 감사와 전체 Resource·Git Closeout은 별도 작업이다.',
        '현재 파일은 그 작업이 완료됐다고 주장하지 않는다.','']
    report.write_text('\n'.join(lines));preserve();print('IDLE_ADDENDUM_SCIENTIFIC_SUMMARY_FROZEN',flush=True)


if __name__=='__main__':main()
