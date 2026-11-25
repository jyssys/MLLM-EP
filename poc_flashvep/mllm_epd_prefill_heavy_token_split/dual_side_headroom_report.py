"""Synthesize measured E/D probes, memory and diagnostic suffix costs."""
import csv
import json
import time

from .common import ROOT,digest,preserve,write
from .token_discovery_analysis import table


def main():
    preserve();sources={}
    def read(name):
        path=ROOT/'analysis'/name;sources[name]=digest(path)
        return json.loads(path.read_text())
    idle=read('low_load_C25_evidence_v0.json')
    probe=read('interference_PROBE_C40R4_audit_v0.json')
    suffix=read('suffix_QUANTA_C41R3_report_audit_v0.json')
    correctness=read('suffix_QUANTA_C41R3_correctness_v0.json')
    reference=read('probe_reference_variation_C40_v0.json')
    assert idle['observer_gate']=='PASS' and len(probe['decisions'])==8
    assert all(not r['observed_HTTP_protection_gate'] for r in probe['decisions'])
    assert correctness['passed'] and len(correctness['cases'])==15 and suffix['query_work_conserved']
    with (ROOT/'analysis/interference_PROBE_C40R4_probes_v0.csv').open(newline='') as stream:work=list(csv.DictReader(stream))
    with (ROOT/'analysis/suffix_QUANTA_C41R3_summary_v0.csv').open(newline='') as stream:costs=list(csv.DictReader(stream))
    assert len(costs)==15 and len(work)==613
    memory=[]
    for block in range(3):
        values=read(f'interference_PROBE_C40R4_B{block}_summary.json')
        assert values['status']=='COMPLETE'
        for phase,ports in values['memory'].items():
            encoder=ports['18840'];helper=ports.get('18843')
            memory.append(dict(block=block,phase=phase,GPU=4,device_total_bytes=encoder['device_total_bytes'],
                device_free_bytes=encoder['device_free_bytes'],E_process_allocated_bytes=encoder['allocated_bytes'],
                E_process_reserved_bytes=encoder['reserved_bytes'],
                E_vision_parameter_logical_bytes=encoder['model_parameter_bytes_by_prefix']['visual'],
                helper_parameter_logical_bytes=helper['model_parameter_logical_bytes'] if helper else None,
                helper_LLM_parameter_logical_bytes=helper['model_parameter_bytes_by_prefix']['language_model'] if helper else None,
                helper_duplicate_vision_parameter_logical_bytes=helper['model_parameter_bytes_by_prefix']['visual'] if helper else None,
                helper_unique_KV_storage_bytes=helper['unique_KV_storage_bytes'] if helper else None,
                scope='Actual memory snapshot. One shared-device value, no pre-helper zero and no maximum batch claim.'))
    quanta=[]
    for result in probe['decisions']:
        arm,mode=result['arm'],result['mode'];selected=[r for r in work if (r['arm'],r['mode'])==(arm,mode)]
        quanta.append(dict(arm=arm,mode=mode,independent_initializations=3,actual_probe_calls=len(selected),
            prefix_tokens=sum(int(r['additional_prefix_tokens']) for r in selected),
            token_layer_rows=sum(int(r['additional_token_layer_rows']) for r in selected),
            actual_additional_work_all_starts=result['actual_additional_work_all_starts'],
            absolute_decode_all_starts=result['absolute_decode_all_starts'],
            complete_HTTP_protection=result['observed_HTTP_protection_gate'],
            decision=result['decision'],client_tasks_completed_by_probe=0,
            evidence='INTERFERENCE_PROBE',scope='Prefix only. No imported KV. Discarded one-token sample is not client output.'))
    for name,rows in [('memory',memory),('actual_probe_work',quanta)]:
        table(ROOT/'analysis'/f'dual_side_headroom_{name}_v0.csv',rows)
    ROI=idle['central_ROI'];final_memory=[r for r in memory if r['phase']=='final']
    ranges={key:[min(float(r[key]) for r in costs),max(float(r[key]) for r in costs)] for key in
        ['clean_before_HTTP_p95_ms','clean_after_HTTP_p95_ms','profile_forward_enclosure_p95_ms']}
    write(ROOT/'analysis/dual_side_headroom_synthesis_v0.json',dict(time=time.time(),sources=sources,
        E_physical_idle='E_IDLE_CONFIRMED_ON_LOW_LOAD_TRACE',
        E_D_physical_overlap='E_D_SIMULTANEOUS_IDLE_CONFIRMED_ON_LOW_LOAD_TRACE',
        prefill_heavy='NOT_CONFIRMED',encode_safe_headroom='ENCODE_SAFE_HEADROOM_NOT_CONFIRMED',
        decode_safe_headroom='NOT_CONFIRMED_ON_TESTED_PROBE_PROTOCOL',
        encoder_exclusive_service_gate='NOT_MEASURED',safe_quantum_distribution='NOT_MEASURED',
        main_task_completion_by_probe=0,additional_probes=len(work),
        additional_prefix_tokens=sum(int(r['additional_prefix_tokens']) for r in work),
        additional_token_layer_rows=sum(int(r['additional_token_layer_rows']) for r in work),
        suffix_cost_evidence=suffix['evidence'],suffix_cost_ranges=ranges,
        task_quality='PENDING',dual_side_research_decision='HOLD',
        scope='Frozen C25/C40/C41 synthesis. No safe future window, actual request-offload speedup or E>D safe-quantum claim.'))
    report=ROOT.parent/'mllm_epd_dual_side_prefill_headroom_report.md';assert not report.exists()
    lines=['# E/D 추가 Prefill Headroom: 실측 종합','',
        '판정은 `HOLD`다. 물리적 Idle과 메모리 적재는 관측했다. 안전한 추가 Prefill Quantum은 확정하지 못했다.',
        '이 문서는 완료된 C25·C40·C41의 범위로 동결한다. 전체 캠페인의 종료 보고서는 아니다.','',
        '## 결과 요약','',
        '| 질문 | 실측 또는 미측정 결과 |',
        '|---|---|',
        '| E와 D의 Idle | 적격 QPS0.5 C25에서 물리적 Idle과 동시 Idle 관측. Prefill-heavy 조건은 아님 |',
        '| E의 추가 Weight 적재 | Original Full Model·중복 Vision·KV8GiB를 E와 같은 GPU4에 실제 적재 |',
        '| 추가 Prefill 실행 | E/D 128·512-token Prefix Probe를 독립 N3로 실행. 모든 조건의 전체 보호 Gate 미통과 |',
        '| Imported-KV Suffix | Input3종×Quantum5종의 15개 Strict 정합성 통과. 실제 Query 작업량 보존 |',
        '| Suffix 비용 | Clean/Profile/Clean Gate 미통과. 비용은 DIAGNOSTIC |',
        '| E가 더 큰 Safe Quantum을 처리하는가 | NOT_MEASURED. E/D의 같은 Imported-KV Suffix 비교도 없음 |',
        '| 실제 요청 완료 기여·Serving Speedup | Probe는 기여하지 않음. 이 표로 Speedup을 주장하지 않음 |',
        '| Encode-safe·Decode-safe Budget | NOT_CONFIRMED. Encoder-exclusive Service Gate는 NOT_MEASURED |','',
        '## 물리적 Idle의 적용 범위','',
        f'C25 중앙 Arrival 구간은 {ROI["duration_seconds"]:.6f}초다. 독립 초기화는 1회다.',
        f'E/D 동시 Kernel Idle은 그 구간의 {100*ROI["E_D_simultaneous_kernel_idle_fraction"]:.6f}%다.',
        f'P Busy와 동시 Idle의 교집합은 전체 구간의 {100*ROI["E_D_simultaneous_idle_during_P_busy_fraction_total"]:.6f}%다.',
        f'같은 교집합은 P Busy 시간의 {100*ROI["E_D_simultaneous_idle_fraction_of_P_busy"]:.6f}%다.',
        '두 분모를 혼합하지 않는다. P Queue가 지속되는 구간의 비율로 대체하지 않는다.',
        '이 QPS0.5 관측을 QPS2 Probe의 실제 미래 Window라고 가정하지 않았다.',
        '고부하의 E 비병목·P 포화·Service Proximity·안전 여유의 동시 조건은 미확인이다.','',
        '## 실제 추가 작업과 보호 Gate','',
        f'실제 Probe는 {len(work)}회다. Prefix Token 합계는 {sum(int(r["additional_prefix_tokens"]) for r in work):,}개다.',
        f'추가 Token-layer Rows는 {sum(int(r["additional_token_layer_rows"]) for r in work):,}개다.',
        '각 Probe의 1-token Sample은 폐기했다. Client Throughput·Goodput 분자에 더하지 않았다.',
        '각 조건은 같은 Main100의 Natural EOS와 Equal-output Replay를 따로 실행했다.','',
        '| Helper | Output | Actual Calls | Prefix Tokens | Absolute Decode≥95% Every Start | Full HTTP Protection |',
        '|---|---|---:|---:|---|---|']
    for row in quanta:
        lines.append(f'| {row["arm"]} | {row["mode"]} | {row["actual_probe_calls"]} | {row["prefix_tokens"]} | '+
            f'{row["absolute_decode_all_starts"]} | {row["complete_HTTP_protection"]} |')
    lines+=['','NativeStrong·NativeCommon·동일 초기화의 No-probe 평균을 각각 대조군으로 사용했다.',
        'E HTTP p95·ITL p99·TPOT p95·TTFT p50/p95의 Paired CI 상한은 각각 1.05 이하여야 한다.',
        'Absolute Decode는 요청별 All-ITL과 TPOT의 AND다. 각 Start에서 95% 이상이어야 한다.',
        f'추가 Probe가 없는 Control24개 중 {reference["failed_absolute_decode_cohorts"]}개도 Absolute Gate를 실패했다.',
        '따라서 모든 드문 Client Gap을 Probe의 GPU 경쟁으로 단정하지 않는다.',
        '실패한 전체 Gate는 유지한다. 다른 일부 Metric의 개선을 이유로 Gate를 제거하지 않는다.',
        'E HTTP는 Encode Queue·Host·실제 Encode의 Enclosure다. Encoder-exclusive GPU Service가 아니다.',
        '고정한 Probe 순서, 다른 Batch와 Natural Output 변화의 한계도 유지한다.','',
        '## 실제 메모리 적재','',
        'E는 원래 Vision Encoder다. 추가 Helper는 LLM만 적재한 경량 Worker가 아니라 원래 Full Model이다.',
        'Logical Vision Parameters는 1,353,100,288 bytes다. 추가 Full Model Parameters는 16,584,333,312 bytes다.',
        '그중 LLM은 15,231,233,024 bytes다. 중복 Vision은 1,353,100,288 bytes다.',
        'Helper Unique KV Storage는 8,587,837,440 bytes다. Native D는 KV42GiB 설정을 유지했다.',
        '서로 같은 KV Capacity라고 표현하지 않는다.','',
        '| Independent Block | Final GPU4 Device Free bytes |', '|---:|---:|']
    for row in final_memory:lines.append(f'| {row["block"]} | {row["device_free_bytes"]} |')
    lines+=['','이는 실제 Workload와 Probe를 마친 뒤의 Device 표본이다. Process Lifetime Peak나 최대 Activation 표본은 아니다.',
        'E와 Helper가 보고한 Device Used를 더하지 않았다. 같은 GPU의 메모리를 중복 집계하지 않았다.',
        'Allocator Reserved는 Weight 크기가 아니다. Process Lifetime Peak도 Cohort별 Peak로 대체하지 않았다.',
        '최대 Encoder Batch Capacity와 OOM 경계는 측정하지 않았다. 메모리 적재만으로 안전한 Prefill을 주장하지 않는다.',
        '이전 C16의 제한된 입력 Memory Fit과 이번 Main100 N3의 표본은 별도 근거다.','',
        '## Prefix와 Suffix 비용의 구분','',
        'C40은 Prefix128/512다. Imported Prefix KV가 없는 실제 추가 작업이다.',
        'C26은 Prefix128/256/512/1024/2048의 별도 E/D 고립 비용이다. 두 Arm의 전체 Observer Gate는 실패했다.',
        'C41은 실제 P Prefix KV를 NIXL로 가져온 D Suffix다. Input은 4831·2961·27840 Tokens다.',
        '15개 조합의 Strict 검사 뒤 Cell·단계별 9개 비용 표본을 측정했다. 독립 초기화는 1회다.',
        f'Clean-before HTTP p95의 Cell 범위는 [{ranges["clean_before_HTTP_p95_ms"][0]:.3f}, {ranges["clean_before_HTTP_p95_ms"][1]:.3f}]ms다.',
        f'Clean-after HTTP p95의 Cell 범위는 [{ranges["clean_after_HTTP_p95_ms"][0]:.3f}, {ranges["clean_after_HTTP_p95_ms"][1]:.3f}]ms다.',
        f'Profile Forward Enclosure p95의 Cell 범위는 [{ranges["profile_forward_enclosure_p95_ms"][0]:.3f}, {ranges["profile_forward_enclosure_p95_ms"][1]:.3f}]ms다.',
        '모든 비용은 전체 5% Gate 미통과로 DIAGNOSTIC이다. GPU-exclusive Kernel 또는 DMA 비용이 아니다.',
        'HTTP는 KV Import·Embedding 소비·Admission·Sampling·Delivery를 포함한다. 겹치는 Forward 비용을 다시 더하지 않는다.',
        'E의 Prefix 비용을 D의 같은 Suffix 비용으로 취급하지 않았다.',
        'C25의 미래 Physical Gap과 비용을 비교한 Fit은 OFFLINE_ORACLE이다. 실제 안전 Window나 Speedup이 아니다.','',
        '## 후속 연구 Gate와 재현 자료','',
        '현재 결과로 ENCODE_SAFE_HEADROOM_CONFIRMED 또는 DECODE_SAFE_HEADROOM_CONFIRMED를 부여하지 않는다.',
        'Request-level Elastic Offloading·Pipeline·Wavefront·Context Parallelism의 실제 3GPU 이득도 미검증이다.',
        '추가 Runtime 정책·Actual Batch 감사는 별도 실험이다. 이 동결된 Probe 결과를 사후 변경하지 않는다.',
        '원래 Vision·Processor·BF16 Weight·Full28·MRoPE·Causal Attention·Sampling을 유지했다.',
        'Model/Processor Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.',
        'Main100 SHA256은 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1이다.',
        'GPU4/5/6, B8192/S1024, Native P/D KV42GiB, Block128, Prefix Cache Off다.',
        'analysis/dual_side_headroom_*_v0의 JSON·CSV에 Source SHA와 실제 작업·메모리 표를 저장했다.',
        'N3의 CI·모든 실패 조합·실제 Probe 수락/거부는 원래 C40 CSV에 보존했다.',
        'Full Task Quality는 PENDING이다. 이 보고서는 전체 캠페인의 종료나 Serving GO를 선언하지 않는다.','']
    report.write_text('\n'.join(lines));preserve();print('DUAL_SIDE_HEADROOM_REPORT_FROZEN',flush=True)


if __name__=='__main__':main()
