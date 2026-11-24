"""Write the required online report from closed, audited Token N3 evidence."""
import csv
import json
import time

from .common import ROOT,digest,preserve,write
from .token_discovery_analysis import table

METHODS=['NativeStrong','NativeCommon','Static10','Idle10']
LATENCY=['TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','TPOT_p95_ms']
RATES=['total_input_output_tokens_per_second','raw_output_tokens_per_second','completed_requests_per_second',
    'SLO_goodput','ITL_pass_percent','TPOT_pass_percent','joint_percent']


def main():
    preserve();report=ROOT.parent/'mllm_epd_generic_token_split_online_serving.md';assert not report.exists()
    source=ROOT/'analysis/token_confirmation_C37_audit_v0.json';audit=json.loads(source.read_text())
    assert audit['status']=='N3_RUNTIME_RECALCULATED' and audit['quality']=='PENDING'
    verification=ROOT/'analysis/campaign_metric_audit_v0.csv'
    with verification.open() as stream:checked={r['run']:r for r in csv.DictReader(stream)}
    assert all(row['run'] in checked and checked[row['run']]['metric_recalculation']=='PASS' for row in audit['starts'])
    statistics=audit['start_mean_CI'];rows=[]
    for subset in ['Main100','Heldout50']:
        for mode in ['natural','replay']:
            for method in METHODS:
                selected=[r for r in audit['starts'] if (r['subset'],r['mode'],r['method'])==(subset,mode,method)]
                assert len(selected)==3
                row=dict(subset=subset,mode=mode,method=method,independent_starts=3,
                    **{key:next(r['mean'] for r in statistics if (r['subset'],r['mode'],r['method'],r['metric'])==(subset,mode,method,key))
                        for key in LATENCY+RATES},
                    absolute_decode_min_start_percent=min(float(checked[r['run']]['actual_all_ITL_AND_TPOT_percent']) for r in selected),
                    quality='PENDING',evidence='MEASURED_ONLINE runtime; source-specific task quality is not fully qualified')
                rows.append(row)
    table(ROOT/'analysis/token_online_C37_start_mean_tables_v0.csv',rows)
    pairs=[r for r in audit['paired_ratios'] if r['metric'] in ['TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput']]
    table(ROOT/'analysis/token_online_C37_paired_gates_v0.csv',pairs)
    lines=['# Generic Exact Token Split: 실제 Online Serving N3','',
        'Runtime 판정은 NO_GO_ON_TESTED_GRID다. 검사한 10% Static·현재 상태 Idle-fallback 정책에 한정한다.',
        '전체 Task Quality는 별도로 HOLD다. Runtime 실패를 품질 미확정만으로 설명하지 않는다.',
        '실제 Online Vision Encoding·Continuous P/D·KV Handoff를 포함했다. C0나 Offline Oracle의 성능표가 아니다.','',
        '## 동결 계약','',
        'Original BF16 Qwen2.5-VL-7B-Instruct, BI0, Full28, 원본 Vision·Processor·MRoPE를 유지했다.',
        'Model·Processor Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.',
        '같은 H100 GPU4/5/6을 E/P/D로 사용했다. P/D는 B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.',
        'Main400을 보존했다. 사용자 변경 지시에 따라 후속 C3는 동결한 Main100을 사용했다.',
        'Main100 SHA256은 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1이다.',
        'Main100과 그 안의 Heldout50은 별도로 평가했다. 이미지·문서 Context 분리는 원래 Manifest를 유지했다.',
        '동일 Arrival Seed2026112402와 Order Seed2026112404를 사용했다. Offered QPS는 2.0이다.',
        '각 Method는 독립 초기화 3회다. Natural EOS와 요청별 동일 Output 수의 Replay는 분리했다.',
        'Natural Output Cap은 2048이다. Replay는 길이만 같으며 Answer String 일치를 보장하지 않는다.','',
        '## 비교군과 Admission','',
        '| Method | 실행 경로 | D 정책 |','|---|---|---|',
        '| NativeStrong | Stock Compiled Full P, 원래 NIXL D | Token Adapter나 추가 D Cap 없음 |',
        '| NativeCommon | 후보와 공통 Compiled Reduction 설정, Full P | 원래 Native Admission 유지 |',
        '| Static10 | 실제 전체 입력의 약10% Contiguous Suffix를 D에 이전 | 기존 D HTTP Stream이 없을 때만 추가 Suffix 실행 |',
        '| Idle10 | 계획 시 P HTTP 작업이 있고 D HTTP 작업이 없을 때만 10% Split, 아니면 Native | Prefix 완료 후 D 상태를 다시 검사하고 안전 조건까지 대기 |','',
        '10%는 실제 LLM Token Sequence 기준이다. Integer Cut은 요청별 Raw Partition에 있다.',
        '필요한 P/D 이미지 Consumer Descriptor는 미리 발행했다. 후보를 On-demand EC RPC 방식으로 잘못 설명하지 않는다.',
        '현재 HTTP 상태는 GPU Queue·미래 Safe Window와 다르다. 이 정책을 검증된 Dynamic Split-size Predictor로 부르지 않는다.',
        '모든 입력 Token은 첫 Output 전에 전체 LLM Layer를 완료했다.',
        'Exact은 계산 누락과 KV 근사가 없다는 뜻이다. 모든 BI0 Batch Geometry의 Byte-wise 출력 동일성을 뜻하지 않는다.',
        '10% 후보는 동결 후 N1/N8 조건부 KV·출력 검사를 통과했다. 25%의 N8 KV 실패는 제외한 채 보존했다.','',
        '## Latency: 독립 Start Metric의 평균','',
        '| Subset | Output | Method | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+f'{row["subset"]} | {row["mode"]} | {row["method"]} | '+
        ' | '.join(f'{row[key]:.3f}' for key in LATENCY)+' |')
    lines+=['','Mean TBT는 모든 Client Token Interval의 평균이다. 같은 Token을 여러 Stage에서 중복 집계하지 않았다.',
        '표의 Percentile은 Start별 Percentile의 평균이다. 요청 300개를 합친 Percentile이 아니다.','',
        '## Throughput와 SLO','',
        '| Subset | Output | Method | Total tok/s | Output tok/s | Completed req/s | Goodput tok/s | ITL pass % | TPOT pass % | Joint % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+f'{row["subset"]} | {row["mode"]} | {row["method"]} | '+
        ' | '.join(f'{row[key]:.3f}' for key in RATES)+' |')
    lines+=['','Total tok/s는 요청의 실제 Input+Output을 첫 예정 Arrival부터 Drain까지의 시간으로 나눈 값이다.',
        'Raw Output tok/s는 완료한 요청의 실제 Output 수를 같은 시간으로 나눈 값이다.',
        'SLO-goodput은 Joint PASS 요청의 Output Token 수만 같은 시간으로 나눈 값이다.',
        'Joint는 TTFT≤5000ms AND 모든 Client ITL≤34ms AND Request TPOT≤33ms/token이다.',
        'Single-output의 TPOT는 0이고 ITL은 빈 집합이다. 원래 Metric 계약을 유지했다.',
        'Absolute Decode는 요청별 All-ITL AND TPOT다. 각 독립 Start에서 95% 이상이어야 한다.',
        '완료된 모든 C37 Raw를 Timestamp에서 다시 계산했고 두 개별 Pass Rate와 실제 AND를 구분했다.',
        '각 Start의 8개 실패 조합·실제 요청 수·Output Token 수는 원시 CSV에 보존했다.',
        'Finite-cohort 처리량을 Sustainable Serving Capacity로 부르지 않는다.','',
        '## Paired Gate와 불확실성','',
        '| Subset | Output | Candidate | Native | Metric | Ratio of Start-means | Paired 95% CI |',
        '|---|---|---|---|---|---:|---|']
    for row in pairs:lines.append(f'| {row["subset"]} | {row["mode"]} | {row["method"]} | {row["baseline"]} | '+
        f'{row["metric"]} | {row["ratio_of_start_means"]:.6f} | {row["paired_start_bootstrap95_CI"]} |')
    lines+=['','Paired CI는 같은 Block의 독립 Start3쌍을 사용했다. 27개 Empirical Resample을 모두 계산했다.',
        'N3의 CI 해상도는 제한적이다. 요청이나 Token을 독립 Server Start로 취급하지 않았다.',
        'Relative ITL p99·TPOT p95와 TTFT p50의 CI 상한은 두 Native 각각에 대해 1.05 이하여야 한다.',
        '개선 조건은 TTFT p95 최소5% 감소 또는 SLO-goodput 최소5% 증가와 우위 방향의 CI다.',
        'Static10은 Decode Gate를 통과했지만 TTFT p50과 Serving 개선 조건을 통과하지 못했다.',
        'Idle10도 전체 Runtime Gate를 통과하지 못했다. Heldout Natural에서는 상대 Decode Gate도 실패했다.',
        'TTFT 개선으로 Decode 악화를 상쇄하거나, 낮은 ITL만으로 성공을 선언하지 않았다.','',
        '## 원인과 미측정','',
        'Main Replay의 Static10은 NativeCommon보다 P HTTP 평균이 305.329ms 짧았다.',
        '반면 추가 D Coordinator 대기는 평균 971.956ms 증가했다.',
        '평균 TTFT 차이는 +933.063ms였다. Paired Start-level 95% CI는 [817.498, 1020.626]ms다.',
        '이는 같은 요청의 중복 없는 Host 경계다. GPU-exclusive 계산이나 TTFT p95의 가법 분해가 아니다.',
        '관측된 D 대기를 0으로 만드는 Oracle은 Queue Feedback과 Decode 경쟁을 다시 계산하지 않는다.',
        '그 Oracle을 실제 Online 개선으로 표시하지 않았다.',
        'Clean 실행의 실제 GPU Batch·GPU-exclusive Operator별 비용·실제 미래 Safe Budget은 NOT_MEASURED다.',
        '적격 저부하 Physical Idle을 고부하의 안전한 Suffix Admission 근거로 대체하지 않았다.',
        'BI0 Encoder Batch 변화와 Task Score 불확실성은 별도의 품질 한계다.',
        'MMLongBench-Doc 공식 외부 Answer Extraction은 승인 없이 호출하지 않았다. Raw Score는 전체 품질 PASS가 아니다.','',
        '## 재현 자료','',
        'Evidence Namespace는 mllm_epd_prefill_heavy_token_split_20261124다.',
        'Config는 configs/token_CONFIRM_C37.json이다. 각 Server Lease에 GPU UUID·Source Archive SHA256·명령이 있다.',
        '원시 Request와 Proxy Timeline은 raw/token_CONFIRM_C37_* 및 raw/proxy_token_CONFIRM_C37_*에 있다.',
        '최초 N3 집계는 analysis/token_confirmation_C37_*_v0다. 새 표는 analysis/token_online_C37_*_v0다.',
        'Figure는 figures/token_confirmation_C37_*_v0와 figures/main100_C37_joint_failure_breakdown_v0다.',
        '각 Figure의 PNG/PDF와 대응 CSV를 보존했다. Native Motivation의 별도 N3 결과와 합쳐 평균내지 않았다.','',
        '관련 보고서: [정합성](mllm_epd_generic_token_split_correctness.md), '+
        '[원시 N3](mllm_epd_protected_token_confirmation_v1.md), '+
        '[Host 경로](mllm_epd_protected_token_critical_path_v1.md), '+
        '[품질 한계](mllm_epd_n3_task_quality_limits_v1.md).','']
    report.write_text('\n'.join(lines))
    write(ROOT/'analysis/token_online_C37_report_audit_v0.json',dict(time=time.time(),source_SHA256=digest(source),
        metric_audit_SHA256=digest(verification),report_SHA256=digest(report),full_quality='PENDING',full_GO=False,
        runtime_decision='NO_GO_ON_TESTED_GRID',scope='Closed C37 only. Whole-request, interference probes and C0 costs are not merged.'))
    preserve();print('TOKEN_ONLINE_REQUIRED_REPORT_RECALCULATED',flush=True)


if __name__=='__main__':main()
