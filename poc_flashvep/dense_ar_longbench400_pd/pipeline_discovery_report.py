"""Report every finite-trace pipeline-load control without winner CIs."""
import json
from .common import ROOT,REPO


def main():
    source=json.loads((ROOT/'analysis/proper_pipeline_load_discovery_v1.json').read_text())
    assert len(source['cells'])==9 and all(r['status']=='DISCOVERY_COMPLETE' for r in source['cells'])
    path=REPO/'poc_flashvep/reports/dense_ar_longbench400_pipeline_load_reference_audit.md'
    marker='## 새 Supplement의 전체 실측 결과'
    prefix=path.read_text().split(marker)[0].rstrip()
    lines=['',marker,'',
        '9개 조건에서 각 10개 Arm을 실행했다. 총 독립 Server Start는 90개다.',
        '각 Method/Cell의 독립 Start는 1회다. 따라서 비율에 Start-level CI가 없다.',
        'Layer/Token의 Cut별 결과와 Native/Hybrid를 모두 보존했다.',
        'Actual P Prefill / D Continuation / D Decode Batch는 1/1/1이다.',
        'B8192/S1024, Source Credit4, KV42GiB/Block128과 BI0를 유지했다.','',
        '| Input Bucket | Output | QPS | N | Reference 상대 Load | Cut | Layer/Token TTFT p95 | Layer/Token SLO-goodput | Layer/Native ITL p99 | Layer/Native TPOT p95 | Token/Native ITL p99 | Token/Native TPOT p95 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cell in source['cells']:
        for c in cell['comparisons']:
            nr=c['native_relative'];ratio=c['Layer_over_Token']
            values=[ratio['TTFT_p95_ms'],ratio['SLO_qualified_output_tokens_per_second'],
                nr['Layer']['ITL_p99_ms'],nr['Layer']['TPOT_p95_ms'],nr['Token']['ITL_p99_ms'],nr['Token']['TPOT_p95_ms']]
            lines.append('| '+f"I{cell['input_bucket']} | {cell['output']} | {cell['QPS']:.6f} | {cell['request_count']} | {cell['reference_relative_load']:.3f} | {c['cut']} | "+
                ' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['','Reference 상대 Load는 Serial Pilot에서 추정한 Pipeline Rate를 기준으로 한다.',
        '실제 Sustainable Capacity의 비율이 아니다. 1.00도 실측 포화점을 보장하지 않는다.',
        'Decode 비율의 Point 값이 1.05 이하여도 독립 반복 Safety PASS로 처리하지 않는다.','',
        '### Queue와 실제 Drain','',
        '다음 표는 Native, Last8 Pair와 Hybrid4/Suffix≈L/8을 포함한다.',
        '다른 Cut의 Request별 구간도 전체 JSON에 보존했다.',
        'Queue Growth는 Arrival이 진행되는 구간의 뒤쪽 절반에서 계산한 OLS 기울기다.',
        '유한 Trace의 Burst와 Drain이 이 값에 영향을 줄 수 있다.','',
        '| Input / Output / QPS | Arm | TTFT p95(ms) | Achieved req/s | P Queue 평균/최대 | D Queue 평균/최대 | P Queue 기울기(req/s) | D Queue 기울기(req/s) |',
        '|---|---|---:|---:|---|---|---:|---:|']
    for cell in source['cells']:
        for row in cell['results']:
            if row['axis'] in ('Layer','Token') and row['cut']!=8:continue
            m=row['metrics'];q=row['coordinator_queue_observations']
            p,d=q['P_pipeline_waiting'],q['D_pipeline_waiting']
            label=row['axis']+(str(row['cut']) if row['axis']!='Native' else '')
            lines.append('| '+f"I{cell['input_bucket']}/O{cell['output']}/q{cell['QPS']:.6f} | {label} | {m['TTFT_p95_ms']:.3f} | {m['completed_requests_per_second']:.6f} | "+
                f"{p['mean']:.3f}/{p['max']} | {d['mean']:.3f}/{d['max']} | "+
                f"{p['second_half_OLS_slope_requests_per_second']:.6f} | {d['second_half_OLS_slope_requests_per_second']:.6f} |")
    lines += ['','Achieved req/s는 전체 측정 Window의 완료 Request 수를 사용한다.',
        'Offered QPS와 같은 값이 아니다. 모든 요청의 Drain 시간을 포함한다.',
        'Coordinator의 Pipeline Waiting은 Worker 내부 Queue나 GPU Idle과도 다르다.',
        'Source Credit과 Whole-request C0 D Guard가 이 Protocol의 일부다.',
        '따라서 이 결과를 C3 Continuous Native 또는 C3 Hybrid의 Capacity로 재사용하지 않는다.','',
        '실측, 실제 Geometry와 모든 Raw Name은 [전체 Pipeline Supplement](dense_ar_longbench400_2d_20261121/analysis/proper_pipeline_load_discovery_v1.json)에 있다.',
        '새 Source Capsule과 Dataset SHA는 [동결 설계](dense_ar_longbench400_2d_20261121/configs/proper_pipeline_load_discovery_frozen_v1.json)에 있다.','']
    path.write_text(prefix+'\n'+'\n'.join(lines),encoding='utf-8')
    print('PIPELINE_LOAD_REPORT',sum(len(r['results']) for r in source['cells']),'starts',flush=True)


if __name__=='__main__':
    main()
