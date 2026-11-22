"""Generate the complete measured C0 matrix without omitting invalid cells."""
import json
from .common import ROOT,REPO,write


def interval(test):
    if test is None:
        return 'NOT_RUN'
    return f"{test['estimate']:.4f} [{test['CI95'][0]:.4f}, {test['CI95'][1]:.4f}]"


def main():
    source=json.loads((ROOT/'analysis/axis_matrix_heldout_confirmation_r2.json').read_text())
    work=[]
    for cell in source['cells']:
        for start in cell['results']:
            actual={}
            for axis in ('Layer','Token'):
                roles={}
                for role in ('prefill','decode'):
                    events=json.loads((ROOT/'raw'/start[axis]/f'axis_events_{role}.json').read_text())['rows']
                    parts={r['request_id']:r for r in events if r['event']=='PARTITION'
                        and r['request_id'].startswith('q') and r['request_id'][1:].isdigit()}
                    assert len(parts)==16
                    roles[role]={k:sum(r[k] for r in parts.values()) for k in ('P_rows','D_rows')}
                assert roles['prefill']==roles['decode']
                actual[axis]=roles['prefill']
            assert actual['Layer']==actual['Token']
            work.append(dict(input_bucket=cell['input_bucket'],output=cell['output'],cut=cell['cut'],
                jobs={a:start[a] for a in ('Layer','Token')},actual_aggregate_rows=actual,signed_error_rows=0))
    write(ROOT/'analysis/axis_matrix_exact_work_audit_v1.json',dict(passed=True,paired_cell_start_blocks=len(work),
        rows=work,evidence='MEASURED_EXECUTION_COVERAGE',
        scope='Audit actual P/D partition records for every Layer/Token pair. Both aggregate Transformer P/D rows match exactly. '
            'Attention,payloads and kernel times are not equal. Native controls are reused across cuts. '
            'The number of cell/start blocks is not the number of independent Native starts.'))
    lines=['# LongBench400: Input × Output × Cut Winner Map','',
        '증거 수준은 `MEASURED_ONLINE`이다. 이 문서의 Primary Map은 C0 Batch1/1/1 Mechanism Control이다.',
        'C3 Continuous Serving의 성능으로 해석하지 않는다. Natural EOS Main400과도 구분한다.','',
        '## 실행 계약','',
        '- Qwen3-8B Revision: `b968826d9c46dd6066d109eabc6255188de91218`.','- BI0, Original BF16, Exact Causal Attention, Full36 Decode다.',
        '- B8192/S1024, KV42GiB/GPU, KV Block128, Source Credit4다.',
        '- 실제 P Prefill/D Suffix/D Decode Batch는 1/1/1이다. D는 Whole-request Admission을 사용한다.',
        '- 각 Start는 원문 Prompt 16개를 사용한다. Output은 정확히 8/16/64/128 Tokens로 통제한다.',
        '- Primary Cut은 Last2/4/8/12다. Token Suffix는 인접 16-Token 길이로 Aggregate Rows를 맞춘다.',
        '- Input별 공통 QPS는 0.910229/0.871211/0.822364/0.763308이다.',
        '- Source Capsule SHA256은 `08918a2e25f957ede7387297119b491ee21fd6ea26cd380ceb6d7e328da0d772`다.','',
        '각 Cell은 독립 Start 3회다. 사전에 지정한 Core Cell 3개는 5회다.',
        f'실제 P/D Partition 기록 {len(work)}개 Paired Cell/Start Block에서 양축의 Aggregate P/D Rows가 정확히 같았다.',
        'GEMM Rows가 같아도 Attention Pair와 Payload Bytes는 다르다. Native의 한 Query 재계산은 별도 비용이다.',
        'CI는 Start별 Paired Bootstrap을 완전 열거했다. n3은 27개, n5는 3125개 재표본을 사용한다.',
        'CI는 Pointwise이며 다중 비교 보정을 하지 않았다. 64개 Cell 전체의 Familywise 오류를 제어하지 않는다.',
        'Request나 Token을 독립 Server Start로 취급하지 않았다.','',
        '## 전체 판정','',
        '| 판정 | Cell 수 |','|---|---:|']
    for name,count in source['counts'].items():
        lines.append(f'| {name} | {count} |')
    lines+=['',
        '`LAYER_CONFIRMED`와 `TOKEN_CONFIRMED`는 TTFT 방향과 Winner의 Decode Gate를 함께 확인한 판정이다.',
        'Native 대비 ITL p99와 TPOT p95 CI 상한은 모두 1.05 이하여야 한다. Absolute Decode Gate도 별도로 검사한다.',
        '어느 축이 다른 축을 이겼다는 사실은 Native보다 좋은 Serving 성과를 뜻하지 않는다.',
        '`INVALID` Cell 5개도 아래 표에 남긴다. 실패한 Decode Gate를 제거하거나 Threshold를 바꾸지 않았다.','',
        '## 64개 Cell','',
        '각 비율은 Layer/Token이다. 대괄호는 95% CI다. Goodput은 Frozen Joint SLO-qualified Output Tokens/s다.','']
    for output in (8,16,64,128):
        lines += [f'### Controlled Output{output}','',
            '| Input Bucket | Cut | Starts | Actual GPU Pair | TTFT p95 L/T [CI] | SLO-goodput L/T [CI] | 판정 |',
            '|---|---:|---:|---|---|---|---|']
        for cell in sorted((r for r in source['cells'] if r['output']==output),key=lambda r:(r['input_bucket'],r['cut'])):
            pairs=set()
            for start in cell['results']:
                configs=[json.loads((ROOT/'raw'/start[a]/'configuration.json').read_text()) for a in ('Native','Layer','Token')]
                assert all(configs[0]['gpus']==c['gpus'] for c in configs)
                pairs.add(tuple(configs[0]['gpus']))
            pairtext=', '.join('/'.join(map(str,p)) for p in sorted(pairs))
            bucket=('3K–6K','6K–10K','10K–20K','20K–36K')[cell['input_bucket']]
            lines.append(f"| {bucket} | {cell['cut']} | {cell['completed_paired_starts']} | {pairtext} | "
                f"{interval(cell.get('TTFT_p95_Layer_over_Token'))} | {interval(cell.get('goodput_Layer_over_Token'))} | {cell['status']} |")
        lines.append('')
    lines += ['## 해석 범위','',
        '작은 Cut이면 항상 Layer가 이긴다는 가설은 성립하지 않았다.',
        'Input, Controlled Output, Cut과 Queue Regime에 따라 Winner가 달랐다.',
        '같은 Input에서 Output만 바뀌어도 D Residency가 달라진다. 다만 모든 구간의 인과성을 이 표 하나로 확정하지 않는다.','',
        '원래 Discovery의 상대 부하는 Serial P+D+Decode Reference로 정했다.',
        '이 값은 실제 Pipeline Bottleneck Capacity가 아니다. 같은 절대 QPS라도 Output128과 Output8의 부하 상태는 다르다.',
        '따라서 전체 64개 Cell을 Low-load라고 부르지 않는다.',
        '별도 Proper-pipeline-load Supplement는 Bottleneck Rate 추정치 주변을 검사한다. 단일 Start는 Winner CI가 아니다.','',
        '## Native-relative Serving 비교','',
        'I0/Output8/Last2의 Layer/Token TTFT 비율은 0.9437 [0.9241,0.9653]이었다.',
        '하지만 같은 Cell의 Layer/Native 비율은 1.0029 [0.9696,1.0324]였다.',
        '이는 Axis Win이지만 Native 대비 TTFT Win은 아니다.','',
        'I3/Output16/Last8의 Layer/Token 비율은 1.1405 [1.1074,1.1774]였다.',
        '이 Cell에서는 Token이 우세했다. 두 축 모두 Native 대비 TTFT와 SLO-goodput 이득을 보였다.',
        '이 결과도 16개 요청의 Finite-cohort 측정이다. Sustainable Capacity로 표현하지 않는다.','',
        '## 예측 한계','',
        '기존 Serial-calibrated State-updating Replay는 명확한 Winner 38개 중 32개를 예측했다.',
        'Accuracy는 32/38, 84.21%였다. 유효한 59개 Cell의 평균 TTFT Regret은 0.841%였다.',
        'p95 Regret은 5.966%였고 최대는 11.530%였다. 모든 명확한 오분류는 실제 Layer Winner를 Token으로 예측한 경우였다.',
        '이 Model은 전체 Controlled Arrival Trace와 Output 길이를 안다. 검증된 1–2개 Online Knob으로 해석하지 않는다.','',
        '## 재현 자료','',
        '- [전체 Cell과 Native-relative Metric](dense_ar_longbench400_2d_20261121/analysis/axis_matrix_heldout_confirmation_r2.json).',
        '- [Original Discovery와 Pipeline Load 구분](dense_ar_longbench400_pipeline_load_reference_audit.md).',
        '- [Held-out Winner Prediction과 Regret](dense_ar_longbench400_2d_20261121/analysis/c0_axis_winner_prediction_audit_v1.json).',
        '- [TTFT Ratio Figure](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_ttft_ratio.png).',
        '- [SLO-goodput Ratio Figure](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_goodput_ratio.png).',
        '',
        'C3 Best-static Confirmation과 Genuine Hybrid의 비교는 별도 보고서에 기록한다.',
        '미실행 C3나 Dynamic 결과로 이 C0 Map을 대체하지 않는다.','']
    destination=REPO/'poc_flashvep/reports/dense_ar_longbench400_input_output_axis_winner_map.md'
    destination.write_text('\n'.join(lines))
    print('WROTE_COMPLETE_AXIS_MAP_REPORT',len(source['cells']),flush=True)


if __name__=='__main__':
    main()
