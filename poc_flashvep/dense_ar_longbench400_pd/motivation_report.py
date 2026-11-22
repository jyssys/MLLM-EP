"""Write the original continuous Agg/Native sweep with all measured points."""
import json
from .common import ROOT,REPO


def number(value):
    return f'{value:,.3f}'


def main():
    source=json.loads((ROOT/'analysis/motivation400_confirmed_metric_table_v1.json').read_text())
    fields={r['metric']:r for r in source['figures']}
    keys=sorted({(r['QPS'],r['method']) for r in fields['TTFT_p95_ms']['points']})
    values={field:{(r['QPS'],r['method']):r for r in fig['points']} for field,fig in fields.items()}
    lines=['# LongBench400: Agg DP2와 Native PD Motivation','',
        '증거 수준은 `MEASURED_ONLINE`이다. 이 보고서는 개선 전 Installed Native와 Agg의 Continuous Sweep을 다룬다.',
        '새 Decode 복원 정책이나 Layer/Token/Hybrid 결과를 이 표에 섞지 않는다.','',
        '## 계약','',
        'Qwen3-8B Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.',
        'BI0, Original BF16, Exact Causal Attention, Greedy, Thinking Disabled와 Natural EOS를 사용한다.',
        'Dataset은 LongBench-Balanced400-v1이다. Output Cap은 2048이다. Prefix Cache는 사용하지 않는다.',
        '모든 Worker는 B8192/S1024와 KV42GiB를 명시한다. KV Block은 128이다.',
        '이 설정이 전역 최적이라고 주장하지 않는다. 실제 Batch와 Chunked Prefill은 각 Worker Log에 있다.','',
        'Agg는 GPU0/1의 Full-model Replica 두 개를 사용한다. Coordinator는 Round-robin으로 배정한다.',
        'Native는 GPU0=P, GPU1=D다. Full36 Prefill과 Full36 Decode를 사용한다.',
        '두 방법은 Installed Default Compile과 정상 Continuous Scheduler를 사용한다.',
        'Native에 Layer/Token Adapter나 Whole-request C0 Guard를 넣지 않는다.','',
        'Installed Python Source는 Wheel RECORD와 대조했다. 이는 Canonical Upstream Wheel과의 독립 동등성 증명은 아니다.',
        '따라서 정확한 범위는 감사한 Installed Runtime의 Original Model/Connector/Default Scheduler 경로다.',
        '실제 Source, Version과 Model Hash를 Evidence에 보존한다.','',
        '같은 QPS/Seed의 두 방법은 Prompt, Request 순서와 Open-loop Poisson Arrival Trace가 같다.',
        'BI0의 Batch별 Numerical Variation 때문에 실제 Natural Output Count는 완전히 같지 않을 수 있다.',
        '모든 Start는 선언한 Task Quality Gate를 통과했다. 이를 모든 Greedy Sequence의 Bitwise 동일성이라고 부르지 않는다.','',
        '## 전체 실측점','',
        '| Offered QPS | Method | 독립 Starts | TTFT p95 | ITL p99 | TPOT p95 | Raw Output Tokens/s | SLO-goodput |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for q,method in keys:
        p=values['TTFT_p95_ms'][q,method]
        v={f:rows[q,method]['mean'] for f,rows in values.items()}
        lines.append(f'| {q:g} | {method} | {p["independent_starts"]} | {number(v["TTFT_p95_ms"])}ms | '
            f'{number(v["ITL_p99_ms"])}ms | {number(v["TPOT_p95_ms"])}ms/token | '
            f'{number(v["generated_output_tokens_per_second"])} | {number(v["SLO_qualified_output_tokens_per_second"])} |')
    lines += ['',
        '표의 값은 Start별 Metric의 평균이다. 1200개 Request를 합친 하나의 p95가 아니다.',
        'QPS1.8/2.8/4.5는 독립 Start 3회다. 다른 QPS는 1회다.',
        'n3 Figure의 95% CI는 Start-mean Bootstrap 27개 재표본을 완전 열거했다.',
        'n1에는 CI를 표시하지 않는다. Pointwise CI이며 다중 비교 보정을 하지 않았다.','',
        '## Trade-off 해석','',
        'QPS1.8에서 Agg의 TTFT p95는 약 1554ms, Native는 약 4886ms였다.',
        '같은 조건에서 ITL p99는 Agg 약 278ms, Native 약 15.9ms였다.',
        'QPS2.8에서는 Agg TTFT p95가 약 2109ms, Native가 약 36683ms였다.',
        'Native는 Decode Tail을 상대적으로 보호했지만 높은 부하에서 TTFT가 악화됐다.','',
        'Raw Throughput과 SLO-goodput은 다르다. Agg의 높은 Raw Throughput이 모든 Decode SLO를 충족한다는 뜻은 아니다.',
        'SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms다.',
        '그림의 ITL p99와 Every-ITL SLO도 서로 다른 통계다.','',
        '모든 요청을 Drain까지 측정했다. 이 값은 Finite-cohort Throughput이다.',
        'Queue가 증가하는 조건을 Sustainable Capacity라고 부르지 않는다.',
        'Offered QPS, Actual Arrival QPS, Dispatch QPS와 Completed Requests/s는 Raw Metric에 따로 있다.','',
        '## Figure','',
        '각 Figure는 PNG와 PDF를 제공한다. X축은 실제 측정한 공통 QPS다.',
        '선은 측정점 연결용이다. 누락된 점을 생성하거나 실측으로 보간하지 않았다.','',
        '- [TTFT p95 PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_ttft_p95.png) · [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_ttft_p95.pdf).',
        '- [ITL p99 PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_itl_p99.png) · [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_itl_p99.pdf).',
        '- [TPOT p95 PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_tpot_p95.png) · [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_tpot_p95.pdf).',
        '- [Output Tokens/s PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_output_tokens.png) · [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_output_tokens.pdf).',
        '- [SLO-goodput PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_slo_goodput.png) · [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_slo_goodput.pdf).','',
        '## Prefill-heavy 판정의 한계','',
        'TTFT 증가만으로 P GPU Compute Saturation을 확정하지 않는다.',
        '원래 Native GPU Capture의 Observer Gate가 실패한 결과는 Diagnostic으로 남긴다.',
        '복원된 공통 정책 Native의 별도 Clean/Profile/Clean 3회 검사는 다른 Runtime 계약이다.',
        '해당 검사의 Kernel Busy와 Queue를 원래 9-point Sweep의 실측값으로 채우지 않는다.',
        'D Kernel Inactivity도 예측 가능한 Decode-safe Executable Window와 다르다.','',
        '## Provenance','',
        'Dataset SHA256은 `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`이다.',
        '이 Sweep의 Source Capsule SHA256은 `edb0da2fa8544a2fea0e1f48b7651bada38e1422b566a42f040b6baa3367e207`이다.',
        '이 결과는 DuetServe-style 방법론을 새 Workload에 적용한 Benchmark다. DuetServe Figure2의 수치 재현이 아니다.',
        '근거: [Metric과 Start별 Provenance](dense_ar_longbench400_2d_20261121/analysis/motivation400_confirmed_metric_table_v1.json).',
        'Source/Command/GPU UUID/Actual Batch/Task Metric은 각 실행의 Raw Folder와 Worker Log에 있다.','']
    path=REPO/'poc_flashvep/reports/dense_ar_longbench400_motivation_agg_native.md'
    path.write_text('\n'.join(lines))
    print('WROTE_MOTIVATION_REPORT',len(keys),'method/QPS points',flush=True)


if __name__=='__main__':
    main()
