"""Recalculate all three common-policy starts without hiding Natural variance."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from .common import ROOT, digest, preserve, write
from .metrics import gates
from .native_common_pilot_report import arrival_contract_SHA256
from .native_confirmation_analysis import interval

TAGS = ['ec_COMMON_C32', 'ec_COMMON_C32R1', 'ec_COMMON_C32R2']
KEYS = ['TTFT_mean_ms', 'TTFT_p50_ms', 'TTFT_p95_ms', 'ITL_mean_ms', 'ITL_p99_ms',
    'TPOT_p95_ms', 'raw_output_tokens_per_second', 'SLO_goodput']
CATEGORIES = ['JOINT_PASS', 'TTFT_FAIL', 'ITL_FAIL', 'TPOT_FAIL', 'TTFT+ITL_FAIL',
    'TTFT+TPOT_FAIL', 'ITL+TPOT_FAIL', 'TTFT+ITL+TPOT_FAIL']


def main():
    preserve()
    completion = ROOT / 'state/common_native_confirmation_C32_complete.json'
    assert json.loads(completion.read_text())['status'] == 'COMPLETE'
    rows = []
    failures = []
    lengths = []
    raw_sources = []
    for start, tag in enumerate(TAGS, 1):
        for role,gpu in [('E',4),('P',5),('D',6)]:
            lease = json.loads((ROOT/'state'/f'server_{tag}_{role}{gpu}.json').read_text())
            assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()
        for mode,suffix in [('Natural EOS','natural_q2'),('Equal-output replay','replay_q2')]:
            run = f'{tag}_{suffix}_v0'
            path = ROOT / 'raw' / f'{run}_requests.jsonl'
            requests = [json.loads(line) for line in path.read_text().splitlines()]
            metrics = json.loads((ROOT/'analysis'/f'{run}_metrics.json').read_text())
            timing = json.loads((ROOT/'state'/f'{run}_complete.json').read_text())
            assert len(requests)==100 and all(r['status']=='MEASURED_ONLINE' for r in requests)
            assert digest(path)==metrics['raw_SHA256']==timing['raw_SHA256']
            output_tokens = sum(r['output_tokens'] for r in requests)
            assert output_tokens==metrics['total_output_tokens']
            assert all(r['usage']['completion_tokens']==len(r['token_ids'])==r['output_tokens'] for r in requests)
            for key,values in [('TTFT_p95_ms',[r['TTFT_ms'] for r in requests]),
                ('ITL_p99_ms',[v for r in requests for v in r['ITL_ms']]),
                ('TPOT_p95_ms',[r['TPOT_ms'] for r in requests])]:
                expected=float(np.percentile(values,99 if key=='ITL_p99_ms' else 95))
                assert expected==metrics[key]
            annotated=[(r,gates(r)) for r in requests]
            qualified=sum(r['output_tokens'] for r,g in annotated if g['JOINT_PASS'])
            assert qualified/timing['duration_seconds']==metrics['SLO_goodput']
            if suffix=='replay_q2':assert output_tokens==5269
            trace = ROOT/'configs'/f'{run}_arrival.json'
            row = dict(start=start,run=run,mode=mode,output_tokens=output_tokens,
                duration_seconds=metrics['duration_seconds'],**{key:metrics[key] for key in KEYS},
                joint_percent=100*metrics['gate_attainment']['JOINT_PASS'],
                ITL_pass_percent=100*metrics['gate_attainment']['ITL_PASS'],
                TPOT_pass_percent=100*metrics['gate_attainment']['TPOT_PASS'],
                actual_arrival_file_SHA256=digest(trace),arrival_contract_SHA256=arrival_contract_SHA256(trace),
                raw_SHA256=digest(path),evidence='MEASURED_ONLINE runtime; task quality PENDING')
            rows.append(row)
            for category in CATEGORIES:
                selected=[r for r,g in annotated if g['category']==category]
                failures.append(dict(start=start,mode=mode,category=category,requests=len(selected),
                    request_denominator=100,output_tokens=sum(r['output_tokens'] for r in selected),
                    total_output_tokens=output_tokens,duration_seconds=timing['duration_seconds']))
            lengths += [dict(start=start,mode=mode,request_id=r['request_id'],source=r['source'],
                input_tokens=r['usage']['prompt_tokens'],output_tokens=r['output_tokens'],
                cap_hit=r['output_cap_hit'],TTFT_ms=r['TTFT_ms'],**g) for r,g in annotated]
            raw_sources.append(dict(run=run,path=str(path),SHA256=digest(path)))
    assert len({row['arrival_contract_SHA256'] for row in rows})==1
    stats = {mode:{key:interval([r[key] for r in rows if r['mode']==mode])
        for key in KEYS+['joint_percent','ITL_pass_percent','TPOT_pass_percent']}
        for mode in ['Natural EOS','Equal-output replay']}
    for name,data in [('starts',rows),('failure_categories',failures),('request_lengths',lengths)]:
        path=ROOT/'analysis'/f'common_native_C32_N3_{name}_v0.csv'
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report_data=ROOT/'analysis/common_native_C32_N3_summary_v0.json'
    write(report_data,dict(time=time.time(),status='RUNTIME_N3_RECALCULATED',independent_starts=3,
        rows=rows,start_mean_statistics=stats,raw_sources=raw_sources,raw_recalculation='PASS',
        task_quality='PENDING',stock_NativeStrong='Unchanged. Historical N3 remains a separate numerical policy.',
        caveat='Start-mean bootstrap, not request pooling. Natural work varies across starts. '
            'No paired stock-versus-common policy CI or speedup is established.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for mode in ['Natural EOS','Equal-output replay']:
        selected=[r for r in rows if r['mode']==mode]
        axes[0].plot([r['start'] for r in selected],[r['TTFT_p95_ms'] for r in selected],marker='o',label=mode)
        axes[1].plot([r['start'] for r in selected],[r['SLO_goodput'] for r in selected],marker='o',label=mode)
    axes[0].axhline(5000,color='gray',linestyle='--',label='TTFT SLO')
    axes[0].set_ylabel('TTFT p95 (ms)');axes[1].set_ylabel('SLO-qualified output tokens/s')
    for axis in axes:axis.set_xlabel('Independent initialization');axis.set_xticks([1,2,3]);axis.grid(alpha=.2);axis.legend()
    figure.suptitle('Common compiled Native | Main100 | QPS2 | Quality pending')
    figure.tight_layout()
    for extension in ['png','pdf']:figure.savefig(ROOT/'figures'/f'common_native_C32_N3_start_variation_v0.{extension}',dpi=180)
    plt.close(figure)
    figure_csv=ROOT/'figures/common_native_C32_N3_start_variation_v0.csv'
    with figure_csv.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Common Compiled Native EPD — 독립 N3 검증','',
        '공통 Compiled 설정의 Native EPD를 독립 서버 초기화 3회로 실행했다.',
        'Main100 Natural EOS와 Equal-output Replay를 별도로 재계산했다.',
        '증거 수준은 `MEASURED_ONLINE` Runtime이다. 전체 Task Quality는 `PENDING`이다.','',
        '## 계약과 재계산','',
        'Qwen2.5-VL-7B, Original BF16, BI0, 원본 Vision Encoder와 Processor를 유지했다.',
        'Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다. GPU4/5/6은 E/P/D다.',
        'B8192/S1024, KV42GiB, Block128, Prefix Cache Off, Natural Cap2048을 유지했다.',
        'P/D 공통 Compiler 설정은 `mllm_epd_compiled_token_correctness_v1.md`에 있다.',
        'Native에는 Token Adapter, 추가 Source Credit 또는 D Cap을 적용하지 않았다.',
        '기존 Stock NativeStrong과 그 독립 N3 결과는 변경하지 않았다.',
        '각 Start에서 N1 및 순차 N8 상태 검사를 다시 통과한 뒤 Timing을 시작했다.',
        'Raw Token 수, Percentile과 Goodput 분자를 다시 계산했다. 저장된 Metric과 일치했다.',
        '모든 요청 순서와 상대 Arrival은 같다. QPS JSON의 2와 2.0만 정규화해 Contract Hash를 비교했다.',
        '원래 Arrival 파일은 수정하지 않았다. 모든 Start를 보존했다.','',
        '| Output | Start | Output Tokens | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Goodput tok/s | Joint % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+r['mode']+' | '+str(r['start'])+' | '+str(r['output_tokens'])+' | '+
            ' | '.join(f'{r[k]:.3f}' for k in ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput','joint_percent'])+' |')
    lines+=['','## Start-mean과 변동성','',
        '| Output | TTFT p95 Start-mean ms | Start-mean 95% CI ms | ITL p99 Start-mean ms | Goodput Start-mean tok/s |',
        '|---|---:|---|---:|---:|']
    for mode,value in stats.items():
        ci=value['TTFT_p95_ms']['bootstrap95_start_mean_CI']
        lines.append(f'| {mode} | {value["TTFT_p95_ms"]["mean"]:.3f} | '
            f'[{ci[0]:.3f}, {ci[1]:.3f}] | {value["ITL_p99_ms"]["mean"]:.3f} | {value["SLO_goodput"]["mean"]:.3f} |')
    lines+=['',
        'CI는 독립 Start 단위 Bootstrap이다. 300개 요청을 독립 서버 시작으로 취급하지 않았다.',
        'N3의 CI 해상도는 제한적이다. Pooling한 p95와 Start별 p95 평균도 구분했다.',
        '세 번째 Natural 실행의 TTFT p95는 6266.307ms였다. 첫 실행의 4493.311ms보다 높았다.',
        '해당 실행을 제외하거나 유리한 결과가 나올 때까지 반복하지 않았다.',
        'Natural Output은 Start별로 5113/5396/5256 Tokens였다. 실제 Batch와 BI0 Numerical Path에 따른 차이가 있다.',
        '이 차이와 Queue 변동의 개별 인과적 기여율은 이 분석에서 확정하지 않았다.',
        'Replay는 모든 Start에서 5269 Tokens다. Answer String의 동일성을 뜻하지 않는다.',
        'Replay TTFT p95도 실제 관측 범위 그대로 보고했다. Natural과 합쳐 평균하지 않았다.','',
        '## SLO와 품질','',
        'TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms/token 계약을 유지했다.',
        'Joint는 세 조건의 AND다. Goodput 분자는 Joint 통과 요청의 Output Tokens다.',
        '분모는 첫 예정 Arrival부터 Drain까지다. Finite-cohort Throughput은 Sustainable Capacity가 아니다.',
        '실패 8개 조합의 Request Count와 Output Token Weight를 각 Start/Output 계약별 CSV로 보존했다.',
        'DocVQA Raw ANLS Floor와 MMLongBench-Doc 공식 답변 추출의 미실행은 여전히 남아 있다.',
        '이 표로 전체 품질 PASS 또는 Token Serving 성공을 선언하지 않는다.',
        'Raw/Derived 대응은 `analysis/common_native_C32_N3_summary_v0.json`에 있다.',
        'CSV는 같은 Prefix의 `starts`, `failure_categories`, `request_lengths` 파일이다.',
        'Figure는 `figures/common_native_C32_N3_start_variation_v0.{png,pdf,csv}`다.','']
    report=ROOT.parent/'mllm_epd_common_compiled_native_confirmation_v1.md';assert not report.exists()
    report.write_text('\n'.join(lines));preserve()
    print('COMMON_NATIVE_N3_RECALCULATED',flush=True)


if __name__=='__main__':main()
