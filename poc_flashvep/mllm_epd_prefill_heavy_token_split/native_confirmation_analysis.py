"""Recalculate independent-start Native results without pooling requests."""
import csv
import json
import time

import numpy as np

from .common import ROOT, digest, preserve, write
from .metrics import summarize


METRICS = ['TTFT_mean_ms', 'TTFT_p50_ms', 'TTFT_p95_ms', 'ITL_mean_ms', 'ITL_p99_ms',
           'TPOT_p95_ms', 'raw_output_tokens_per_second', 'SLO_goodput',
           'completed_requests_per_second', 'total_input_output_tokens_per_second']


def interval(values):
    values = np.asarray(values, dtype=float)
    result = dict(starts=len(values), start_values=values.tolist(), mean=float(values.mean()))
    if len(values) >= 3:
        rng = np.random.default_rng(2026112409)
        samples = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
        result['bootstrap95_start_mean_CI'] = np.percentile(samples, [2.5, 97.5]).tolist()
    else:
        result['bootstrap95_start_mean_CI'] = None
    return result


def main():
    preserve()
    rows = []
    source = []
    for start in [1, 2, 3]:
        tag = f'ec_NATIVE_C11_S{start}'
        record = json.loads((ROOT / 'analysis' / f'{tag}.json').read_text())
        assert record['status'] == 'COMPLETE'
        for role, gpu in [('E', 4), ('P', 5), ('D', 6)]:
            path = ROOT / 'state' / f'server_{tag}_{role}{gpu}.json'
            lease = json.loads(path.read_text())
            assert lease['finished'] and lease['ready']
            source.append(dict(start=start, role=role, gpu=gpu, GPU_info=lease['GPU_info'],
                source_archive_SHA256=lease['source_archive_SHA256'], lease_SHA256=digest(path),
                ready_time=lease['ready_time'], pid=lease['pid'], command=lease['command']))
        for mode in ['natural100', 'replay100']:
            run = f'{tag}_{mode}_q2_v0'
            metric = summarize(run)
            assert metric['completed_count'] == 100 and metric['failed_count'] == 0
            if mode == 'replay100':
                assert metric['total_output_tokens'] == 5269
            row = dict(start=start, mode=mode, qps=2, run=run,
                raw_SHA256=metric['raw_SHA256'], output_tokens=metric['total_output_tokens'],
                **{key: metric[key] for key in METRICS},
                **{key: 100 * value for key, value in metric['gate_attainment'].items()})
            rows.append(row)
    arrivals = [digest(ROOT / 'configs' / f'{row["run"]}_arrival.json') for row in rows]
    assert len(set(arrivals)) == 1, 'Request order or relative arrival trace differs'
    groups = {}
    for mode in ['natural100', 'replay100']:
        selected = [row for row in rows if row['mode'] == mode]
        groups[mode] = {key: interval([row[key] for row in selected])
                        for key in METRICS + ['TTFT_PASS', 'ITL_PASS', 'TPOT_PASS', 'JOINT_PASS']}
    path = ROOT / 'analysis/native_inline_independent_N3_v0.json'
    assert not path.exists(), 'Keep earlier analyses frozen'
    result = dict(time=time.time(), evidence='MEASURED_ONLINE runtime; primary quality PENDING',
        independent_starts=3, requests_per_start=100, QPS=2, rows=rows, start_mean_statistics=groups,
        arrival_trace_SHA256=arrivals[0], sources=source, quality_gate='PENDING',
        interpretation='Start-mean bootstrap, not request pooling. Natural and replay remain separate. '
            'There is no independent N3 CPU-transport or other-method comparator in this table. '
            'Absolute decode attainment and native-relative protection are separate gates. '
            'Three starts give limited confidence-interval resolution. No token-split success is established.')
    write(path, result)
    csv_path = ROOT / 'analysis/native_inline_independent_N3_v0.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Native EPD runtime confirmation — Main100, independent N3', '',
        '실제 E/P/D Online 실행을 독립 서버 초기화 3회로 확인했다. 전체 Task Quality Gate는 미확정이다.',
        'Token Split은 이 표에 없다. CPU 전송 경로 대비 독립 N3 개선율도 이 표로 판정하지 않는다.', '',
        'GPU는 4/5/6이다. 모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.',
        'Original BF16, BI0, Full28, B8192/S1024, KV42GiB, Block128을 유지했다.',
        'E는 원본 Vision Encoder를 실제 요청 도착 후 실행했다. E의 Token Budget은 114688이다.',
        'Early P Descriptor와 완료 후 File ACK를 사용했다. Producer는 D 완료와 ACK 전까지 Tensor를 보존했다.', '',
        '| Output contract | TTFT p95 ms | ITL p99 ms | TPOT p95 ms | Raw output tok/s | SLO-goodput tok/s | Joint % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode, label in [('natural100', 'Natural EOS'), ('replay100', 'Equal-output replay')]:
        value = groups[mode]
        keys = ['TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'raw_output_tokens_per_second', 'SLO_goodput', 'JOINT_PASS']
        lines.append('| ' + label + ' | ' + ' | '.join(f'{value[key]["mean"]:.3f}' for key in keys) + ' |')
    lines += ['', '표의 값은 Start별 Metric의 평균이다. 300개 요청을 Pooling한 Percentile이 아니다.',
        '각 Start의 수치와 Start-mean Bootstrap 95% CI는 대응 JSON에 보존했다.',
        'Natural EOS와 Replay의 평균은 서로 합치지 않았다. Replay는 각 Start에서 5269 Output Tokens다.',
        '모든 Start의 상대 Arrival Trace와 요청 순서는 같았다. 절대 실행 시각은 다르다.', '',
        'TTFT SLO는 5000ms다. 모든 Client ITL은 34ms 이하여야 한다. Request TPOT는 33ms/token 이하여야 한다.',
        'Joint SLO는 세 조건의 AND다. Goodput은 Joint 통과 요청의 Output Tokens를 전체 측정 기간으로 나눈 값이다.',
        '측정 기간은 첫 예정 Arrival부터 마지막 Drain까지다. Finite-cohort Throughput은 Sustainable Capacity가 아니다.', '',
        'CUPTI Profile은 독립 성능 측정과 분리했다. C11 S2 Profile은 5% 계측 기준을 통과하지 못했다.',
        '따라서 해당 Kernel Busy와 Idle 수치는 진단용이다. Decode-safe Headroom은 아직 미검증이다.',
        'MMLongBench-Doc의 공식 GPT 답변 추출은 미실행이다. Raw Answer 점수를 완전한 공식 품질 평가로 사용하지 않는다.', '',
        '대응 CSV: `analysis/native_inline_independent_N3_v0.csv`.',
        '대응 JSON: `analysis/native_inline_independent_N3_v0.json`.', '']
    report = ROOT.parent / 'mllm_epd_native_runtime_confirmation_v1.md'
    assert not report.exists()
    report.write_text('\n'.join(lines))
    preserve()
    print(json.dumps({mode: {key: value[key]['mean'] for key in
          ['TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'JOINT_PASS']} for mode, value in groups.items()}))


if __name__ == '__main__':
    main()
