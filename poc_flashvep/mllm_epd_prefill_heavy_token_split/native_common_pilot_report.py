"""Describe the closed N1 common Native pilot, not a new serving winner."""
import csv
import hashlib
import json
import time

from .common import ROOT, digest, preserve, write


def arrival_contract_SHA256(path):
    record = json.loads(path.read_text())
    assert type(record['QPS']) in (int, float)
    record['QPS'] = float(record['QPS'])
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def same_arrival_contract(left, right):
    return arrival_contract_SHA256(left) == arrival_contract_SHA256(right)


def main():
    preserve()
    tag = 'ec_COMMON_C32'
    state_path = ROOT / 'analysis' / f'{tag}_correctness.json'
    state = json.loads(state_path.read_text())
    assert state['passed'] and len(state['rows']) == 9
    native = json.loads((ROOT / 'analysis/native_inline_independent_N3_v0.json').read_text())
    pilot_path = ROOT / 'analysis' / f'{tag}_pilot_summary.json'
    pilot = json.loads(pilot_path.read_text())
    assert len(pilot['results']) == 3
    quality_path = ROOT / 'analysis/common_native_C32_task_scores_v0.json'
    quality = json.loads(quality_path.read_text())
    rows = []
    arrivals = []
    metrics = ['TTFT_mean_ms', 'TTFT_p50_ms', 'TTFT_p95_ms', 'ITL_mean_ms', 'ITL_p99_ms',
        'TPOT_p95_ms', 'raw_output_tokens_per_second', 'SLO_goodput']
    for mode, suffix in [('natural100', 'natural_q2'), ('replay100', 'replay_q2')]:
        run = f'{tag}_{suffix}_v0'
        current = next(r for r in pilot['results'] if r['run'] == run)
        assert current['primary_execution_valid']
        for policy, starts in [('Stock NativeStrong', 3), ('Common compiled Native', 1)]:
            value = native['start_mean_statistics'][mode] if starts == 3 else current
            row = dict(mode=mode, method=policy, independent_starts=starts,
                **{key: value[key]['mean'] if starts == 3 else value[key] for key in metrics},
                joint_percent=value['JOINT_PASS']['mean'] if starts == 3 else 100 * value['gate_attainment']['JOINT_PASS'],
                output_tokens='See per-start rows' if starts == 3 else current['total_output_tokens'],
                quality='PENDING', evidence='MEASURED_ONLINE runtime; not a paired policy speedup')
            rows.append(row)
        base_trace = ROOT / 'configs' / f'ec_NATIVE_C11_S1_{mode}_q2_v0_arrival.json'
        common_trace = ROOT / 'configs' / f'{run}_arrival.json'
        assert same_arrival_contract(base_trace, common_trace)
        arrivals.append(dict(mode=mode, stock_file_SHA256=digest(base_trace),
            common_file_SHA256=digest(common_trace), canonical_contract_SHA256=arrival_contract_SHA256(base_trace),
            note='Normalize only the QPS JSON number2 versus2.0. Original request IDs, offsets and seeds remain exact.'))
    publication = json.loads((ROOT / 'state/publication_commit_checkpoint_v4.json').read_text())
    timing = {}
    for suffix in ['natural_q2', 'replay_q2']:
        record = json.loads((ROOT / 'state' / f'{tag}_{suffix}_v0_complete.json').read_text())
        assert publication['time'] < record['wall_start'], 'Do not hide concurrent publication work in measured trials'
        timing[suffix] = dict(wall_start=record['wall_start'], wall_end=record['wall_end'])
    csv_path = ROOT / 'analysis/common_native_C32_runtime_pilot_v0.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(ROOT / 'analysis/common_native_C32_pilot_report_audit_v0.json', dict(time=time.time(),
        state_SHA256=digest(state_path), pilot_SHA256=digest(pilot_path), quality_SHA256=digest(quality_path),
        CSV_SHA256=digest(csv_path), publication_end_before_both_measured_runs=True,
        measured_wall_intervals=timing, arrivals=arrivals, common_policy_starts=1, stock_policy_starts=3,
        paired_CI='NOT_AVAILABLE', task_quality='PENDING', token_split='NOT_IN_THIS_TABLE'))
    lines = ['# Common Compiled Native EPD — 첫 독립 실행', '',
        '새 공통 Compiler 설정의 Native EPD 상태 검사와 Main100 실행이 완료됐다.',
        '상태 검사는 `CORRECTNESS_ONLY`다. 아래 성능은 `MEASURED_ONLINE` Runtime 결과다.',
        'Task Quality Gate는 미확정이다. Token Split 성능은 이 표에 없다.', '',
        '## 공정한 기준선 분리', '',
        '기존 Stock NativeStrong은 변경하지 않았다. Stock 수치는 이전 독립 N3의 Start-mean이다.',
        '새 Common Native는 독립 N1이다. 이 표에서 Paired CI나 정책 우위를 판정하지 않는다.',
        'Main100 Manifest, 요청 순서, QPS2.0과 상대 Poisson Arrival Trace는 동일하다.',
        'Arrival 파일은 QPS의 JSON 표기 2와 2.0만 다르다. 모든 요청 ID, Offset과 Seed는 정확히 같다.',
        '원래 파일은 수정하지 않았다. 두 Raw SHA와 QPS만 정규화한 Contract SHA를 함께 보존했다.',
        'Natural EOS와 Replay는 별도 계약이다. 두 계약의 Metric을 합쳐 평균하지 않는다.',
        '모델/Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.',
        'GPU4/5/6, Original BF16, BI0, Full28, B8192/S1024, KV42GiB, Block128이다.',
        'Common P/D는 Deterministic Reduction 설정을 공유한다. E는 기존 경로를 유지한다.',
        'Native에는 Token Split Adapter, 추가 D Cap 또는 Source Credit 제한을 적용하지 않았다.', '',
        '## N1/N8 상태 검사', '',
        'N1 뒤에 8개 요청을 순서대로 검사했다. 총 9회이며 고유 요청은 8개다.',
        '같은 공통 설정의 원본 E+P Full Teacher와 Native EPD를 비교했다.',
        'P의 Full Prompt KV는 Teacher와 28/28 Layer Strict Match였다.',
        'D로 전달한 첫 L−1개의 KV Row도 28/28 Layer Strict Match였다.',
        'Native D는 마지막 Prompt Query 1개를 다시 계산한다. 이 Row는 별도로 검사했다.',
        '전체 Prompt 평균으로 마지막 Row의 오차를 희석하지 않았다.',
        '고정한 기준은 Relative L2≤0.02와 Cosine≥0.999다. 각 Case가 통과했다.',
        '전체 Prompt KV Strict Match는 각 Case에서 1/28 Layer였다. 나머지 차이는 마지막 Row에 있었다.',
        '원본 Prompt IDs, MRoPE, 마지막 Query Position과 Embedding은 일치했다.',
        '9회 모두 Teacher와 Native의 전체 Greedy Token이 일치했다.',
        '다른 Batch/Query Geometry의 BI0 Bitwise Invariance를 일반적으로 보장하지는 않는다.', '',
        '| Request | Natural Output | 마지막 Query 최대 Relative L2 |',
        '|---|---:|---:|']
    for row in state['rows']:
        lines.append(f'| {row["request_id"]} | {row["native_output_tokens"]} | '
            f'{max(r["last_query"]["relative_L2"] for r in row["layers"]):.8f} |')
    lines += ['', '## Main100 QPS2.0', '',
        '| Output | Native Policy | Starts | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Raw tok/s | Goodput tok/s | Joint % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + row['mode'] + ' | ' + row['method'] + ' | ' + str(row['independent_starts']) + ' | ' +
            ' | '.join(f'{row[k]:.3f}' for k in ['TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms',
                'raw_output_tokens_per_second', 'SLO_goodput', 'joint_percent']) + ' |')
    lines += ['',
        '새 Natural 실행은 5113 Output Tokens다. Replay는 기존 요청별 길이와 같은 5269 Tokens다.',
        'Natural Output 차이를 Dataset 변경 또는 같은 계산량으로 표현하지 않는다.',
        '같은 Replay 길이도 같은 Answer String을 뜻하지 않는다.',
        'TTFT5000ms, 모든 ITL34ms와 TPOT33ms/token SLO를 유지했다.',
        'Joint는 세 요청별 Gate의 AND다. Goodput 분자는 Joint 통과 요청의 Output Tokens다.',
        '분모는 첫 예정 Arrival부터 마지막 Drain까지다. Sustainable Capacity가 아니다.',
        'CPU Tensor 검사와 GPU Profiler는 두 성능 실행에서 껐다.',
        'Checkpoint 압축과 Push는 Warm-up 중 종료됐다. 두 비계측 성능 실행과 겹치지 않았다.', '',
        '## 품질과 다음 확인', '',
        '첫 Common Native 실행의 Raw DocVQA ANLS, Mantis 점수와 MP-DocVQA ANLS는 Reference 점수와 같았다.',
        'DocVQA는 양쪽 모두 ANLS0이다. 이 Floor를 품질 보존의 충분한 근거로 사용하지 않는다.',
        'MMLongBench-Doc의 공식 GPT Answer Extraction은 미실행이다.',
        'Raw Answer 진단 점수를 완전한 공식 Task Metric으로 표시하지 않는다.',
        '따라서 전체 Quality Gate는 계속 `PENDING`이다.',
        '같은 설정의 새 초기화 두 번을 추가한다. 아직 N1인 결과를 N3로 표시하지 않는다.',
        '대응 CSV는 `analysis/common_native_C32_runtime_pilot_v0.csv`다.',
        'Raw 8-category SLO Count와 Token Weight는 각 Run의 `_metrics.json`에 있다.', '']
    report = ROOT.parent / 'mllm_epd_common_compiled_native_pilot_v1.md'
    assert not report.exists()
    report.write_text('\n'.join(lines))
    preserve()
    print('COMMON_NATIVE_N1_PILOT_RECORDED', flush=True)


if __name__ == '__main__':
    main()
