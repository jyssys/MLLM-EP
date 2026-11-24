"""Report actual batched KV import without claiming decode or quality gates."""
import csv
import json
import time

from .common import ROOT, digest, preserve, write


def main():
    preserve()
    tag = 'token_BATCH_C31'
    summary_path = ROOT / 'analysis' / f'{tag}_summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['passed'] and len(summary['cohorts']) == 2
    pool = json.loads((ROOT / 'data/calibration_manifest_template_v2.json').read_text())['requests']
    indices = [0, 1, 4, 6, 12, 17, 20, 31]
    rows = []
    for cohort in summary['cohorts']:
        run = cohort['run']
        outputs = json.loads((ROOT / 'raw' / f'{run}_responses.json').read_text())['rows']
        assert len(outputs) == len(cohort['checks'])
        for ordinal, (check, output) in enumerate(zip(cohort['checks'], outputs)):
            assert check['passed'] and len(check['imported_layers_strict']) == 28
            index = indices[ordinal] if run.endswith('N8') else 0
            assert output['row_id'] == pool[index]['request_id']
            teacher = ROOT / 'raw' / f'token_CORRECTNESS_C30_N8_{index}_TEACHER_requests.jsonl'
            ref = json.loads(teacher.read_text().splitlines()[0])
            actual_ids = output['response']['choices'][0]['token_ids']
            rows.append(dict(cohort=run, request_id=output['row_id'], calibration_index=index,
                input_tokens=check['full_prompt_tokens'], prefix_tokens=check['prefix_tokens'],
                suffix_tokens=check['full_prompt_tokens'] - check['prefix_tokens'],
                image_count=pool[index]['image_count'], imported_KV_strict_layers=28,
                P_query_coverage=check['query_coverage']['P'], D_query_coverage=check['query_coverage']['D'],
                prefix_ids_strict=check['prefix_ids_strict'], prefix_mrope_strict=check['prefix_mrope_strict'],
                first_token=actual_ids[0], serial_teacher_first_token=ref['token_ids'][0],
                first_token_matches_serial_teacher=actual_ids == ref['token_ids'][:1],
                teacher_SHA256=digest(teacher), evidence='CORRECTNESS_ONLY'))
    table = ROOT / 'analysis/batch_token_C31_request_correctness_v0.csv'
    with table.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = ROOT / 'analysis/batch_token_C31_report_audit_v0.json'
    write(provenance, dict(time=time.time(), evidence='CORRECTNESS_ONLY',
        summary_SHA256=digest(summary_path), CSV_SHA256=digest(table),
        actual_batch_histograms={r['run']: r['actual_batch_histograms'] for r in summary['cohorts']},
        comparisons=rows, independent_initializations=1, output_cap=1, EOS=True,
        no_subsequent_decode=True, task_quality='PENDING', full_suffix_KV_equivalence='NOT_MEASURED',
        active_decode_insertion='NOT_RUN', continuous_serving_speedup='NOT_RUN'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 3.6))
    grid = [[int(row['imported_KV_strict_layers'] == 28)] * 28 for row in rows if row['cohort'].endswith('N8')]
    axis.imshow(grid, vmin=0, vmax=1, cmap='Greens', aspect='auto')
    axis.set(xticks=[0, 3, 7, 11, 15, 19, 23, 27], xticklabels=[1, 4, 8, 12, 16, 20, 24, 28],
        yticks=range(8), yticklabels=['Calib' + str(i) for i in indices], xlabel='Transformer layer',
        ylabel='Concurrent request', title='CORRECTNESS_ONLY: exact NIXL prefix import, all 28 layers')
    figure.text(.5, .01, 'One first output per request. Not a decode-protection or task-quality result.',
        ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .035, 1, 1))
    stem = ROOT / 'figures/batch_token_C31_prefix_correctness_v0'
    for extension in ['png', 'pdf']:
        assert not stem.with_suffix('.' + extension).exists()
        figure.savefig(stem.with_suffix('.' + extension), dpi=180)
    plt.close(figure)
    n8 = [r for r in rows if r['cohort'].endswith('N8')]
    first_matches = sum(r['first_token_matches_serial_teacher'] for r in n8)
    lines = ['# Generic Token Split — 동시 요청 Prefix 정합성', '',
        'C31 N1과 동시 제출 N8의 Prefix 상태 검사가 통과했다.',
        '증거 수준은 `CORRECTNESS_ONLY`다. Online 성능과 Decode 보호 결과가 아니다.', '',
        '## 실행 계약', '',
        'Qwen2.5-VL-7B, Original BF16, BI0, 원본 Processor, 전체 28개 Layer를 유지했다.',
        '모델/Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.',
        'GPU4/5/6은 E/P/D다. P/D B8192/S1024, KV42GiB, Block128이다.',
        '공통 Compiled 설정은 `mllm_epd_compiled_token_correctness_v1.md`와 같다.',
        '원본 이미지 전체를 E에서 실제 Encoding했다. 전체 Sequence의 마지막 20%가 D Suffix다.',
        '각 요청의 생성 상한은 1 Token이다. EOS를 허용했다.',
        '이는 정합성 전용 계약이다. Main100 Natural EOS Cap2048 계약은 변경하지 않았다.', '',
        '## 동시 요청과 실제 Batch', '',
        'N1 뒤에 서로 다른 Calibration 요청 8개를 함께 제출했다.',
        '모든 P Prefix가 끝난 뒤 D Continuation 8개를 함께 제출했다.',
        '이 Barrier는 Pending KV의 Lifetime을 검사한다. 제안하는 Serving 정책이 아니다.',
        '기본 vLLM Continuous Scheduler와 원본 Forward는 유지했다.',
        '동시 요청 수 8을 실제 GPU Batch8로 표시하지 않는다.', '',
        '| Cohort | P 실제 Batch별 Forward 수 | D 실제 Batch별 Forward 수 |',
        '|---|---|---|']
    for cohort in summary['cohorts']:
        hist = cohort['actual_batch_histograms']
        lines.append(f'| {cohort["run"]} | {hist["P"]} | {hist["D"]} |')
    lines += ['', '## 요청별 검사', '',
        'P/D의 Prefix IDs, Prefix MRoPE와 실제 NIXL 수신 KV Hash를 비교했다.',
        '28개 Layer 모두 Byte 단위로 일치했다. Physical Block 번호는 비교하지 않았다.',
        'P Query는 0→k, D Query는 k→L을 빠짐없이 한 번씩 처리했다.',
        '실제 Forward의 Query Position은 요청별 원본 MRoPE Slice와 일치했다.',
        '읽기 계측은 원본 Forward 뒤에 CPU Hash를 계산한다. 모든 Timing은 성능 평가에서 제외한다.', '',
        '| Calibration | Input | P Prefix | D Suffix | Images | Strict Prefix Layers | Serial Teacher 첫 Token 일치 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in n8:
        lines.append('| ' + str(row['calibration_index']) + ' | ' + ' | '.join(str(row[k]) for k in
            ['input_tokens', 'prefix_tokens', 'suffix_tokens', 'image_count', 'imported_KV_strict_layers',
             'first_token_matches_serial_teacher']) + ' |')
    lines += ['', f'Serial Teacher의 첫 Token과 일치한 요청은 {first_matches}/8이다.',
        'BI0는 다른 Batch Geometry 사이의 완전 불변성을 보장하지 않는다.',
        '모든 요청 완료 뒤 E Embedding Ownership은 비어 있었다.',
        '미확인 ACK를 무시하거나 사용 중인 Buffer를 조기에 반환하지 않았다.', '',
        '## 이 검사가 입증하지 않는 항목', '',
        '전체 Suffix KV의 Cross-batch 동등성과 Task Quality는 이 검사에서 미확정이다.',
        '긴 Natural Output, 후속 Decode Residency, Active Decode 중 삽입은 검사하지 않았다.',
        '독립 N3 C3 성능, Native 대비 ITL/TPOT 보호와 SLO-goodput도 미측정이다.',
        '동시 Prefix Handoff 통과를 전체 Continuous Serving 성공으로 표시하지 않는다.',
        'Raw Source, 명령, GPU Lease는 C31 Namespace에 보존했다.',
        '대응 CSV는 `analysis/batch_token_C31_request_correctness_v0.csv`다.',
        'Figure는 `figures/batch_token_C31_prefix_correctness_v0.png`와 같은 이름의 PDF다.', '']
    report = ROOT.parent / 'mllm_epd_batched_token_prefix_correctness_v1.md'
    assert not report.exists()
    report.write_text('\n'.join(lines))
    preserve()
    print('BATCH_TOKEN_CORRECTNESS_RECORDED', len(n8), first_matches, flush=True)


if __name__ == '__main__':
    main()
