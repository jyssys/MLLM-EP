"""Audit a raw-answer score floor without changing the scorer or answers."""
import csv
import json
import re
import time

from .common import ROOT,digest,preserve,write


def normalized(text):
    return ' '.join(re.sub(r'[^\w]+',' ',str(text).lower()).split())


def main():
    preserve();source=ROOT/'analysis/C14_Agg_EPD2_N3_quality_v0.json'
    value=json.loads(source.read_text());rows=[];summaries=[]
    for run,observations in value['runs'].items():
        selected=[r for r in observations['requests'] if r['source']=='lmms-lab/DocVQA']
        assert len(selected)==13
        for item in selected:
            answer=' '+normalized(item['raw_answer'])+' '
            # This uses the reference. It is a format diagnostic, not accuracy.
            contains=any((' '+normalized(reference)+' ') in answer for reference in item['reference_answers'])
            rows.append(dict(run=run,request_id=item['request_id'],official_raw_ANLS=item['score'],
                reference_text_present=contains,output_tokens=item['output_tokens'],
                raw_answer=item['raw_answer'],reference_answers=json.dumps(item['reference_answers'],ensure_ascii=False),
                evidence='DIAGNOSTIC, not a task accuracy score'))
        details=[r for r in rows if r['run']==run]
        summaries.append(dict(run=run,requests=13,raw_ANLS_mean=sum(r['official_raw_ANLS'] for r in details)/13,
            reference_text_present=sum(r['reference_text_present'] for r in details),
            no_quality_gate_change=True))
    with (ROOT/'analysis/DocVQA_raw_answer_floor_v0.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/DocVQA_raw_answer_floor_v0.json',dict(time=time.time(),source_SHA256=digest(source),
        evidence='DIAGNOSTIC',summaries=summaries,official_scores_unchanged=True,
        reference_aware_answer_rewriting=False,task_quality='PENDING',
        scope='Reference-text presence in a longer answer does not prove correctness. '
              'It can also occur in an incorrect explanation. Do not use this diagnostic as accuracy, '
              'an alternative task metric, an extraction policy or a quality-gate pass.'))
    report=ROOT.parent/'mllm_epd_docvqa_raw_answer_score_floor_v1.md';assert not report.exists()
    lines=['# DocVQA Raw Answer 점수의 Floor 진단','',
        '원래 DocVQA ANLS 결과를 유지했다. 답변과 공식 Scorer는 수정하지 않았다.',
        '일부 답변은 정답 Text를 포함하지만 문장 전체의 Edit Distance 때문에 Raw ANLS가 0이다.',
        '증거 수준은 `DIAGNOSTIC`이다. Task Accuracy 또는 품질 Gate PASS로 사용하지 않는다.','',
        '| Run | Requests | Raw ANLS Mean | Reference Text Present |',
        '|---|---:|---:|---:|']
    for item in summaries:
        lines.append(f"| {item['run']} | 13 | {item['raw_ANLS_mean']:.6f} | {item['reference_text_present']}/13 |")
    lines+=['','Reference Text Present는 소문자·구두점 정규화 후 연속된 단어열의 포함 여부다.',
        '이 지표는 Reference Answer를 사용한다. 잘못된 설명에도 정답 단어가 포함될 수 있다.',
        '따라서 올바른 Answer Extraction이나 의미적 정답 판정을 대신하지 않는다.',
        '예를 들어 원래 답변 `The total pages scanned is 3.`은 Reference `3`을 포함한다.',
        '원래 Raw ANLS는 0이다. 이 진단 뒤에도 그 점수는 0으로 남겼다.',
        'Main100 Prompt와 Natural Output 계약은 동결 상태다. 결과를 보고 Short-answer 지시를 추가하지 않았다.',
        'MMLongBench-Doc의 공식 외부 Answer Extraction도 여전히 `NOT_RUN`이다.',
        'Raw ANLS 0을 곧바로 모델의 의미적 Accuracy 0으로 해석하지 않는다.',
        '반대로 이 Format 진단으로 전체 Quality PASS를 부여하지도 않는다.',
        '대응 Request CSV는 `analysis/DocVQA_raw_answer_floor_v0.csv`다.',
        '원래 Task Score와 Source Scorer Provenance는 `analysis/C14_Agg_EPD2_N3_quality_v0.json`에 있다.',
        '모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.','']
    report.write_text('\n'.join(lines));preserve()
    print('DOCVQA_RAW_SCORE_FLOOR_DIAGNOSTIC_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
