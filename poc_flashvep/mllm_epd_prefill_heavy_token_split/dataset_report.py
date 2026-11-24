"""Recalculate the frozen input profile and separately join measured outputs."""
from collections import Counter
import csv
import json
from pathlib import Path
import time
import numpy as np

from .common import ROOT,write,digest,preserve


def describe(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return None
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),std_population=float(x.std()),
        min=float(x.min()),p25=float(np.percentile(x,25)),p75=float(np.percentile(x,75)),
        p90=float(np.percentile(x,90)),p95=float(np.percentile(x,95)),max=float(x.max()))


def main():
    frozen=json.loads((ROOT/'state/dataset_final_input_freeze.json').read_text())
    manifest=Path(frozen['main_manifest']);data=json.loads(manifest.read_text());rows=data['requests']
    assert digest(manifest)==frozen['main_SHA256']
    natural={}
    for file in sorted((ROOT/'raw').glob('reference_main400_natural_template_v2*_requests.jsonl')):
        for line in file.read_text().splitlines():
            r=json.loads(line)
            if r['status']=='MEASURED_ONLINE':
                assert r['request_id'] not in natural,'Do not pool repeated profiling starts as different requests'
                natural[r['request_id']]=r
    for r in rows:
        if r['request_id'] in natural:assert natural[r['request_id']]['prompt_SHA256']==r['prompt_SHA256']
    columns=['total_input_tokens','visual_tokens','text_tokens','special_tokens']
    summaries={k:describe([r[k] for r in rows]) for k in columns}
    summaries['natural_output_tokens']=describe([natural[r['request_id']]['output_tokens'] for r in rows if r['request_id'] in natural])
    source=Counter(r['source'] for r in rows);stratum=Counter(r['stratum'] for r in rows)
    inputs=sum(r['total_input_tokens'] for r in rows);visual=sum(r['visual_tokens'] for r in rows)
    hashes=[i['SHA256'] for r in rows for i in r['images']]
    pixel_hashes=[i['pixel_SHA256'] for r in rows for i in r['images']]
    docs=[r['document_id'] for r in rows if r.get('document_id')]
    feasibility=json.loads((ROOT/'data/feasibility_audit.json').read_text())['requests']
    exclusions=Counter(r['reason'] for r in feasibility if not r['eligible'])
    output_complete=len(natural)==len(rows)
    image_stats=describe([r['image_count'] for r in rows])
    source_statistics={s:dict(requests=source[s],image_count=describe([r['image_count'] for r in rows if r['source']==s]),
        input_tokens=describe([r['total_input_tokens'] for r in rows if r['source']==s]),
        output_tokens=describe([natural[r['request_id']]['output_tokens'] for r in rows if r['source']==s and r['request_id'] in natural])) for s in source}
    cap_hits=sum(bool(r.get('output_cap_hit')) for r in natural.values())
    result=dict(time=time.time(),main_manifest=str(manifest),main_SHA256=digest(manifest),
        model_revision=rows[0]['model_revision'],processor_revision=rows[0]['processor_revision'],
        request_count=len(rows),source_counts=source,stratum_counts=stratum,statistics=summaries,
        image_count_statistics=image_stats,source_statistics=source_statistics,natural_output_cap_hits=cap_hits,
        natural_output_cap_denominator=len(natural),natural_output_total_tokens=sum(r['output_tokens'] for r in natural.values()),
        image_count_distribution=Counter(r['image_count'] for r in rows),
        page_count_distribution=Counter(r['page_count'] for r in rows if r['page_count'] is not None),
        visual_token_ratio_weighted=visual/inputs,total_input_tokens=inputs,total_visual_tokens=visual,
        total_text_tokens=sum(r['text_tokens'] for r in rows),total_special_tokens=sum(r['special_tokens'] for r in rows),
        image_references=len(hashes),unique_image_bytes=len(set(hashes)),unique_image_pixels=len(set(pixel_hashes)),
        unique_request_IDs=len(set(r['request_id'] for r in rows)),unique_documents=len(set(docs)),
        context_components=data['context_components'],calibration_count=32,discovery_count=len(data['discovery_ids']),
        heldout_count=len(data['heldout_ids']),image_document_disjoint=True,truncated_requests=0,
        inspected_for_original_template=len(feasibility),excluded_original_template=len(feasibility)-sum(r['eligible'] for r in feasibility),
        exclusion_reasons=exclusions,natural_output_count=len(natural),natural_output_complete=output_complete,
        task_quality_status='NOT_MEASURED',evidence='CPU_PROCESSOR_MEASUREMENT plus official single-GPU measured generation lengths. Not EPD correctness qualification.')
    write(ROOT/'analysis/dataset_profile.json',result)
    with (ROOT/'analysis/dataset_requests.csv').open('w',newline='') as f:
        keys=['request_id','source','stratum','total_input_tokens','visual_tokens','text_tokens','special_tokens',
            'image_count','page_count','natural_output_tokens','output_cap_hit','prompt_SHA256']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in rows:
            n=natural.get(r['request_id'],{})
            writer.writerow({k:(n.get('output_tokens') if k=='natural_output_tokens' else n.get('output_cap_hit')
                if k=='output_cap_hit' else r.get(k)) for k in keys})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,series in [('input_distribution',[(k,[r[k] for r in rows]) for k in columns[:3]])]+\
        ([('natural_output_distribution',[('Natural EOS',[r['output_tokens'] for r in natural.values()])])] if natural else []):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for label,values in series:ax.hist(values,bins=32,alpha=.45,label=label)
        ax.set_xlabel('Tokens per request');ax.set_ylabel('Request count');ax.legend();ax.grid(alpha=.2)
        ax.set_title(f'MLLM image-only Main400 | {len(natural)}/400 Natural outputs measured')
        fig.tight_layout()
        for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/(name+'.'+extension),dpi=160)
        plt.close(fig)
    if natural:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for s in source:
            group=[r for r in rows if r['source']==s and r['request_id'] in natural]
            ax.scatter([r['total_input_tokens'] for r in group],
                [natural[r['request_id']]['output_tokens'] for r in group],s=18,alpha=.65,label=s)
        ax.set_xlabel('Actual LLM input tokens');ax.set_ylabel('Measured Natural output tokens');ax.legend(fontsize=7)
        ax.set_title('Official single-GPU reference | Not online EPD performance');fig.tight_layout()
        for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/('input_output_scatter.'+extension),dpi=160)
        plt.close(fig)
    lines=['# Image-only MLLM400 Dataset Profile','',
        '입력은 실제 Processor로 측정했다. 이 표는 EPD 성능 결과가 아니다.',
        f'공식 단일 GPU Natural EOS 출력은 현재 {len(natural)}/400개다.',
        '미측정 출력은 0으로 채우지 않았다. Encoder Output Shape는 아직 구조적 예상값이다.','',
        '## 원문과 선정','',
        f'Main400 SHA256은 `{digest(manifest)}`다.',
        f'Model·Processor Revision은 `{rows[0]["model_revision"]}`다.',
        'Seed는 `2026112401`이다. 성능과 출력 길이를 보기 전에 요청을 선정했다.',
        'Calibration32·Discovery200·Heldout200은 이미지·문서 연결 Component로 분리했다.',
        '같은 문서의 질문이 Discovery와 Heldout 양쪽에 들어가지 않는다.','',
        '| Source | Requests | Mean Images | Min–Max Images | Mean Input Tokens | Mean Output Tokens |',
        '|---|---:|---:|---:|---:|---:|']
    for source_name,count in source.items():
        s=source_statistics[source_name];i=s['image_count'];o=s['output_tokens']
        lines.append(f'| {source_name} | {count} | {i["mean"]:.2f} | {i["min"]:.0f}–{i["max"]:.0f} | {s["input_tokens"]["mean"]:.2f} | '+(f'{o["mean"]:.2f}' if o else 'NOT_MEASURED')+' |')
    lines += ['', 'Source는 고정 HF Revision을 사용했다. MP-DocVQA와 DocVQA는 명시한 공개 Mirror다.',
        '원문 ID·정답·이미지 순서·Media Hash는 Manifest에 보존했다.',
        'PDF는 모든 페이지를 72 DPI RGB로 Rendering했다. Crop과 페이지 삭제는 하지 않았다.',
        '선택한 요청의 Truncation 비율은 0/400이다. Input+Output Context 상한은 32,768 Tokens다.',
        'Output Reserve는 2,048 Tokens다. 이는 실제 Natural Output Length가 아니다.',
        '첫 Option Template의 중복 Label을 수정했다. 요청·이미지·정답·Split은 바꾸지 않았다.',
        '이 수정은 Main400 성능 실험 전에 완료했다. 기존 Manifest와 Trace는 그대로 보존했다.','',
        '## 실제 Token 통계','',
        '| Statistic | Total Input | Visual | Text | Special | Natural Output |','|---|---:|---:|---:|---:|---:|']
    for key in ['mean','median','std_population','min','p25','p75','p90','p95','max']:
        values=[summaries[k] for k in columns+['natural_output_tokens']]
        lines.append('| '+key+' | '+' | '.join(f'{v[key]:.2f}' if v else 'NOT_MEASURED' for v in values)+' |')
    lines += ['',f'Input Token 합계는 {inputs:,}개다. Visual Token 합계는 {visual:,}개다.',
        f'Token 가중 Visual 비율은 {visual/inputs:.6f}이다. Text·Special·Visual 합계는 Total Input과 일치한다.',
        f'Image Reference는 {len(hashes):,}개다. 고유 Byte Hash는 {len(set(hashes)):,}개다.',
        f'고유 Pixel Hash는 {len(set(pixel_hashes)):,}개다. 요청 ID는 400/400개가 고유하다.',
        '원문 Dataset의 동일 이미지·문서 재사용은 보존했다. 새로운 이미지 복제는 하지 않았다.','',
        f'요청당 이미지 수는 평균 {image_stats["mean"]:.2f}장, 중앙값 {image_stats["median"]:.0f}장이다.',
        f'p95는 {image_stats["p95"]:.0f}장이다. 최소 {image_stats["min"]:.0f}장, 최대 {image_stats["max"]:.0f}장이다.',
        '문서의 각 페이지 이미지를 포함한 수다. 이미지 해상도가 다르므로 장수만으로 Token 작업량을 비교하지 않는다.',
        f'출력 상한 2,048 Tokens에 도달한 요청은 측정 {len(natural)}개 중 {cap_hits}개다.',
        'Cap Hit 출력은 EOS 종료와 구분한다. 상한에 도달한 요청도 통계에서 제외하지 않았다.','',
        '## 제외와 한계','',
        f'초기 Template 적합성 검사 분모는 {len(feasibility):,}개 요청이다.',
        f'그중 부적격 요청은 {result["excluded_original_template"]:,}개다.',
        '이 분모는 전체 후보 Pool이 아니다. 목표 수 확보 후 남은 후보는 처리하지 않았다.',
        '정확한 제외 사유는 `feasibility_audit.json`에 있다.',
        'Pinned LFS 크기와 응답이 다른 `mi_phone.pdf`는 원문 검증 실패로 기록했다.',
        '해당 응답을 고정 Revision의 정상 원문으로 대체하지 않았다.',
        'Task Quality는 아직 미확정이다. 위 출력 길이는 정확도 점수가 아니다.',
        '이 Dataset은 Image-only 장문 QA 실험이다. 전체 Production Traffic의 대표 분포가 아니다.','',
        'CSV는 `analysis/dataset_requests.csv`다. 집계 JSON은 `analysis/dataset_profile.json`이다.',
        'Natural Output은 부분 측정치와 최종 400개 통계를 구분한다.','']
    (ROOT.parent/'mllm_prefill_heavy400_dataset_profile.md').write_text('\n'.join(lines))
    preserve()


if __name__=='__main__':main()
