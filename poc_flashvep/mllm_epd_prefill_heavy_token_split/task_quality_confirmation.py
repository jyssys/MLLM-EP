"""Keep unchanged raw scores, context clusters and unresolved quality gates explicit."""
import csv
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .dataset_build import keys
from .token_discovery_analysis import table


def context_components(rows):
    parent=list(range(len(rows)));owners={}
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    for index,row in enumerate(rows):
        for key in keys(row):
            if key in owners:parent[find(index)]=find(owners[key])
            else:owners[key]=index
    groups={}
    for index in range(len(rows)):groups.setdefault(find(index),[]).append(index)
    return list(groups.values())


def quality_difference(actual,reference,groups,seed=2026112518,samples=4000):
    actual=np.asarray(actual,dtype=float);reference=np.asarray(reference,dtype=float)
    assert actual.shape==reference.shape and actual.shape[0]==3
    assert sorted(index for group in groups for index in group)==list(range(actual.shape[1]))
    delta=actual-reference;rng=np.random.default_rng(seed);values=[]
    for _ in range(samples):
        blocks=rng.integers(0,3,3)
        selected=rng.integers(0,len(groups),len(groups))
        indices=[index for group in selected for index in groups[group]]
        values.append(float(delta[blocks][:,indices].mean()))
    return dict(difference_pp=100*float(delta.mean()),
        hierarchical_start_context_bootstrap95_CI_pp=(100*np.percentile(values,[2.5,97.5])).tolist(),
        context_clusters=len(groups),request_ids=actual.shape[1],independent_start_pairs=3,resamples=samples,
        scope='Paired start and whole image/document context resampling. Request repeats are not independent server starts.')


def main():
    preserve();source=ROOT/'analysis/token_confirmation_C37_raw_quality_v0.csv'
    with source.open(newline='') as stream:observations=list(csv.DictReader(stream))
    manifest=ROOT/'data/main100_manifest_template_v2.json';rows=json.loads(manifest.read_text())['requests']
    lookup={r['request_id']:r for r in rows};summaries=[];comparisons=[]
    for subset in ['Main100','Heldout50']:
        for task in sorted({r['source'] for r in observations if r['subset']==subset}):
            ids=sorted({r['request_id'] for r in observations if r['subset']==subset and r['source']==task})
            clusters=context_components([lookup[key] for key in ids]);matrices={}
            for method in ['NativeStrong','NativeCommon','Static10','Idle10']:
                selected=[r for r in observations if r['subset']==subset and r['source']==task and r['method']==method]
                by_id={(int(r['block']),r['request_id']):r for r in selected}
                assert len(by_id)==len(ids)*3
                actual=np.array([[float(by_id[block,key]['score']) for key in ids] for block in range(3)])
                matrices[method]=actual
                summaries.append(dict(subset=subset,source=task,method=method,request_ids=len(ids),
                    independent_starts=3,observations=len(selected),context_clusters=len(clusters),
                    unchanged_raw_score_mean=float(actual.mean()),all_scorers_complete=all(r['complete_task_scorer']=='True' for r in selected),
                    strict_R0_output_observations=sum(r['raw_reference_tokens_strict']=='True' for r in selected),
                    full_task_quality='PENDING',scope='Unchanged raw-response task score. R0 strict count is descriptive and not an accuracy requirement.'))
            for method in ['Static10','Idle10']:
                for reference in ['NativeStrong','NativeCommon']:
                    result=quality_difference(matrices[method],matrices[reference],clusters)
                    floor=bool(np.all(matrices[reference]==0))
                    complete=all(r['complete_task_scorer']=='True' for r in observations if r['source']==task)
                    comparisons.append(dict(subset=subset,source=task,method=method,baseline=reference,**result,
                        raw_score_lower_CI_meets_minus2pp=result['hierarchical_start_context_bootstrap95_CI_pp'][0]>=-2,
                        reference_raw_score_floor=floor,official_full_scorer=complete,
                        full_quality_decision='HOLD',
                        reason='Official MMLong answer extraction not run' if not complete else
                            'Reference raw-answer floor prevents a useful semantic noninferiority claim' if floor else
                            'This task alone cannot qualify all sources, state coverage or cross-geometry correctness'))
    table(ROOT/'analysis/task_quality_C37_source_summary_v0.csv',summaries)
    table(ROOT/'analysis/task_quality_C37_comparisons_v0.csv',comparisons)
    write(ROOT/'analysis/task_quality_C37_audit_v0.json',dict(time=time.time(),source_SHA256=digest(source),
        manifest_SHA256=digest(manifest),summaries=summaries,comparisons=comparisons,
        full_task_quality='HOLD',answers_rewritten=False,external_API_calls=0,prompt_or_EOS_changed=False))
    report=ROOT.parent/'mllm_epd_n3_task_quality_limits_v1.md';assert not report.exists()
    lines=['# N3 Natural 출력의 Task 품질 감사','',
        '원래 Answer와 고정한 Task Scorer를 그대로 사용했다. Prompt·EOS·출력을 수정하지 않았다.',
        'Natural EOS만 평가했다. Equal-output Replay는 품질 동등성 검사로 사용하지 않았다.',
        '전체 품질 판정은 `HOLD`다. Runtime 결과의 실패와 이 품질 미확정을 분리한다.','',
        '## Task별 원래 점수','',
        '| Subset | Source | Method | Request IDs | Contexts | Starts | Raw Score | Strict R0 Outputs/Observations |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f'| {row["subset"]} | {row["source"]} | {row["method"]} | {row["request_ids"]} | '+
            f'{row["context_clusters"]} | 3 | {row["unchanged_raw_score_mean"]:.6f} | '+
            f'{row["strict_R0_output_observations"]}/{row["observations"]} |')
    lines+=['','## CI와 해석','',
        '동일 Start Block과 이미지·문서 Context를 함께 Resampling했다. 같은 Context의 질문을 분리하지 않았다.',
        '독립 서버 시작은 3회다. Request 또는 Token 수를 독립 Start 수로 취급하지 않았다.',
        'Paired 차이는 pp다. 고정 품질 Margin은 CI 하한≥−2pp다. Source별 결과는 대응 CSV에 있다.',
        'N3와 작은 Context 수는 CI의 한계다. 유리한 Task만 남기거나 Scorer를 변경하지 않았다.',
        'Mantis의 원래 Parser 결과도 보존했다. 같은 선택지를 다른 문장으로 출력하면 Parser 점수가 달라질 수 있다.',
        'DocVQA는 Native와 후보 모두 Raw ANLS Floor가 남았다. 이 값으로 의미적 품질 동등성을 증명하지 않는다.',
        'MP-DocVQA의 낮은 Raw Score도 그대로 보존했다.',
        'MMLongBench-Doc의 공식 GPT 기반 Answer Extraction은 미실행이다. Raw 점수는 완전한 공식 품질 지표가 아니다.',
        '승인된 외부 API 또는 평가 Key 없이 유료 추출을 실행하지 않았다.',
        'BI0는 서로 다른 Batch Geometry에서 완전히 같은 Greedy 출력을 보장하지 않는다.',
        '그러나 이 사실을 구현 오류나 Task 품질 저하를 허용하는 근거로 사용하지 않는다.',
        '원래 E 출력이 Geometry에 따라 달라지는 진단과 KV 정합성 검사는 별도 보고서에 있다.',
        '동일 E·LLM Geometry의 Strict 상태 검사도 모든 Task 품질을 대신하지 않는다.','',
        '## 재현 자료','',
        '모델 Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.',
        'Main100 SHA256은 '+digest(manifest)+'다. Main100과 기존 Heldout50을 각각 평가했다.',
        'GPU4/5/6, Original BF16, BI0, B8192/S1024, KV42GiB, Block128을 유지했다.',
        'Scorer Source는 `references/quality_sources/`의 고정 Revision과 Hash를 사용했다.',
        '원시 점수는 `analysis/token_confirmation_C37_raw_quality_v0.csv`다.',
        '집계와 CI는 `analysis/task_quality_C37_*_v0`에 있다.','']
    report.write_text('\n'.join(lines));preserve();print('N3_TASK_QUALITY_LIMITS_RECALCULATED',flush=True)


if __name__=='__main__':main()
