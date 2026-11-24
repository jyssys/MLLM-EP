"""Report conditional full P/D equivalence from shared actual online E outputs."""
import argparse
import json
from pathlib import Path
import time

from .common import ROOT,digest,preserve,write
from .token_discovery_analysis import table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='whole_DEFLECTION_C43R4')
    args=parser.parse_args();tag=args.tag;preserve()
    source=ROOT/'analysis'/f'{tag}_CORRECT_summary.json';value=json.loads(source.read_text())
    assert value['status']=='CORRECTNESS_PASS' and len(value['cases'])==9
    rows=[];manifests=[]
    for record in value['cases']:
        path=ROOT/'analysis'/f'{record["base"]}_same_E_comparison.json'
        actual=json.loads(path.read_text());assert actual==record and actual['passed']
        assert len(actual['layers'])==28 and all(r['strict'] for r in actual['layers'])
        assert all(actual['fields'].values()) and all(r['strict'] for r in actual['prepared_inputs'])
        assert actual['strict_greedy_tokens']
        for item in actual['manifests']:
            manifest=Path(item['path']);assert digest(manifest)==item['SHA256']
            metadata=json.loads(manifest.read_text())
            manifests.append(dict(case=actual['base'],manifest_path=str(manifest),manifest_SHA256=item['SHA256'],
                tensor_file=metadata['tensor_file'],tensor_SHA256=metadata['tensor_SHA256'],
                scope='Original tensor hashes recorded and checked by the actual GPU comparison. Payloads remain local.'))
        rows.append(dict(case=actual['base'],request_id=actual['request_id'],
            N1_or_N8='N1' if '_N1_' in actual['base'] else 'N8',
            prompt_IDs_strict=actual['fields']['prompt_token_ids'],
            full_MRoPE_strict=actual['fields']['full_prompt_mrope'],delta_strict=actual['fields']['mrope_delta'],
            strict_KV_layers=sum(r['strict'] for r in actual['layers']),total_KV_layers=28,
            prepared_inputs_and_positions_strict=all(r['strict'] for r in actual['prepared_inputs']),
            full_greedy_output_strict=actual['strict_greedy_tokens'],output_tokens=actual['output_tokens'],
            actual_D_batch=actual['actual_D_batch'],teacher_P_batch=actual['teacher_P_batch'],
            independent_E_P_D_initializations=1,evidence='CORRECTNESS_ONLY',
            scope='Shared actual E output. Full P and full D use the same original full-model query geometry. Not cross-encoder-geometry quality.'))
    table(ROOT/'analysis'/f'{tag}_whole_correctness_v0.csv',rows)
    write(ROOT/'analysis'/f'{tag}_whole_correctness_audit_v0.json',dict(time=time.time(),source_SHA256=digest(source),
        cases=9,unique_request_ids=len({r['request_id'] for r in rows}),independent_initializations=1,
        strict_all28_cases=9,strict_full_outputs=9,manifests=manifests,full_task_quality='PENDING',
        serving_gain='NOT_TESTED_IN_THESE_CAPTURED_COHORTS',source='Actual online E, no offline embedding replacement or tensor correction.'))
    report=ROOT.parent/'mllm_epd_whole_deflection_correctness_v1.md';assert not report.exists()
    lines=['# Whole-request Deflection — Same-E 정합성 검사','',
        '증거 수준은 `CORRECTNESS_ONLY`다. 실제 Online E 출력을 Full P와 Full D가 공유했다.',
        'N1+N8의 9개 실행 사례가 통과했다. 고유 Request ID는 8개다. 독립 E/P/D 초기화는 1회다.',
        '같은 E 출력을 조건으로 한 검사다. 서로 다른 Encoder Batch의 출력 동등성이나 전체 Task 품질을 증명하지 않는다.','',
        '## 실제 경로','',
        '원래 E가 실제 이미지 전체를 처리했다. 전체 Prompt는 P를 거치지 않고 D에서 Full28 Prefill과 Decode를 수행했다.',
        '이 요청은 P Prefix나 P→D Prefix KV를 사용하지 않았다. Layer/Token Split이 아니다.',
        '실제 D 요청이 끝난 뒤, 같은 E Export를 Full P Teacher에 전달했다.',
        '두 실제 소비자의 Clone ACK가 끝난 뒤에만 E 소유권을 반환했다.',
        '첫 Greedy 답변이나 KV를 고쳐서 Reference에 맞추지 않았다.',
        'D Full Prefill은 이전 D Stream이 없을 때만 시작했다. Capture의 CPU Copy 때문에 모든 Timing은 성능 판정에서 제외했다.','',
        '## 모든 실행 사례','',
        '| Phase | Request | Prompt/MRoPE/Delta Strict | KV Layers Strict | Prepared Input Strict | Full Output Strict | Output Tokens |',
        '|---|---|---|---:|---|---|---:|']
    for row in rows:
        positions=row['prompt_IDs_strict'] and row['full_MRoPE_strict'] and row['delta_strict']
        lines.append(f'| {row["N1_or_N8"]} | {row["request_id"]} | {positions} | 28/28 | '+
            f'{row["prepared_inputs_and_positions_strict"]} | {row["full_greedy_output_strict"]} | {row["output_tokens"]} |')
    lines+=['','## 계약과 한계','',
        'Original BF16 Qwen2.5-VL-7B, BI0, 원래 Vision/Processor/Attention을 유지했다.',
        '모델 Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. GPU4/5/6은 E/P/D다.',
        'B8192/S1024, KV42GiB, Block128, Prefix Cache Off와 공통 Compiled Reduction 설정이다.',
        '단일 이미지·여러 이미지·3페이지·28페이지·45페이지를 포함했다.',
        'D의 실제 Full-request 실행은 정합성 정책에 따라 직렬화했다. C3 Main의 Actual Batch를 1로 가정하지 않는다.',
        'N8의 E는 동시 요청을 받았다. Teacher는 각 요청의 같은 실제 E 출력을 사용했다.',
        '이 조건부 증거는 이전 Encoder Geometry 차이나 Token25%의 KV 실패를 지우지 않는다.',
        'Main100과 동결한 Supplemental50의 Natural/Replay C3 결과는 별도로 측정한다.',
        'Whole-request 선택은 Kairos의 공식 재현 또는 새로운 Modality-aware Method로 명명하지 않는다.',
        '현재 전체 Task Quality는 PENDING이다. 이 보고서만으로 Online Serving GO를 부여하지 않는다.','',
        '원시 Tensor Manifest와 해시는 대응 Audit JSON에 있다.',
        f'실행별 CSV는 `analysis/{tag}_whole_correctness_v0.csv`다.',
        '공통 경로는 `mllm_epd_prefill_heavy_token_split_20261124/`다.','']
    report.write_text('\n'.join(lines));preserve();print('WHOLE_SAME_E_N1_N8_CORRECTNESS_REPORTED',flush=True)


if __name__=='__main__':main()
