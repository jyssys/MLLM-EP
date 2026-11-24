"""Audit the saved frozen-fraction rejection and controlled encoder geometry change."""
import csv
import json
import random
import time

from .common import ROOT,digest,preserve,write
from .natural_token_correctness_control import INDICES
from .numerical import embedding_metrics
from .token_split import select_prefix


def main():
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2);preserve()
    pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    rows=[pool[i] for i in INDICES];random.Random(2026112404).shuffle(rows)
    run='token_FROZEN_C36_F25_N8'
    dpath=ROOT/'raw/batch_prefix'/f'{run}_D/manifest.json'
    dm=json.loads(dpath.read_text())
    actual_outputs={r['request_id']:r for r in map(json.loads,(ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines())}
    by_id={row['request_id']:i for i,row in enumerate(pool)}
    numerical=[];layer_rows=[]
    for ordinal,row in enumerate(rows):
        entry=dm['full_prompt_tensor_files'][f'{run}_{ordinal:04d}']
        assert digest(entry['path'])==entry['SHA256']
        actual=load_file(entry['path'])
        index=by_id[row['request_id']]
        teacher=f'token_FROZEN_C36_F25_N8_{index}_TEACHER'
        path=ROOT/'raw/llm_reference'/f'{teacher}_P/manifest.json'
        tm=json.loads(path.read_text());assert digest(tm['tensor_file'])==tm['tensor_SHA256']
        target=load_file(tm['tensor_file']);prefix=select_prefix(row['total_input_tokens'],.25)
        layers=[]
        for layer in range(28):
            a,b=actual[f'prompt_KV_{layer}'],target[f'prompt_KV_{layer}']
            assert a.shape==b.shape and len(a)==row['total_input_tokens']
            p=embedding_metrics(a[:prefix],b[:prefix]);s=embedding_metrics(a[prefix:],b[prefix:])
            layers.append(dict(layer=layer,strict=p['strict'] and s['strict'],
                prefix_L2=p['relative_L2'],suffix_L2=s['relative_L2'],
                passed=p['finite'] and s['finite'] and p['relative_L2']<=.02 and s['relative_L2']<=.02 and
                    p['cosine']>=.999 and s['cosine']>=.999))
        output=json.loads((ROOT/'raw'/f'{teacher}_requests.jsonl').read_text().splitlines()[0])
        item=dict(request_id=row['request_id'],calibration_index=index,
            input_tokens=row['total_input_tokens'],prefix_tokens=prefix,
            strict_layers=sum(r['strict'] for r in layers),tolerance_layers=sum(r['passed'] for r in layers),
            max_prefix_L2=max(r['prefix_L2'] for r in layers),max_suffix_L2=max(r['suffix_L2'] for r in layers),
            output_tokens=actual_outputs[row['request_id']]['output_tokens'],
            strict_greedy_tokens=actual_outputs[row['request_id']]['token_ids']==output['token_ids'],
            all_KV_tolerance=all(r['passed'] for r in layers),
            actual_tensor_SHA256=entry['SHA256'],teacher_manifest_SHA256=digest(path),
            evidence='Saved CORRECTNESS_ONLY state. No new GPU execution.')
        numerical.append(item);layer_rows.extend(dict(request_id=row['request_id'],**r) for r in layers)
        del actual,target
    assert len(numerical)==8 and any(not r['all_KV_tolerance'] for r in numerical)
    control_path=ROOT/'analysis/vision_BATCH_C38R1_comparison.json'
    control=json.loads(control_path.read_text())
    assert control['solo_repeat_strict'] and control['numerical_effect_supported']
    manifests=[];tensors=[];image_lists=[]
    for label in ['Joint_A','Joint_B']:
        base=f'vision_BATCH_C38R1_{label}'
        path=ROOT/'raw/llm_reference'/f'{base}_LLM_P/manifest.json'
        meta=json.loads(path.read_text());assert digest(meta['tensor_file'])==meta['tensor_SHA256']
        manifests.append(digest(path));tensors.append(load_file(meta['tensor_file']))
        image_lists.append(json.loads((ROOT/'raw/encoder_reference'/f'{base}_E/manifest.json').read_text())['rows'])
    joint_repeat=dict(all28_KV_strict=all(torch.equal(tensors[0][f'prompt_KV_{i}'],tensors[1][f'prompt_KV_{i}']) for i in range(28)),
        original_encoder_images_strict=all(
            a['input_pixel_SHA256']==b['input_pixel_SHA256'] and
            torch.equal(load_file(a['file'])['embedding'],load_file(b['file'])['embedding'])
            for a,b in zip(*image_lists)),manifest_SHA256=manifests)
    assert joint_repeat['all28_KV_strict'] and joint_repeat['original_encoder_images_strict']
    image_rows=[];kv_rows=[]
    for comparison in control['comparisons']:
        for i,image in enumerate(comparison['images']):
            image_rows.append(dict(case=comparison['label'],image=i,solo_images=image['reference_images_in_call'],
                actual_images=image['actual_images_in_call'],relative_L2=image['relative_L2'],
                cosine=image['cosine'],strict=image['strict'],same_output=comparison['output_tokens_same']))
        kv_rows.extend(dict(case=comparison['label'],layer=r['layer'],relative_L2=r['relative_L2'],strict=r['strict'])
            for r in comparison['layers'])
    for name,data in [('cases',numerical),('layers',layer_rows),('encoder_images',image_rows),('controlled_KV',kv_rows)]:
        path=ROOT/'analysis'/f'frozen_fraction_C36_encoder_C38_{name}_v0.csv'
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary=dict(time=time.time(),evidence='CORRECTNESS_ONLY and controlled numerical DIAGNOSTIC',
        C36_saved_cases=numerical,C38_control_SHA256=digest(control_path),joint_repeat=joint_repeat,
        conclusion='Changing only the original encoder batch envelope can change visual embeddings and downstream KV. '
            'The exact unrecorded encoder geometry of C36 remains unproved.',
        no_quality_gate_relaxation=True,no_encoder_algorithm_change=True,no_online_speedup=True)
    write(ROOT/'analysis/frozen_fraction_C36_encoder_C38_summary_v0.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for case in ['Solo_B','Joint_A','Joint_B']:
        selected=[r for r in image_rows if r['case']==case]
        axes[0].plot([r['image'] for r in selected],[r['relative_L2'] for r in selected],marker='o',label=case)
        selected=[r for r in kv_rows if r['case']==case]
        axes[1].plot([r['layer'] for r in selected],[r['relative_L2'] for r in selected],label=case)
    axes[0].set_xlabel('Unchanged target image');axes[0].set_ylabel('Encoder embedding relative L2')
    axes[1].set_xlabel('LLM layer');axes[1].set_ylabel('Full-prompt KV relative L2')
    axes[1].axhline(.02,color='gray',linestyle='--',label='Frozen numerical tolerance')
    for axis in axes:axis.grid(alpha=.2);axis.legend()
    figure.suptitle('CORRECTNESS_ONLY | Encoder2 vs5 images | Same pixels and LLM chunks')
    figure.tight_layout()
    for extension in ['png','pdf']:figure.savefig(ROOT/'figures'/f'encoder_batch_C38_numerical_effect_v0.{extension}',dpi=180)
    plt.close(figure)
    lines=['# Encoder Batch 수치 차이 — Frozen Fraction 정합성 감사','',
        '증거 수준은 `CORRECTNESS_ONLY`, 저장 상태의 `TRACE_ANALYSIS`, 통제된 Numerical `DIAGNOSTIC`이다.',
        '이 결과는 C3 Speedup이나 Task Quality PASS가 아니다. 기존 실패와 허용 오차를 변경하지 않았다.','',
        '## C36: 동결한 후보의 정합성 검사','',
        'Discovery에서 TTFT 기준으로 선택한 10%와 Goodput 기준 25%를 검사했다.',
        '10%는 N1 및 동시 N8에서 모든 28 Layers KV와 전체 Greedy Output이 Byte-exact였다.',
        '25%는 N1을 통과했다. 동시 N8의 Mantis cw_9에서 Prefix/Suffix KV 허용 오차를 초과했다.',
        '원래 검사는 첫 실패에서 중단했다. 아래 표는 보존된 8개 실제 실행을 CPU에서 모두 다시 비교한 결과다.',
        'GPU 실행을 반복하지 않았다. 원래 FAILED_EXECUTION 기록은 보존했다.','',
        '| Calib | Input | Strict KV Layers | Tolerance Layers | Max Prefix L2 | Max Suffix L2 | Output Tokens | Strict Output |',
        '|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in numerical:
        lines.append(f'| {r["calibration_index"]} | {r["input_tokens"]} | {r["strict_layers"]}/28 | '
            f'{r["tolerance_layers"]}/28 | {r["max_prefix_L2"]:.6f} | {r["max_suffix_L2"]:.6f} | '
            f'{r["output_tokens"]} | {r["strict_greedy_tokens"]} |')
    lines+=['',
        '실패 요청의 P→D Prefix Import는 28/28 Layers에서 Byte-exact였다. Token IDs와 MRoPE도 일치했다.',
        'Layer0에서는 Visual2916 Rows만 달랐다. Text/Special45 Rows는 일치했다.',
        'P Query는 0→15→2221이었다. Teacher Query는 0→2221→2961이었다.',
        '10% 검사에서도 0→15의 추가 Chunk가 있었지만 모든 KV가 일치했다.',
        '따라서 추가 15-token Chunk만으로 이 실패를 설명할 수 없다.','',
        '## C38R1: Encoder Batch만 변경한 통제 실험','',
        '원본 Target은 같은 Mantis cw_9의 실제 이미지 2개다. 원본 Companion은 Calibration17의 이미지 3개다.',
        'Solo는 Target2개를 처리했다. Joint는 동일 Target2개와 Companion3개를 하나의 Encoder Envelope에서 처리했다.',
        'Joint Envelope는 진단용이다. Main100 요청이나 Prompt를 변경하지 않았다.',
        'Original BF16, 원본 Vision/Processor/Weight/Attention과 GPU4 E/GPU5 P를 유지했다.',
        'Target의 실제 Processor Pixel Hash와 Grid를 검사했다. LLM Prompt, MRoPE, Query0→2221→2961을 고정했다.',
        '각 Geometry를 두 번 실행했다. Solo 반복과 Joint 반복 모두 Embedding 및 28/28 KV가 Byte-exact였다.',
        'CPU Tensor Capture가 있으므로 모든 Timing은 성능 평가에서 제외했다.','',
        '| Comparison | Target Image | Encoder Images/Call | Embedding Relative L2 | Same Greedy Output |',
        '|---|---:|---|---:|---|']
    for r in image_rows:
        lines.append(f'| {r["case"]} vs Solo_A | {r["image"]} | {r["solo_images"]}→{r["actual_images"]} | '
            f'{r["relative_L2"]:.6f} | {r["same_output"]} |')
    lines+=['',
        'Joint의 Full-prompt KV 최대 Relative L2는 '+f'{max(r["relative_L2"] for r in kv_rows):.6f}'+ '였다.',
        'Encoder Batch Geometry만 바꿔도 같은 Pixel에서 Embedding과 후속 KV 차이가 발생했다.',
        '이는 통제 실험에서 재현한 인과 효과다. 정확한 Operator별 기여는 측정하지 않았다.',
        'C36 당시 Encoder Geometry는 기록하지 않았다. 따라서 C36 실패의 직접 원인은 아직 확정하지 않았다.',
        '이 효과를 Prefix Transfer 오류 또는 Token Split 자체의 근사 계산으로 해석하지 않는다.',
        '반대로 출력이 같다는 이유로 KV 오차 실패를 PASS로 바꾸지 않는다.',
        '25%는 Primary 후보에서 제외한다. 10%의 통과 범위도 검사한 요청과 Geometry로 한정한다.','',
        '## Harness 오류와 수정 범위','',
        '초기 C38은 P가 독립 Processor 식별자를 생성해서 원본 E Export를 찾지 못했다.',
        'Strict EC는 Local Vision Fallback 없이 중단했다. 해당 실패를 보존했다.',
        '새 진단 전용 Handoff는 원본 Target Pixel/Token/Grid를 보존하고 실제 E Export 식별자에 연결한다.',
        'Target KV나 Output을 수정하지 않는다. 기존 Main Serving 경로는 변경하지 않았다.',
        '실제 Handoff 경로의 CPU Regression은 수정 전 실패하고 수정 후 통과했다.',
        '전체 CPU Regression은 139개 통과했다. 관련 Red/Green Log를 보존했다.','',
        '## 한계와 재현','',
        'Task Quality는 PENDING이다. MMLongBench-Doc 공식 외부 답변 추출은 승인 없이 실행하지 않았다.',
        '이 실험으로 전체 BI0 Batch Invariance나 안전한 Active-decode 삽입을 보장하지 않는다.',
        '원본 C36/C38/C38R1 Raw, Config, Source ZIP/SHA256, GPU UUID와 Failure Log를 보존했다.',
        '재계산 명령은 `python -m poc_flashvep.mllm_epd_prefill_heavy_token_split.vision_batch_intervention_report`다.',
        'Derived CSV/JSON은 `analysis/frozen_fraction_C36_encoder_C38_*_v0`에 있다.',
        'Figure는 `figures/encoder_batch_C38_numerical_effect_v0.{png,pdf}`다.',
        'Figure 좌측/우측 수치는 각각 `encoder_images`와 `controlled_KV` CSV에 대응한다.','']
    report=ROOT.parent/'mllm_epd_encoder_batch_numerical_root_cause_v1.md'
    assert not report.exists();report.write_text('\n'.join(lines));preserve()
    print('ENCODER_BATCH_SAVED_STATE_RECALCULATED',sum(r['all_KV_tolerance'] for r in numerical),len(numerical),flush=True)


if __name__=='__main__':main()
