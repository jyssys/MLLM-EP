"""Audit real multimodal prefix positions without running a token-split method."""
import csv
import hashlib
import json
import time
from types import SimpleNamespace,MethodType

from .common import ROOT,digest,preserve,write
from .client import payload
from .processor_handoff import RenderStore


def main():
    import torch
    from safetensors import safe_open
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Model
    preserve()
    download=json.loads((ROOT/'state/model_download_complete.json').read_text())
    from pathlib import Path
    config_path=Path(download['directory'])/'config.json'
    config=json.loads(config_path.read_text())
    model=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(
        spatial_merge_size=config['vision_config']['spatial_merge_size'])))
    model.iter_mm_grid_thw=MethodType(Model.iter_mm_grid_thw,model)
    calibration=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    samples=[(calibration[6],'ec_CPU_C6_KV_Mantis6_v0',0)]
    main=json.loads((ROOT/'data/main100_manifest_template_v2.json').read_text())['requests']
    lookup={r['request_id']:r for r in main}
    run='ec_CPU_C6_natural_q2_v0'
    raw_path=ROOT/'raw'/f'{run}_requests.jsonl'
    observed=[json.loads(s) for s in raw_path.read_text().splitlines()]
    for result in [max(observed,key=lambda r:r['input_tokens_manifest']),
                   min((r for r in observed if r['image_count']==1),key=lambda r:r['input_tokens_manifest'])]:
        samples.append((lookup[result['request_id']],run,result['index']))
    rows=[];proofs=[]
    store=RenderStore('/dev/shm/mllm_epd_20261124_ec_CPU_C6/processed_prompts')
    for row,run,index in samples:
        key=hashlib.sha256(f'{run}_{index:04d}'.encode()).hexdigest()
        request=ChatCompletionRequest(**payload(row,run,None))
        conversation,inputs=store.load(key,request,time.time())
        prompt=inputs[0];ids=prompt['prompt_token_ids'];length=len(ids)
        positions=prompt['mm_placeholders']['image']
        features=[SimpleNamespace(modality='image',mm_position=position,data=item) for position,item in
            zip(positions,prompt['mm_kwargs']['image'])]
        assert length==row['total_input_tokens']
        full,delta=Model.get_mrope_input_positions(model,ids,features)
        assert full.shape==(3,length)
        captured_match=None
        if row['request_id']==calibration[6]['request_id']:
            captured=json.loads((ROOT/'raw/llm_reference/ec_CPU_C6_KV_P/manifest.json').read_text())
            assert digest(captured['tensor_file'])==captured['tensor_SHA256']
            with safe_open(captured['tensor_file'],framework='pt',device='cpu') as file:
                captured_match=torch.equal(full,file.get_tensor('full_prompt_mrope'))
                assert captured_match and torch.equal(torch.tensor(ids),file.get_tensor('prompt_token_ids'))
        cuts=[('before_visual',positions[0].offset),
              ('inside_visual',positions[0].offset+positions[0].length//2),
              ('after_visual',positions[0].offset+positions[0].length),
              ('text_tail',length-8)]
        if len(positions)>1:cuts.append(('between_images',positions[1].offset))
        for boundary,cut in cuts:
            if not 0<cut<length:continue
            prefix,prefix_delta=Model.get_mrope_input_positions(model,ids[:cut],features)
            prefix_match=prefix.shape[1]>=cut and torch.equal(prefix[:,:cut],full[:,:cut])
            p_pairs=cut*(cut+1)//2;all_pairs=length*(length+1)//2
            rows.append(dict(request_id=row['request_id'],source=row['source'],images=row['image_count'],
                length=length,boundary=boundary,prefix_tokens=cut,suffix_tokens=length-cut,
                generated_prefix_position_columns=prefix.shape[1],prefix_position_match=prefix_match,
                original_full_delta=delta,prefix_function_delta=prefix_delta,
                P_token_layer_rows=28*cut,D_token_layer_rows=28*(length-cut),
                P_attention_pairs_all28=28*p_pairs,D_attention_pairs_all28=28*(all_pairs-p_pairs),
                full_attention_pairs_all28=28*all_pairs))
            assert prefix_match
        proofs.append(dict(request_id=row['request_id'],full_positions_shape=list(full.shape),
            source_metadata_SHA256=digest(store.directory(key)/'manifest.json'),
            actual_Native_GPU_capture_match=captured_match))
    path=ROOT/'analysis/generic_token_boundary_position_audit_v0.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(time=time.time(),evidence=['CPU_POSITION_FUNCTION_MEASUREMENT','STRUCTURAL_ACCOUNTING'],
        scope='Original installed Qwen position function applied to actual online E-produced frozen prompts. '
            'No GPU prefix/suffix method, no continuation KV transfer and no online speedup were run.',
        full_model_config_SHA256=digest(config_path),source_model_revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',
        rows=rows,samples=proofs,position_checks_passed=True,generic_token_split_GPU_status='NOT_RUN',
        limitations=['All original image grids and full multimodal features remained present in this CPU audit.',
            'The original position function can return columns beyond the actual prefix query length.',
            'Prefix delta is not the original full-request decode delta. D must retain the original full metadata.',
            'Engine prompt-length validation, partial-image scheduler coverage and actual query placement remain unverified.',
            'Native NIXL full-prompt matching must not overstate a shorter remote prefix count.',
            'Attention pairs are structural counts, not measured GPU time or FLOPs.'])
    write(ROOT/'analysis/generic_token_boundary_position_audit_v0.json',result)
    preserve();print('MM_TOKEN_POSITION_AUDIT_PASS',len(samples),'requests',len(rows),'boundaries','GPU_TOKEN_SPLIT_NOT_RUN')


if __name__=='__main__':main()
