"""Original HF vision outputs versus actual online E exports. Not serving timing."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types

from .common import ROOT,write,digest


def main():
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoProcessor,AutoConfig,Qwen2_5_VLForConditionalGeneration
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as RuntimeModel
    p=argparse.ArgumentParser();p.add_argument('--encoder-run',required=True);p.add_argument('--ec-path',type=Path,required=True)
    p.add_argument('--name',required=True);args=p.parse_args()
    model_receipt=json.loads((ROOT/'state/model_download_complete.json').read_text())
    manifest=ROOT/'data/calibration_manifest_template_v2.json'
    rows=json.loads(manifest.read_text())['requests'];indices=[0,1,4,6,12,17,20,31]
    gates=dict(embedding_relative_L2_max=.02,embedding_cosine_min=.999,
        MRoPE_positions_equal=True,MRoPE_delta_equal=True,exact_placeholder_count=True)
    write(ROOT/'configs'/(args.name+'.json'),dict(time=time.time(),gates=gates,indices=indices,
        manifest_SHA256=digest(manifest),model_revision=model_receipt['revision'],
        scope='CORRECTNESS_ONLY. Original HF vision reference. No precomputed embeddings used for online performance.'))
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_receipt['directory'],
        dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    model.model.visual.to('cuda')
    proc=AutoProcessor.from_pretrained(model_receipt['directory'])
    config=AutoConfig.from_pretrained(model_receipt['directory'])
    position_model=types.SimpleNamespace(config=config)
    position_model.iter_mm_grid_thw=types.MethodType(RuntimeModel.iter_mm_grid_thw,position_model)
    results=[]
    for index in indices:
        row=rows[index];images=[Image.open(m['path']).convert('RGB') for m in row['images']]
        batch=proc(text=[row['prompt']],images=images,return_tensors='pt')
        metadata=np.load(row['token_metadata_file'])
        assert np.array_equal(batch['input_ids'][0].numpy(),metadata['input_ids'])
        features=[types.SimpleNamespace(modality='image',mm_position=types.SimpleNamespace(offset=span[0]),
            data={'image_grid_thw':types.SimpleNamespace(data=torch.tensor(grid))})
            for span,grid in zip(row['image_visual_spans'],row['grid_thw'])]
        positions,delta=RuntimeModel.get_mrope_input_positions(position_model,batch['input_ids'][0].tolist(),features)
        position_equal=np.array_equal(positions.numpy(),metadata['positions'][:,0,:])
        delta_equal=delta==int(metadata['rope_delta'].reshape(-1)[0])
        with torch.inference_mode():
            tensors=model.get_image_features(batch['pixel_values'].to('cuda',dtype=torch.bfloat16),
                batch['image_grid_thw'].to('cuda')).pooler_output
        per_image=[]
        for i,reference in enumerate(tensors):
            h=hashlib.sha256((args.encoder_run+row['request_id']+str(i)).encode()).hexdigest()
            target=args.ec_path/h/'encoder_cache.safetensors'
            assert target.is_file(),('Missing exact request UUID export',str(target))
            exported=load_file(target,device='cpu')['ec_cache'];teacher=reference.float().cpu()
            actual=exported.float();assert actual.shape==teacher.shape==(row['images'][i]['visual_tokens'],3584)
            difference=actual-teacher
            relative=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(teacher))
            cosine=float(torch.nn.functional.cosine_similarity(actual.reshape(1,-1),teacher.reshape(1,-1)))
            per_image.append(dict(index=i,shape=list(actual.shape),export_file=str(target),
                export_SHA256=digest(target),teacher_mean=float(teacher.mean()),teacher_RMS=float(teacher.square().mean().sqrt()),
                relative_L2=relative,cosine=cosine,max_abs=float(difference.abs().max()),
                mean_abs=float(difference.abs().mean()),
                passed=relative<=gates['embedding_relative_L2_max'] and cosine>=gates['embedding_cosine_min']))
        result=dict(index=index,request_id=row['request_id'],MRoPE_positions_equal=position_equal,
            MRoPE_delta_equal=delta_equal,images=per_image,
            passed=position_equal and delta_equal and all(r['passed'] for r in per_image))
        results.append(result);write(ROOT/'analysis'/(args.name+'.json'),dict(time=time.time(),
            completed=len(results),target=len(indices),gates=gates,requests=results,
            passed=len(results)==len(indices) and all(r['passed'] for r in results),evidence='CORRECTNESS_ONLY'))
        print('HF_ENCODER_CHECK',index,result['passed'],max(r['relative_L2'] for r in per_image),flush=True)


if __name__=='__main__':main()
