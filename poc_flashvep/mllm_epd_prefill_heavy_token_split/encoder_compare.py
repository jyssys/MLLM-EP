"""Compare actual online E exports with unmodified colocated vLLM vision outputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from .common import ROOT,write,digest
from .numerical import embedding_metrics


def main():
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoProcessor
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--ec-path',type=Path,required=True)
    p.add_argument('--teacher',default='colocated_calibration32_cache1_v2');p.add_argument('--name',required=True)
    p.add_argument('--indices',default='0,1,4,6,12,17,20,31');p.add_argument('--require-pass',action='store_true');args=p.parse_args()
    torch.set_num_threads(1)
    rows=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    teacher_file=ROOT/'raw/encoder_reference'/args.teacher/'manifest.json'
    teacher=json.loads(teacher_file.read_text());mapping=defaultdict(list)
    for r in teacher['rows']:mapping[r['input_pixel_SHA256']].append(r)
    model=json.loads((ROOT/'state/model_download_complete.json').read_text())
    proc=AutoProcessor.from_pretrained(model['directory'],use_fast=True);results=[]
    for index in map(int,args.indices.split(',')):
        row=rows[index];images=[Image.open(m['path']).convert('RGB') for m in row['images']]
        batch=proc(text=[row['prompt']],images=images,return_tensors='pt')
        pixels=batch['pixel_values'].to(torch.bfloat16);offset=0;items=[]
        for i,grid in enumerate(batch['image_grid_thw']):
            count=int(grid.prod());sample=pixels[offset:offset+count].contiguous();offset+=count
            key=hashlib.sha256(str(grid.tolist()).encode()+sample.view(torch.uint8).numpy().tobytes()).hexdigest()
            references=mapping.get(key,[])
            assert len(references)==1,('Reference input-pixel key coverage',index,i,key,len(references))
            reference=references[0];target=load_file(reference['file'])['embedding'].float()
            h=hashlib.sha256((args.run+row['request_id']+str(i)).encode()).hexdigest()
            export=args.ec_path/h/'encoder_cache.safetensors'
            actual=load_file(export)['ec_cache'].float();assert actual.shape==target.shape
            metrics=embedding_metrics(actual,target)
            items.append(dict(index=i,input_pixel_SHA256=key,teacher_file=reference['file'],export_file=str(export),
                teacher_SHA256=digest(reference['file']),export_SHA256=digest(export),shape=list(actual.shape),
                **metrics,passed=metrics['finite'] and metrics['relative_L2']<=.02 and metrics['cosine']>=.999))
        results.append(dict(index=index,request_id=row['request_id'],images=items,passed=all(x['passed'] for x in items)))
    value=dict(time=time.time(),requests=results,passed=all(r['passed'] for r in results),
        teacher_manifest_SHA256=digest(teacher_file),gates=dict(relative_L2_max=.02,cosine_min=.999),
        evidence='CORRECTNESS_ONLY',scope='Actual online E exports versus actual original colocated official vLLM vision. '
            'Exact quantized input-pixel hashes link each image. This is not a throughput measurement.')
    write(ROOT/'analysis'/(args.name+'.json'),value)
    print('ENCODER_COMPARE',value['passed'],len(results),max(x['relative_L2'] for r in results for x in r['images']))
    if args.require_pass:assert value['passed'],'Actual online E embeddings fail the frozen correctness gate'


if __name__=='__main__':main()
