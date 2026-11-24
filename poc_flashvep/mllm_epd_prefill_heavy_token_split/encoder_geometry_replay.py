"""Replay actual teacher vision batch geometry. Correctness only, not a policy."""
import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from .common import ROOT,MODEL_ID,write,append,digest,preserve
from .encoder_capture_control import rpc
from .client import messages
from .numerical import embedding_metrics


async def replay(args):
    import aiohttp
    from PIL import Image
    import torch
    from transformers import AutoProcessor
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    model=json.loads((ROOT/'state/model_download_complete.json').read_text())
    processor=AutoProcessor.from_pretrained(model['directory'])
    manifest=ROOT/'data/calibration_manifest_template_v2.json'
    rows=json.loads(manifest.read_text())['requests']
    teacher_file=ROOT/'raw/encoder_reference'/args.teacher/'manifest.json'
    teacher=json.loads(teacher_file.read_text())['rows']
    candidates=defaultdict(list)
    for item in teacher:candidates[item['input_pixel_SHA256']].append(item)
    by_pixel={}
    for key,items in candidates.items():
        signatures={(r['SHA256'],r['call_id'],r['images_in_call'],r['patches_in_call']) for r in items}
        if len(signatures)==1:by_pixel[key]=items[0]
    result_file=ROOT/'analysis'/(args.run+'.json');assert not result_file.exists()
    plan=[]
    for index in map(int,args.indices.split(',')):
        row=rows[index];images=[Image.open(m['path']).convert('RGB') for m in row['images']]
        batch=processor(text=[row['prompt']],images=images,return_tensors='pt')
        pixels=batch['pixel_values'].to(torch.bfloat16);offset=0;groups=defaultdict(list)
        for i,grid in enumerate(batch['image_grid_thw']):
            n=int(grid.prod());sample=pixels[offset:offset+n].contiguous();offset+=n
            key=hashlib.sha256(str(grid.tolist()).encode()+sample.view(torch.uint8).numpy().tobytes()).hexdigest()
            assert key in by_pixel,('Selected image has ambiguous teacher geometry',index,i,key)
            ref=by_pixel[key];groups[ref['call_id']].append(dict(index=i,key=key,reference=ref))
        for call,items in groups.items():
            assert len(items)==items[0]['reference']['images_in_call'],('Teacher batch spans requests',call)
            plan.append(dict(request_index=index,request_id=row['request_id'],call_id=call,items=items))
    write(ROOT/'configs'/(args.run+'.json'),dict(time=time.time(),plan=plan,
        teacher_manifest_SHA256=digest(teacher_file),manifest_SHA256=digest(manifest),source_SHA256=digest(__file__),
        scope='CORRECTNESS_ONLY. Uses recorded teacher batch geometry. Not an online selector or serving policy.'))
    rpc(args.url,'encoder_capture/begin',dict(name=args.run))
    results=[]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=1,keepalive_timeout=1)) as session:
            for group in plan:
                row=rows[group['request_index']];full=messages(row,args.run)[0]['content']
                parts=[full[v['index']] for v in group['items']]+[full[-1]]
                body=dict(model=MODEL_ID,messages=[dict(role='user',content=parts)],temperature=0,max_tokens=1,stream=False)
                begin=time.monotonic_ns()
                async with session.post(args.url+'/v1/chat/completions',json=body,
                    headers={'x-request-id':args.run+':call:'+str(group['call_id'])}) as reply:
                    response=await reply.json();assert reply.status==200,(reply.status,response)
                append(ROOT/'raw'/(args.run+'_http.jsonl'),dict(call=group['call_id'],begin_ns=begin,
                    end_ns=time.monotonic_ns(),image_indices=[r['index'] for r in group['items']],response=response))
                for item in group['items']:
                    identifier=hashlib.sha256((args.run+row['request_id']+str(item['index'])).encode()).hexdigest()
                    export=args.ec_path/identifier/'encoder_cache.safetensors'
                    a=load_file(export)['ec_cache'];b=load_file(item['reference']['file'])['embedding']
                    metric=embedding_metrics(a,b)
                    results.append(dict(request_id=row['request_id'],image_index=item['index'],call=group['call_id'],
                        teacher_file=item['reference']['file'],export_file=str(export),**metric))
                print('MATCHED_GEOMETRY',group['request_index'],group['call_id'],len(group['items']),
                    all(r['strict'] for r in results),flush=True)
    finally:
        rpc(args.url,'encoder_capture/end',{})
    captured=json.loads((ROOT/'raw/encoder_reference'/args.run/'manifest.json').read_text())['rows']
    expected={item['key']:item['reference'] for group in plan for item in group['items']}
    actual_geometry=all(r['images_in_call']==expected[r['input_pixel_SHA256']]['images_in_call']
        and r['patches_in_call']==expected[r['input_pixel_SHA256']]['patches_in_call'] for r in captured)
    value=dict(time=time.time(),images=results,actual_geometry_equal=actual_geometry,
        passed=len(captured)==len(results)==sum(len(group['items']) for group in plan) and actual_geometry and all(r['strict'] for r in results),
        evidence='CORRECTNESS_ONLY',scope='Actual original BF16 encoder and exact teacher geometry. No speedup or online policy claim.')
    write(result_file,value);preserve();print('GEOMETRY_STRICT_PASS',value['passed'],len(results),flush=True)
    assert value['passed'],'Same-geometry encoder strict equality failed'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--ec-path',type=Path,required=True)
    p.add_argument('--teacher',default='colocated_calibration32_geometry_v3')
    p.add_argument('--indices',default='0,1,4,6,12,17,20,31');p.add_argument('--url',default='http://127.0.0.1:18400')
    asyncio.run(replay(p.parse_args()))


if __name__=='__main__':main()
