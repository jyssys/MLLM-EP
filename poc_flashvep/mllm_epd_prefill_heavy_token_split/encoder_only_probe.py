"""Online encoder export control. Dummy E outputs are not model answers."""
import argparse
import asyncio
import json
import time
from .common import ROOT,MODEL_ID,write,append,digest
from .client import messages


async def run(url,name,indices,whole_request=False):
    import aiohttp
    rows=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200),
        connector=aiohttp.TCPConnector(limit=0,keepalive_timeout=1)) as session:
        for index in indices:
            row=rows[index];parts=messages(row,name)[0]['content'][:-1]
            if whole_request:
                payload=dict(model=MODEL_ID,messages=messages(row,name),temperature=0,max_tokens=1,stream=False)
                begin=time.monotonic_ns()
                async with session.post(url+'/v1/chat/completions',json=payload,
                    headers={'x-request-id':name+'_'+str(index)+'_E_full'}) as reply:
                    output=await reply.json();assert reply.status==200,(reply.status,output)
                append(ROOT/'raw'/(name+'_encoder.jsonl'),dict(index=index,request_id=row['request_id'],
                    whole_request=True,image_count=len(parts),UUIDs=[p['uuid'] for p in parts],
                    begin_ns=begin,end_ns=time.monotonic_ns(),response=output,
                    scope='CORRECTNESS_ONLY. Original full multimodal request to encoder-only worker. Dummy E answer is not a model answer.'))
                print('ENCODER_WHOLE_EXPORT',name,index,len(parts),flush=True)
                continue
            async def one(i,part):
                request=dict(model=MODEL_ID,messages=[dict(role='user',content=[part])],
                    temperature=0,max_tokens=1,stream=False)
                begin=time.monotonic_ns()
                async with session.post(url+'/v1/chat/completions',json=request,
                    headers={'x-request-id':name+'_'+str(index)+'_E_'+str(i)}) as reply:
                    output=await reply.json();assert reply.status==200,(reply.status,output)
                return dict(image_index=i,UUID=part['uuid'],begin_ns=begin,end_ns=time.monotonic_ns(),response=output)
            outputs=await asyncio.gather(*(one(i,p) for i,p in enumerate(parts)))
            append(ROOT/'raw'/(name+'_encoder.jsonl'),dict(index=index,request_id=row['request_id'],images=outputs,
                scope='CORRECTNESS_ONLY. E-stage output tokens are dummy values,not generated answers.'))
            print('ENCODER_EXPORT',name,index,len(outputs),flush=True)
    write(ROOT/'state'/(name+'_complete.json'),dict(time=time.time(),passed=True,indices=indices,
        raw_SHA256=digest(ROOT/'raw'/(name+'_encoder.jsonl'))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18400');p.add_argument('--run',required=True)
    p.add_argument('--indices',default='0,1,4,6,12,17,20,31');p.add_argument('--whole-request',action='store_true');args=p.parse_args()
    asyncio.run(run(args.url,args.run,list(map(int,args.indices.split(','))),args.whole_request))


if __name__=='__main__':main()
