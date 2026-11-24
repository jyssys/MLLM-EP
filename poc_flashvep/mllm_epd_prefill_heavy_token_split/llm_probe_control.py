"""Paired original-model LLM/KV observation for one public calibration request."""
import argparse
import asyncio
import json
import time
import urllib.request

from .common import ROOT,write,digest,preserve
from .client import execute


def rpc(port,path,body):
    request=urllib.request.Request(f'http://127.0.0.1:{port}/campaign/'+path,
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--index',type=int,default=6);args=parser.parse_args()
    manifest=ROOT/'data/calibration_manifest_template_v2.json'
    row=json.loads(manifest.read_text())['requests'][args.index]
    write(ROOT/'configs'/(args.run+'.json'),dict(time=time.time(),request_id=row['request_id'],index=args.index,
        manifest_SHA256=digest(manifest),source_SHA256=digest(__file__),scope='CORRECTNESS_ONLY. No timing claim.'))
    for arm,ports,url in [('Native',[(18401,'P'),(18402,'D')],'http://127.0.0.1:18403'),
                         ('OfficialPD',[(18604,'P'),(18605,'D')],'http://127.0.0.1:18603')]:
        active=[]
        try:
            for port,role in ports:
                value=rpc(port,'llm_probe/begin',dict(name=args.run+'_'+arm+'_'+role,max_logits=16))
                active.append((port,role));print('PROBE_BEGIN',arm,role,value,flush=True)
            results=asyncio.run(execute([row],[url],args.run+'_'+arm,.25,2026112402,None,True,
                metadata=dict(manifest=str(manifest),index=args.index,evidence='CORRECTNESS_ONLY',
                    scope='Full prompt KV and logits capture. Extra CPU copies exclude all timings.')))
            print('PROBE_REQUEST',arm,results[0]['status'],results[0].get('output_tokens'),flush=True)
        finally:
            for port,role in active:
                value=rpc(port,'llm_probe/end',{})
                print('PROBE_END',arm,role,value,flush=True)
                write(ROOT/'state'/(args.run+'_'+arm+'_'+role+'_close.json'),value)
    preserve()


if __name__=='__main__':main()
