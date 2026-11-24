"""Capture actual E/colocated inputs, weights and outputs for a matched image loop."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from .common import ROOT,write,digest,preserve
from .numerical import embedding_metrics


def rpc(url,path,body=None):
    request=urllib.request.Request(url+'/campaign/'+path,
        data=json.dumps(body or {}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)


def main():
    from safetensors.torch import load_file
    from .client import execute
    from .encoder_only_probe import run as encoder_run
    import torch
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--encoder-url',default='http://127.0.0.1:18400')
    parser.add_argument('--reference-url',default='http://127.0.0.1:18303')
    parser.add_argument('--ec-path',type=Path,required=True)
    parser.add_argument('--indices',default='0');parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--whole-request',action='store_true');args=parser.parse_args()
    indices=list(map(int,args.indices.split(',')))
    name=args.run;result_file=ROOT/'analysis'/(name+'.json')
    assert not result_file.exists(),'Do not overwrite a control'
    manifest=ROOT/'data/calibration_manifest_template_v2.json'
    rows=json.loads(manifest.read_text())['requests'];selected=[rows[i] for i in indices]
    write(ROOT/'configs'/(name+'.json'),dict(time=time.time(),indices=indices,
        manifest_SHA256=digest(manifest),source_SHA256=digest(__file__),
        evidence='CORRECTNESS_ONLY',encoder=args.encoder_url,reference=args.reference_url,
        frozen_gate=dict(relative_L2_max=.02,cosine_min=.999)))
    roles=[('E',args.encoder_url)] if args.export_only else [('E',args.encoder_url),('R',args.reference_url)]
    for role,url in roles:
        rpc(url,'encoder_capture/begin',dict(name=name+'_'+role))
        if role=='E':asyncio.run(encoder_run(url,name+'_E',indices,args.whole_request))
        else:asyncio.run(execute(selected,[url],name+'_R',.25,2026112402,1,serial=True))
        end=rpc(url,'encoder_capture/end')
        write(ROOT/'state'/(name+'_'+role+'_capture.json'),end)
        fingerprint=rpc(url,'vision_fingerprint')
        write(ROOT/'raw/encoder_reference'/(name+'_'+role)/'weights.json',fingerprint)
    if args.export_only:
        write(result_file,dict(time=time.time(),indices=indices,status='EXPORTED_NO_COMPARISON',
            evidence='CORRECTNESS_ONLY',scope='Actual E inputs and outputs captured. Run encoder_compare separately.'))
        preserve();print('ENCODER_CAPTURE_EXPORTED',name,len(indices),flush=True);return
    left=json.loads((ROOT/'raw/encoder_reference'/(name+'_E')/'manifest.json').read_text())['rows']
    right=json.loads((ROOT/'raw/encoder_reference'/(name+'_R')/'manifest.json').read_text())['rows']
    mapping={r['input_pixel_SHA256']:r for r in right}
    assert len(mapping)==len(right),'Use disjoint image inputs for this minimal control'
    results=[]
    for item in left:
        reference=mapping.get(item['input_pixel_SHA256'])
        assert reference is not None,('Encoder and colocated actual pixel bytes differ',item)
        a=load_file(item['file'])['embedding'];b=load_file(reference['file'])['embedding']
        metric=embedding_metrics(a,b)
        results.append(dict(E=item,R=reference,**metric,
            passed=metric['finite'] and metric['relative_L2']<=.02 and metric['cosine']>=.999))
    e=json.loads((ROOT/'raw/encoder_reference'/(name+'_E')/'weights.json').read_text())[0]
    r=json.loads((ROOT/'raw/encoder_reference'/(name+'_R')/'weights.json').read_text())[0]
    value=dict(time=time.time(),images=results,passed=bool(results) and all(v['passed'] for v in results),
        actual_pixel_hashes_match=len(left)==len(right)==len(results),
        vision_parameters_equal=e['vision_parameters']==r['vision_parameters'],
        vision_buffers_equal=e['vision_buffers']==r['vision_buffers'],
        E_runner=e['runner_type'],R_runner=r['runner_type'],
        E_backend=e['vision_backend'],R_backend=r['vision_backend'],
        evidence='CORRECTNESS_ONLY',scope='Actual original model passthrough. No output correction or performance claim.')
    write(result_file,value);preserve()
    print('ENCODER_MATCHED_CONTROL',json.dumps({k:v for k,v in value.items() if k!='images'}),flush=True)


if __name__=='__main__':main()
