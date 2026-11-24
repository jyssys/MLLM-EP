"""Actual E/LLM memory co-residency. No extra prefill or serving-gain claim."""
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .client import payload, resolve_manifest
from .common import ROOT, REPO, PYTHON, digest, preserve, write
from .frontend_supply_control import wait_ready, retire
from .llm_probe_control import rpc
from .source_snapshot import snapshot

TAG='encoder_LLM_MEMORY_C16'
EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
E=TAG+'_E4'
LLM=TAG+'_LLM4'


def info(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/campaign/worker_info',timeout=120) as response:
        value=json.load(response)
    assert len(value)==1
    return value[0]


def compare_encoder_captures():
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    captures=[]
    for phase in ['before','with_llm']:
        path=ROOT/'raw/encoder_reference'/f'{TAG}_{phase}'/'manifest.json'
        captures.append(json.loads(path.read_text())['rows'])
    before,after=captures
    assert len(before)==len(after) and before
    rows=[]
    for a,b in zip(before,after):
        assert a['input_pixel_SHA256']==b['input_pixel_SHA256'] and a['grid']==b['grid']
        x,y=load_file(a['file'])['embedding'],load_file(b['file'])['embedding']
        rows.append(dict(pixel_SHA256=a['input_pixel_SHA256'],shape=list(x.shape),
            strict=torch.equal(x,y),finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            before_file=a['file'],after_file=b['file']))
    assert all(r['strict'] and r['finite'] for r in rows),'Co-residency changed an original encoder result'
    return rows


async def encode(rows,label):
    import aiohttp
    results=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
        for index,row in enumerate(rows):
            run=f'{TAG}_{label}_{index}'
            key=hashlib.sha256(run.encode()).hexdigest()
            request=payload(row,run,1,False)
            headers={'x-request-id':run,'x-campaign-metadata-key':key,'x-campaign-metadata-mode':'grid_only'}
            async with session.post('http://127.0.0.1:18840/v1/chat/completions',json=request,headers=headers) as response:
                output=await response.json()
                assert response.status==200,output
            assert output['usage']['prompt_tokens']==row['total_input_tokens']
            barrier=rpc(18840,'cuda_barrier',{})
            memory=info(18840)
            async with session.post('http://127.0.0.1:18840/campaign/ec_ipc/release',json=dict(key=key)) as response:
                released=await response.json()
                assert response.status==200 and released[0]['live_images']==released[0]['outstanding_exports']==0
            results.append(dict(request_id=row['request_id'],input_tokens=row['total_input_tokens'],
                image_count=row['image_count'],output=output,barrier=barrier,memory=memory,
                release=released[0],metadata_SHA256=digest(EC/'processed_prompts'/key/'manifest.json')))
    return results


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--after-token-tag',default='token_CORRECTNESS_C15R1')
    args=parser.parse_args()
    assert args.after_token_tag.isidentifier() and args.after_token_tag.startswith('token_CORRECTNESS_C15')
    preserve()
    for role,gpu in [('E',4),('P',5),('D',6)]:
        lease=json.loads((ROOT/'state'/f'server_{args.after_token_tag}_{role}{gpu}.json').read_text())
        assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists(), 'Wait for the same owned correctness leases to close'
    path=resolve_manifest(None)
    rows=json.loads(path.read_text())['requests']
    choices=[max(rows,key=lambda r:r['total_input_tokens']),max(rows,key=lambda r:r['image_count']),
             max(rows,key=lambda r:max(i['visual_tokens'] for i in r['images']))]
    selected=list({r['request_id']:r for r in choices}.values())
    plan=ROOT/'configs'/f'{TAG}.json'
    assert not plan.exists() and not EC.exists()
    source=snapshot(TAG)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_CO_RESIDENCY_RESULTS',
        source_archive_SHA256=source['archive_SHA256'],manifest_SHA256=digest(path),
        selected_requests=[r['request_id'] for r in selected],
        selection='Predefined maximum input, image count and single-image visual-token memory stresses. Not performance-based.',
        GPU=4,helper_KV_GiB=8,helper_gpu_memory_utilization=.35,
        original_encoder='Unchanged BF16 model/processor, cache0/T2/B114688',
        additional_engine='Original full Qwen2.5-VL BF16 model. Includes an explicitly counted duplicate vision tower.',
        evidence='ISOLATED_MEMORY_MEASUREMENT and CORRECTNESS_ONLY',
        no_extra_prefill=True,no_online_interference_probe=True,
        limitations='Observed memory fit is not maximum encoder-batch capacity, encode-safe headroom or serving throughput.'))
    names=[]
    try:
        args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
            '--name',E,'--gpu','4','--role','encoder','--port','18840','--side-port','19840',
            '--ec-path',str(EC),'--observe','--ec-transport','cuda_ipc',
            '--compiled-encoder','--encoder-native-ops','--encoder-sync-scheduling',
            '--mm-cache-gb','0','--renderer-workers','2']
        subprocess.run(args,cwd=REPO,check=True);names.append(E);wait_ready([E])
        before=info(18840)
        rpc(18840,'encoder_capture/begin',dict(name=TAG+'_before'))
        try:warm=asyncio.run(encode(selected,'before'))
        finally:rpc(18840,'encoder_capture/end',{})
        after_encode=info(18840)
        args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
            '--name',LLM,'--gpu','4','--role','pd','--port','18841','--side-port','19841',
            '--ec-path',str(EC),'--observe','--ec-transport','cuda_ipc','--co-resident-with',E,
            '--kv-cache-gib','8','--gpu-memory-utilization','.35']
        subprocess.run(args,cwd=REPO,check=True);names.append(LLM);wait_ready([LLM])
        with_llm_E,with_llm_LLM=info(18840),info(18841)
        assert with_llm_LLM['model_parameter_bytes_by_prefix'].get('language_model',0)>0
        rpc(18840,'encoder_capture/begin',dict(name=TAG+'_with_llm'))
        try:after=asyncio.run(encode(selected,'with_llm'))
        finally:rpc(18840,'encoder_capture/end',{})
        feature_comparison=compare_encoder_captures()
        result=dict(time=time.time(),status='OBSERVED_CO_RESIDENCY_FIT',GPU=4,
            evidence='ISOLATED_MEMORY_MEASUREMENT',initial_E=before,E_after_encode=after_encode,
            with_LLM_E=with_llm_E,additional_full_model=with_llm_LLM,
            before_requests=warm,after_requests=after,final_E=info(18840),final_LLM=info(18841),
            encoder_feature_comparison=feature_comparison,
            additional_weights_include_duplicate_vision=True,
            language_model_parameter_bytes=with_llm_LLM['model_parameter_bytes_by_prefix']['language_model'],
            helper_KV_storage_bytes=with_llm_LLM['unique_KV_storage_bytes'],
            encode_safe_headroom='NOT_MEASURED',additional_prefill_quantum='NOT_RUN',
            exclusive_runtime_workspace='NOT_MEASURED',maximum_encoder_batch_capacity='NOT_MEASURED',
            scope='Original E successfully encoded the predefined real stress requests before and after full model loading. '
                'The extra LLM did not process client prefill. No latency/SLO gate or serving speedup is established.')
        write(ROOT/'analysis'/f'{TAG}.json',result)
        print('E_LLM_MEMORY_OBSERVED_FIT',result['language_model_parameter_bytes'],result['helper_KV_storage_bytes'],flush=True)
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error)))
        raise
    finally:
        if LLM in names:retire([LLM])
        if E in names:retire([E])
    preserve()


if __name__=='__main__':main()
