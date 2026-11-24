"""Change only an encoder batch envelope. Keep the downstream LLM query fixed."""
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .client import messages, one, payload
from .common import ROOT, REPO, PYTHON, digest, experiment_deadline, preserve, write
from .frontend_supply_control import wait_ready, retire
from .llm_probe_control import rpc
from .numerical import embedding_metrics
from .source_snapshot import snapshot
from .token_split import TEACHER_KEY

TAG='vision_BATCH_C38R1'
EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'


async def case(label,target,companion,template=None):
    import aiohttp
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from .processor_handoff import RenderStore
    from .vision_batch_handoff import prepare,Session
    nonce=TAG+'_'+label;key=hashlib.sha256(nonce.encode()).hexdigest()
    encoder_body=payload(target,TAG+'_TARGET',stream=False);encoder_body['max_tokens']=1
    if companion is not None:encoder_body['messages']+=messages(companion,TAG+'_COMPANION')
    headers={'x-request-id':nonce+':E','x-campaign-metadata-key':key,'x-campaign-metadata-mode':'full'}
    e_capture=TAG+'_'+label+'_E'
    rpc(18840,'encoder_capture/begin',dict(name=e_capture))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
            async with session.post('http://127.0.0.1:18840/v1/chat/completions',json=encoder_body,headers=headers) as response:
                encoded=await response.json();assert response.status==200,encoded
    finally:rpc(18840,'encoder_capture/end',{})
    store=RenderStore(EC/'processed_prompts')
    encoder_rendered=store.load(key,ChatCompletionRequest(**encoder_body),arrival_time=time.time())
    if template is None:
        assert label=='Solo_A' and companion is None
        template=encoder_rendered
    body=payload(target,TAG+'_TARGET')
    body['kv_transfer_params']={TEACHER_KEY:dict(prefix_tokens=2221,full_prompt_tokens=target['total_input_tokens'])}
    bridge_key=hashlib.sha256((nonce+'_target_handoff').encode()).hexdigest()
    consumer_headers=prepare(store,template,ChatCompletionRequest(**body),encoder_rendered,bridge_key,time.time())
    llm_run=TAG+'_'+label+'_LLM';probe_name=llm_run+'_P'
    rpc(18841,'llm_probe/begin',dict(name=probe_name,max_logits=16,include_decode_inputs=False))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
            result=await one(Session(session,consumer_headers),'http://127.0.0.1:18841',target,body,time.monotonic(),llm_run,0)
        assert result['status']=='MEASURED_ONLINE'
    finally:rpc(18841,'llm_probe/end',{})
    released=rpc(18840,'ec_ipc/release',dict(key=key))
    assert len(released)==1 and released[0]['live_images']==released[0]['outstanding_exports']==0
    return dict(label=label,E_capture=e_capture,LLM_capture=probe_name,output=result,
        encoder_usage=encoded['usage'],release=released,
        target_handoff_SHA256=digest(store.directory(bridge_key)/'manifest.json'),
        scope='CORRECTNESS_ONLY. The combined envelope is an encoder batch diagnostic, not a Main workload request.'),template


def compare(cases):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    captures={}
    for case in cases:
        epath=ROOT/'raw/encoder_reference'/case['E_capture']/'manifest.json'
        ep=json.loads(epath.read_text())['rows']
        lpath=ROOT/'raw/llm_reference'/case['LLM_capture']/'manifest.json'
        lp=json.loads(lpath.read_text());assert digest(lp['tensor_file'])==lp['tensor_SHA256']
        captures[case['label']]=dict(images=ep,manifest=lp,tensors=load_file(lp['tensor_file']),
            E_manifest_SHA256=digest(epath),LLM_manifest_SHA256=digest(lpath))
    base=captures['Solo_A'];comparisons=[]
    for label in ['Solo_B','Joint_A','Joint_B']:
        other=captures[label];mapping={r['input_pixel_SHA256']:r for r in other['images']}
        images=[]
        for reference in base['images']:
            actual=mapping[reference['input_pixel_SHA256']]
            a=load_file(actual['file'])['embedding'];b=load_file(reference['file'])['embedding']
            images.append(dict(pixel_SHA256=reference['input_pixel_SHA256'],
                reference_images_in_call=reference['images_in_call'],actual_images_in_call=actual['images_in_call'],
                reference_patches_in_call=reference['patches_in_call'],actual_patches_in_call=actual['patches_in_call'],
                **embedding_metrics(a,b)))
        a,b=other['tensors'],base['tensors']
        assert torch.equal(a['prompt_token_ids'],b['prompt_token_ids'])
        assert torch.equal(a['full_prompt_mrope'],b['full_prompt_mrope'])
        assert [(e['start'],e['query_tokens']) for e in other['manifest']['events'] if e['start']<2961]==[(0,2221),(2221,740)]
        layers=[dict(layer=i,**embedding_metrics(a[f'prompt_KV_{i}'],b[f'prompt_KV_{i}'])) for i in range(28)]
        inputs=[]
        for event in [0,1]:
            key=f'inputs_embeds_{event}'
            if key in a and key in b:inputs.append(dict(event=event,**embedding_metrics(a[key],b[key])))
        comparisons.append(dict(label=label,images=images,prepared_inputs=inputs,layers=layers,
            output_tokens_same=cases[[c['label'] for c in cases].index(label)]['output']['token_ids']==cases[0]['output']['token_ids']))
    repeat=next(c for c in comparisons if c['label']=='Solo_B')
    stable=all(r['strict'] for r in repeat['images']) and all(r['strict'] for r in repeat['layers'])
    joint=next(c for c in comparisons if c['label']=='Joint_A')
    geometry_changed=all(r['actual_images_in_call']!=r['reference_images_in_call'] for r in joint['images'])
    changed=any(not r['strict'] for r in joint['images'])
    effect=stable and geometry_changed and changed and any(not r['strict'] for r in joint['layers'])
    proof=dict(time=time.time(),evidence='CORRECTNESS_ONLY and controlled numerical DIAGNOSTIC',
        identical_actual_pixel_hashes=True,solo_repeat_strict=stable,encoder_geometry_changed=geometry_changed,
        numerical_effect_supported=effect,comparisons=comparisons,
        task_quality='NOT_ESTABLISHED',online_speedup='NOT_RUN',
        scope='The same original target images and fixed LLM query chunks follow different encoder envelopes. '
            'This is not an encoder algorithm change, a serving policy, a correctness repair or a reclassification of C36.')
    write(ROOT/'analysis'/f'{TAG}_comparison.json',proof)
    print('VISION_BATCH_INTERVENTION',stable,geometry_changed,effect,flush=True)


def main():
    preserve();assert time.time()<experiment_deadline()
    for role,gpu in [('E',4),('P',5),('D',6)]:
        lease=json.loads((ROOT/'state'/f'server_token_FROZEN_C36_{role}{gpu}.json').read_text())
        assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()
    pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    assert pool[6]['total_input_tokens']==2961
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists() and not EC.exists()
    archive=snapshot(TAG)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_ENCODER_GEOMETRY_INTERVENTION',
        source_archive_SHA256=archive['archive_SHA256'],GPU_group=[4,5],unused_GPU6='Approved burner',
        target=pool[6]['request_id'],companion=pool[17]['request_id'],
        one_factor='Original encoder processes the target alone or in an envelope with original companion images.',
        fixed='Same target pixels, processor, weights, BF16, E configuration, P model, query chunks0:2221:2961 and natural sampling.',
        cases=['Solo_A','Solo_B','Joint_A','Joint_B'],
        scope='Diagnostic batch envelope only. Main100/Main400 and their prompts remain unchanged.',
        success='Same-geometry repeat is strict. Different original encoder geometry changes embedding and downstream KV state.',
        limitations='A reproduced numerical effect does not prove the exact unrecorded encoder geometry of C36. No timing claim.'))
    write(ROOT/'configs'/f'{TAG}_harness_correction.json',dict(
        prior_failed_run='vision_BATCH_C38',cause='P rerendered engine-local image identifiers instead of consuming original E metadata.',
        correction='Diagnostic-only target metadata retains original processor pixels, tokens and grids. '
            'Only export identifiers bind to the first target images in the actual E envelope. No output or KV correction.',
        prior_failure_preserved=True,Main_serving_path_changed=False))
    names=[];results=[]
    try:
        for role,gpu,port,side in [('encoder',4,18840,19840),('prefill',5,18841,19841)]:
            name=TAG+'_'+role[0].upper()+str(gpu);names.append(name)
            args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
                '--name',name,'--gpu',str(gpu),'--role',role,'--port',str(port),'--side-port',str(side),
                '--ec-path',str(EC),'--observe','--ec-transport','cuda_ipc']
            if role=='encoder':args+=['--encoder-native-ops','--compiled-encoder','--encoder-sync-scheduling','--mm-cache-gb','0','--renderer-workers','2']
            else:args+=['--token-split','--fixed-reduction','--deterministic-reductions']
            subprocess.run(args,cwd=REPO,check=True)
        wait_ready(names)
        rpc(18840,'ec_ipc/inline_policy',dict(enabled=False))
        rpc(18841,'ec_ipc/ack_policy',dict(mode='file'))
        template=None
        for label,companion in [('Solo_A',None),('Solo_B',None),('Joint_A',pool[17]),('Joint_B',pool[17])]:
            assert time.time()<experiment_deadline()
            result,template=asyncio.run(case(label,pool[6],companion,template))
            results.append(result)
        write(ROOT/'analysis'/f'{TAG}_cases.json',dict(time=time.time(),cases=results,evidence='CORRECTNESS_ONLY'))
        compare(results)
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_cases=len(results)))
        raise
    finally:
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])
    preserve()


if __name__=='__main__':main()
