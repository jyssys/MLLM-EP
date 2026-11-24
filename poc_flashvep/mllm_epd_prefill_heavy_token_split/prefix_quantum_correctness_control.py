"""Original real-embedding prefix quanta on E and D, with matched teachers."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from .client import payload
from .common import ROOT,REPO,PYTHON,digest,preserve,write,experiment_deadline
from .encoder_llm_memory_control import info
from .frontend_supply_control import wait_ready,retire
from .ipc_ec_control import stop_proxy,ownership_empty
from .llm_probe_control import rpc
from .prefix_probe_contract import KEY
from .source_snapshot import snapshot
from .token_split import TEACHER_KEY
from .token_correctness_control import launch_proxy

TAG='prefix_QUANTA_C22'
EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
E=TAG+'_E4';HE=TAG+'_LE4';HD=TAG+'_LD6'


async def request(row,run,url,quantum,teacher):
    import aiohttp
    body=payload(row,run,1,False)
    body['kv_transfer_params']={(TEACHER_KEY if teacher else KEY):dict(
        prefix_tokens=quantum,full_prompt_tokens=row['total_input_tokens'])}
    config=ROOT/'configs'/f'{run}.json';assert not config.exists()
    write(config,dict(request_id=row['request_id'],prompt_SHA256=row['prompt_SHA256'],
        input_tokens_original=row['total_input_tokens'],quantum=quantum,teacher=teacher,
        payload=body,task_completion=False,evidence='CORRECTNESS_ONLY'))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
        begin=time.monotonic()
        async with session.post(url+'/v1/chat/completions',json=body,headers={'x-request-id':run}) as response:
            output=await response.json();assert response.status==200,output
        end=time.monotonic()
    expected=row['total_input_tokens'] if teacher else quantum
    assert output['usage']['prompt_tokens']==expected and output['usage']['completion_tokens']==1
    write(ROOT/'raw'/f'{run}.json',dict(output=output,task_completion=False,
        HTTP_begin=begin,HTTP_end=end,HTTP_wall_ms=(end-begin)*1000,
        scope='Discarded one-token prefix sample. Not a task answer or throughput result.'))
    ownership_empty(run,'cuda_ipc')
    return dict(run=run,quantum=quantum,HTTP_wall_ms=(end-begin)*1000,
        begin=begin,end=end,prompt_tokens=expected,output_tokens=1,task_completion=False)


def captured(row,run,url,port,quantum,teacher):
    rpc(port,'llm_probe/begin',dict(name=run,max_logits=1,include_decode_inputs=True))
    try:asyncio.run(request(row,run,url,quantum,teacher))
    finally:write(ROOT/'state'/f'{run}_probe_close.json',rpc(port,'llm_probe/end',{}))


def compare(teacher_run,probe_run,quantum):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    records=[];values=[]
    for run in [teacher_run,probe_run]:
        record=json.loads((ROOT/'raw/llm_reference'/run/'manifest.json').read_text())
        assert digest(record['tensor_file'])==record['tensor_SHA256']
        records.append(record);values.append(load_file(record['tensor_file']))
    teacher,probe=values
    first=records[0]['events'][0];actual=records[1]['events'][0]
    assert first['start']==actual['start']==0 and first['query_tokens']==actual['query_tokens']==quantum
    fields=dict(prompt=torch.equal(teacher['prompt_token_ids'][:quantum],probe['prompt_token_ids']),
        MRoPE=torch.equal(teacher['full_prompt_mrope'][:,:quantum],probe['full_prompt_mrope']),
        prepared_embeddings=torch.equal(teacher['inputs_embeds_0'],probe['inputs_embeds_0']),
        positions=torch.equal(teacher['positions_0'],probe['positions_0']))
    layers=[dict(layer=i,strict=torch.equal(teacher[f'prompt_KV_{i}'][:quantum],probe[f'prompt_KV_{i}'])) for i in range(28)]
    result=dict(teacher=teacher_run,probe=probe_run,quantum=quantum,fields=fields,layers=layers,
        evidence='CORRECTNESS_ONLY',task_completion=False,observer='CPU tensor copies invalidate every timing',
        strict_pass=all(fields.values()) and all(r['strict'] for r in layers))
    write(ROOT/'analysis'/f'{probe_run}_comparison.json',result)
    assert result['strict_pass'],'The isolated helper prefix differs from its same-GPU matched full teacher'
    print('REAL_PREFIX_QUANTUM_CORRECTNESS_PASS',probe_run,quantum,flush=True)
    return result


def main():
    import argparse
    global TAG,EC,E,HE,HD
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default=TAG)
    parser.add_argument('--wait-for-tag')
    parser.add_argument('--measure-quanta',action='store_true')
    parser.add_argument('--cost-arm',choices=['both','E','D'],default='both')
    options=parser.parse_args()
    assert options.tag.isidentifier() and options.tag.startswith(('prefix_QUANTA_C22','prefix_QUANTA_C26'))
    if options.wait_for_tag:assert options.wait_for_tag.isidentifier()
    TAG=options.tag;EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
    E=TAG+'_E4';HE=TAG+'_LE4';HD=TAG+'_LD6'
    preserve();source=snapshot(TAG)
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists() and not EC.exists()
    manifest=ROOT/'data/calibration_manifest_template_v2.json'
    row=json.loads(manifest.read_text())['requests'][6]
    write(plan,dict(time=time.time(),source_archive_SHA256=source['archive_SHA256'],
        manifest_SHA256=digest(manifest),request_id=row['request_id'],quantums=[128,256,512,1024,2048],
        actual_input_tokens=row['total_input_tokens'],images=row['image_count'],GPU_group=[4,6],
        encoder_GPU=4,E_helper_GPU=4,D_helper_GPU=6,helper_KV_GiB=8,B=8192,S=1024,
        embedding_transport='Original E encodes all images. On-demand descriptors cover only scheduled consumers.',
        inline_descriptors=False,clone_complete_ACK_required=True,
        isolated_cost_measurement_requested=options.measure_quanta,cost_arm=options.cost_arm,wait_for_tag=options.wait_for_tag,
        evidence='CORRECTNESS_ONLY',task_completion=False,
        cost_timing='PLANNED_SEPARATE_CPC' if options.measure_quanta else 'NOT_RUN',
        scope='Original BF16 compiled helpers use real original E embeddings. Prefix quanta have no imported prefix KV. '
            'This is not suffix continuation, online insertion, a full task answer or serving improvement.'))
    if options.wait_for_tag:
        while True:
            files=[p for p in (ROOT/'state').glob(f'server_{options.wait_for_tag}_*.json')
                if not p.name.endswith('.source_snapshot.json')]
            leases=[json.loads(p.read_text()) for p in files]
            assert len(leases)==3,'Expected three exact preceding owned leases'
            if all(r['finished'] and not Path('/proc',str(r['guardian'])).exists() for r in leases):break
            for r in leases:
                proc=Path('/proc',str(r['guardian']))
                if not r['finished']:assert proc.exists() and proc.stat().st_uid==os.getuid()
            assert time.time()<experiment_deadline();time.sleep(1)
    while True:
        leases=[json.loads((ROOT/'state'/f'server_token_NUMERICAL_C24_{role}{gpu}.json').read_text())
                for role,gpu in [('E',4),('P',5),('D',6)]]
        if all(r['finished'] and not Path('/proc',str(r['guardian'])).exists() for r in leases):break
        for r in leases:
            proc=Path('/proc',str(r['guardian']))
            if not r['finished']:assert proc.exists() and proc.stat().st_uid==os.getuid()
        assert time.time()<experiment_deadline();time.sleep(1)
    names=[];proxies=[];results=[]
    try:
        args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
            '--name',E,'--gpu','4','--role','encoder','--port','18840','--side-port','19840',
            '--ec-path',str(EC),'--observe','--ec-transport','cuda_ipc','--compiled-encoder',
            '--encoder-native-ops','--encoder-sync-scheduling','--mm-cache-gb','0','--renderer-workers','2']
        names.append(E);subprocess.run(args,cwd=REPO,check=True);wait_ready([E])
        for name,gpu,port in [(HE,4,18841),(HD,6,18842)]:
            args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
                '--name',name,'--gpu',str(gpu),'--role','pd','--port',str(port),
                '--side-port',str(port+1000),'--ec-path',str(EC),'--observe','--ec-transport','cuda_ipc',
                '--kv-cache-gib','8','--gpu-memory-utilization','.35','--prefix-probe']
            if name==HE:args+=['--co-resident-with',E]
            names.append(name);subprocess.run(args,cwd=REPO,check=True);wait_ready([name])
            rpc(port,'ec_ipc/ack_policy',dict(mode='file'))
            rpc(port,'boundary_teacher/install',{})
        # A cropped probe can leave later images unused. Never issue a descriptor
        # for those images, but keep every issued descriptor's clone ACK gate.
        rpc(18840,'ec_ipc/inline_policy',dict(enabled=False,consumer_role='pd'))
        memory={str(p):info(p) for p in [18840,18841,18842]}
        for arm,port,proxy_port in [('E',18841,18848),('D',18842,18849)]:
            proxy,url=launch_proxy(TAG+'_'+arm+'_proxy','e_pd',proxy_port,names,decoder_port=port)
            proxies.append(proxy)
            for quantum in [128,256,512,1024,2048]:
                assert time.time()<experiment_deadline()
                teacher=f'{TAG}_{arm}_q{quantum}_TEACHER';probe=f'{TAG}_{arm}_q{quantum}_PROBE'
                captured(row,teacher,url,port,quantum,True)
                captured(row,probe,url,port,quantum,False)
                results.append(compare(teacher,probe,quantum))
            if options.measure_quanta and options.cost_arm in ('both',arm):
                from .prefix_quantum_cost import measure
                measure(TAG,arm,row,url,port,18840)
            stop_proxy(proxy);proxies.remove(proxy)
        write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='CORRECTNESS_PASS',
            cases=len(results),strict_cases=sum(r['strict_pass'] for r in results),memory=memory,
            scope='Matched same-GPU real prefix compute only. No suffix KV import, task quality, timing or online safety claim.',
            prefix_quantum_cost='SEPARATE_CPC_RESULT' if options.measure_quanta else 'NOT_MEASURED',
            online_interference='NOT_RUN'))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_cases=len(results)))
        raise
    finally:
        for proxy in proxies:stop_proxy(proxy)
        for group in [[HE,HD],[E]]:
            owned=[name for name in group if name in names and (ROOT/'state'/f'server_{name}.json').exists()]
            if owned:retire(owned)
    preserve()


if __name__=='__main__':main()
