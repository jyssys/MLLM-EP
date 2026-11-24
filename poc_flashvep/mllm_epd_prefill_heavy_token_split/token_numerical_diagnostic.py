"""Locate strict suffix-KV differences before any token performance run."""
import json
from pathlib import Path
import time

from .common import ROOT,digest,preserve,write,experiment_deadline
from .frontend_supply_control import retire
from .ipc_ec_control import stop_proxy
from .llm_probe_control import rpc
from . import token_correctness_control as control
from .source_snapshot import snapshot

TAG='token_NUMERICAL_C17'


def analyze(results):
    import torch
    from safetensors.torch import load_file
    from .numerical import embedding_metrics
    torch.set_num_threads(2)
    data={};records={}
    for label,run,role in results:
        record=json.loads((ROOT/'raw/llm_reference'/f'{run}_{role}/manifest.json').read_text())
        assert digest(record['tensor_file'])==record['tensor_SHA256']
        data[label]=load_file(record['tensor_file']);records[label]=record
    comparisons=[]
    reference=data['P_A']
    for label in ['P_B','D_teacher','D_teacher_B','Token_D']:
        if label not in data:continue
        actual=data[label];kv=[]
        for layer in range(28):
            name=f'prompt_KV_{layer}'
            kv.append(dict(layer=layer,**embedding_metrics(actual[name],reference[name])))
        suffix=next(e for e in records[label]['events'] if e['start']==2415)
        target=next(e for e in records['P_A']['events'] if e['start']==2415)
        attention=[]
        for name in ['query','key','value','output','query_start_loc','seq_lens',
                     'scheduler_metadata','block_table','slot_mapping']:
            key=f'attention_0_{name}_{suffix["index"]}'
            ref=f'attention_0_{name}_{target["index"]}'
            if key in actual and ref in reference:
                attention.append(dict(field=name,actual_shape=list(actual[key].shape),
                    reference_shape=list(reference[ref].shape),strict=torch.equal(actual[key],reference[ref]),
                    numerical=embedding_metrics(actual[key],reference[ref])
                        if actual[key].is_floating_point() else None))
        comparisons.append(dict(comparison=label+'_vs_P_A',KV=kv,first_layer_attention=attention,
            actual_attention=suffix.get('attention_0'),reference_attention=target.get('attention_0')))
    direct=[]
    for layer in range(28):
        key=f'prompt_KV_{layer}'
        direct.append(dict(layer=layer,
            D_teacher_token_strict=torch.equal(data['D_teacher'][key],data['Token_D'][key]),
            P_to_D_prefix_strict=torch.equal(data['Token_P'][key],data['Token_D'][key][:2415])))
    assert all(r['P_to_D_prefix_strict'] for r in direct),'Imported P prefix changed'
    return dict(comparisons=comparisons,direct_checks=direct,
        greedy_tokens={label:json.loads((ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines()[0])['token_ids']
            for label,run,role in results if label!='Token_P'},
        scope='Read-only first-layer Q/K/V/output and attention metadata. Original forward passthrough. '
            'Strict failures remain failures. No tolerance or task-quality gate changes.')


def main():
    import argparse
    global TAG
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default=TAG)
    args=parser.parse_args()
    assert args.tag.isidentifier() and args.tag.startswith('token_NUMERICAL_C17')
    TAG=args.tag
    preserve()
    source=snapshot(TAG)
    path=ROOT/'configs'/f'{TAG}.json'
    assert not path.exists()
    write(path,dict(time=time.time(),status='FROZEN_BEFORE_DIAGNOSTIC_RESULTS',
        source_archive_SHA256=source['archive_SHA256'],index=0,prefix_tokens=2415,
        hypotheses=['Attention metadata or backend scheduling differs for remote versus locally produced prefix.',
            'Cross-GPU or repeated original BI0 execution differs despite matched query rows.',
            'NIXL readiness or prefix lifetime changes the continuation input.'],
        arms=['Same-P teacher A','Same-P teacher B','Same-D complete teacher A',
              'Same-D complete teacher B','Actual P-prefix/D-suffix'],
        output='Original Natural EOS cap2048',evidence='CORRECTNESS_ONLY and DIAGNOSTIC',
        no_performance=True,no_active_decode_insertion=True,
        prior_failure='token_CORRECTNESS_C15R1_N1_inside_single_image_TOKEN_comparison.json'))
    while True:
        active=[]
        for name in ['encoder_LLM_MEMORY_C16_E4','encoder_LLM_MEMORY_C16_LLM4']:
            lease=ROOT/'state'/f'server_{name}.json'
            if not lease.exists():active.append(name);continue
            state=json.loads(lease.read_text())
            if not state['finished'] or Path('/proc',str(state['guardian'])).exists():active.append(name)
        if not active:break
        assert time.time()<experiment_deadline()
        time.sleep(1)
    control.TAG=TAG
    control.EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
    names=[];proxies=[];results=[]
    row=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests'][0]
    try:
        control.wait_baselines()
        control.launch_models(names)
        for label,port in [('P_A',18841),('P_B',18841),('D_teacher',18842),('D_teacher_B',18842)]:
            role='P' if port==18841 else 'D'
            rpc(18840,'ec_ipc/inline_policy',dict(enabled=True,consumer_role='prefill' if role=='P' else 'decode'))
            run=TAG+'_'+label
            proxy,url=control.launch_proxy(run+'_proxy','e_pd',18848,names,decoder_port=port)
            proxies.append(proxy)
            try:control.capture(row,run,url,[(port,role)],2415,include_attention=True)
            finally:stop_proxy(proxy);proxies.remove(proxy)
            results.append((label,run,role))
        rpc(18840,'ec_ipc/inline_policy',dict(enabled=True,consumer_role='prefill'))
        run=TAG+'_TOKEN'
        proxy,url=control.launch_proxy(run+'_proxy','epd',18849,names,2415)
        proxies.append(proxy)
        try:control.capture(row,run,url,[(18841,'P'),(18842,'D')],include_attention=True)
        finally:stop_proxy(proxy);proxies.remove(proxy)
        results.extend([('Token_P',run,'P'),('Token_D',run,'D')])
        value=analyze(results)
        write(ROOT/'analysis'/f'{TAG}.json',dict(time=time.time(),evidence='DIAGNOSTIC',**value))
        print('TOKEN_NUMERICAL_DIAGNOSTIC_RECORDED',flush=True)
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_arms=results))
        raise
    finally:
        for proxy in proxies:stop_proxy(proxy)
        if names:
            retire([name for name in names if '_E4' not in name])
            retire([name for name in names if '_E4' in name])
    preserve()


if __name__=='__main__':main()
