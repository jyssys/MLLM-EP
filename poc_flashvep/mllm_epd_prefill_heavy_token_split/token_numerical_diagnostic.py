"""Locate strict suffix-KV differences before any token performance run."""
import json
import os
from pathlib import Path
import time

from .common import ROOT,digest,preserve,write,experiment_deadline
from .frontend_supply_control import retire
from .ipc_ec_control import stop_proxy
from .llm_probe_control import rpc
from . import token_correctness_control as control
from .source_snapshot import snapshot

TAG='token_NUMERICAL_C17'


def analyze(results,attention_layer=0):
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
        gemm=[]
        if target.get('first_layer_GEMM'):
            assert len(target['first_layer_GEMM'])==len(suffix.get('first_layer_GEMM',[]))==4
            for index,stage in enumerate(['attention_output_projection','post_attention_norm_then_gate_up',
                                          'SiLU_gate_then_down_projection','next_input_norm_then_QKV']):
                fields=[]
                for field in ['input','output']:
                    key=f'gemm_0_{field}_{suffix["index"]}_{index}'
                    ref=f'gemm_0_{field}_{target["index"]}_{index}'
                    fields.append(dict(field=field,**embedding_metrics(actual[key],reference[ref])))
                gemm.append(dict(stage=stage,actual=suffix['first_layer_GEMM'][index],
                    reference=target['first_layer_GEMM'][index],values=fields))
        comparisons[-1]['first_layer_GEMM']=gemm
        norm=[]
        for field in ['residual','weight']:
            key=f'norm_0_{field}_{suffix["index"]}'
            ref=f'norm_0_{field}_{target["index"]}'
            if key in actual and ref in reference:
                norm.append(dict(field=field,**embedding_metrics(actual[key],reference[ref])))
        comparisons[-1].update(norm_inputs=norm,actual_norm_runtime=suffix.get('norm_runtime'),
            reference_norm_runtime=target.get('norm_runtime'))
    direct=[]
    for layer in range(28):
        key=f'prompt_KV_{layer}'
        direct.append(dict(layer=layer,
            D_teacher_token_strict=torch.equal(data['D_teacher'][key],data['Token_D'][key]),
            P_to_D_prefix_strict=torch.equal(data['Token_P'][key],data['Token_D'][key][:2415])))
    assert all(r['P_to_D_prefix_strict'] for r in direct),'Imported P prefix changed'
    return dict(comparisons=comparisons,direct_checks=direct,captured_attention_layer=attention_layer,
        greedy_tokens={label:json.loads((ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines()[0])['token_ids']
            for label,run,role in results if label!='Token_P'},
        scope='Read-only selected-layer Q/K/V/output and attention metadata. Original forward passthrough. '
            'Strict failures remain failures. No tolerance or task-quality gate changes.')


def main():
    import argparse
    global TAG
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default=TAG)
    parser.add_argument('--eager',action='store_true',help='Correctness-only original eager P/D, never a serving arm')
    parser.add_argument('--gemm',action='store_true',help='Read-only first-layer compiled GEMM tensors, no performance claim')
    parser.add_argument('--fixed-reduction',action='store_true',help='Compiled-only one-factor dynamic_scale_rblock=False control')
    parser.add_argument('--deterministic-reductions',action='store_true')
    parser.add_argument('--attention-layer',type=int,choices=[0,1],default=0)
    parser.add_argument('--wait-for-tag',help='An exact prior E4/P5/D6 lease group that must close first')
    args=parser.parse_args()
    assert args.tag.isidentifier() and args.tag.startswith(('token_NUMERICAL_C17','token_NUMERICAL_C18','token_NUMERICAL_C21','token_NUMERICAL_C24','token_NUMERICAL_C28','token_NUMERICAL_C29'))
    assert not args.gemm or not args.eager,'This diagnostic targets the actual compiled GEMM call sites'
    assert not args.gemm or args.attention_layer==0
    assert not args.fixed_reduction or not args.eager
    assert not args.deterministic_reductions or not args.eager
    assert args.wait_for_tag is None or args.wait_for_tag.isidentifier()
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
        compiler=('Explicit original P/D eager reference' if args.eager else
            'Explicit common compiled numerical control, not stock policy' if args.fixed_reduction or args.deterministic_reductions else
            'Unchanged stock compiled P/D'),
        first_layer_GEMM_copies=args.gemm,wait_for_tag=args.wait_for_tag,
        captured_attention_layer=args.attention_layer,
        dynamic_scale_rblock=False if args.fixed_reduction else 'Original default',
        deterministic_reductions=args.deterministic_reductions,
        deterministic_combo_benchmark=False if args.deterministic_reductions else 'Original default',
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
    if args.wait_for_tag:
        while True:
            leases=[json.loads((ROOT/'state'/f'server_{args.wait_for_tag}_{role}{gpu}.json').read_text())
                for role,gpu in [('E',4),('P',5),('D',6)]]
            if all(r['finished'] and not Path('/proc',str(r['guardian'])).exists() for r in leases):break
            for r in leases:
                proc=Path('/proc',str(r['guardian']))
                if not r['finished']:
                    assert proc.exists() and proc.stat().st_uid==os.getuid(), 'Inspect the same terminal prior lease'
                    assert b'mllm_epd_prefill_heavy_token_split.servers' in (proc/'cmdline').read_bytes()
            assert time.time()<experiment_deadline()
            time.sleep(1)
    control.TAG=TAG
    control.EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
    names=[];proxies=[];results=[]
    row=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests'][0]
    try:
        control.wait_baselines()
        control.launch_models(names,diagnostic_eager=args.eager,fixed_reduction=args.fixed_reduction,
            deterministic_reductions=args.deterministic_reductions)
        for label,port in [('P_A',18841),('P_B',18841),('D_teacher',18842),('D_teacher_B',18842)]:
            role='P' if port==18841 else 'D'
            rpc(18840,'ec_ipc/inline_policy',dict(enabled=True,consumer_role='prefill' if role=='P' else 'decode'))
            run=TAG+'_'+label
            proxy,url=control.launch_proxy(run+'_proxy','e_pd',18848,names,decoder_port=port)
            proxies.append(proxy)
            try:control.capture(row,run,url,[(port,role)],2415,include_attention=True,include_gemm=args.gemm,attention_layer=args.attention_layer)
            finally:stop_proxy(proxy);proxies.remove(proxy)
            results.append((label,run,role))
        rpc(18840,'ec_ipc/inline_policy',dict(enabled=True,consumer_role='prefill'))
        run=TAG+'_TOKEN'
        proxy,url=control.launch_proxy(run+'_proxy','epd',18849,names,2415)
        proxies.append(proxy)
        try:control.capture(row,run,url,[(18841,'P'),(18842,'D')],include_attention=True,include_gemm=args.gemm,attention_layer=args.attention_layer)
        finally:stop_proxy(proxy);proxies.remove(proxy)
        results.extend([('Token_P',run,'P'),('Token_D',run,'D')])
        value=analyze(results,args.attention_layer)
        if args.gemm:
            assert all(len(r['first_layer_GEMM'])==4 for r in value['comparisons']), 'The actual compiled GEMM hooks did not capture four stages'
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
