"""Unchanged native TP4 on the frozen baseline-extension requests."""
import argparse
import collections
import dataclasses
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import signal
import time

from poc_ouro_parallelism_deep_discovery.contract import ROOT,MODELS,GPUS,UUIDS,read,sha,validate_mask,write


def require(condition,message):
    if not condition:
        raise ValueError(message)


def check_environment(environ):
    require(not any(key.startswith('POC4_') or key=='OURO_CAMPAIGN_STORAGE_CONTROL' for key in environ),
        'Reject inherited model-control environment before native imports')


def finish_reason(native_finish,native_stop,EOS):
    require(native_finish in ('length','stop'),'Unsupported native completion reason')
    return 'length' if native_finish=='length' else 'eos' if native_stop in (None,EOS) else 'stop'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',choices=MODELS,required=True)
    args=parser.parse_args()
    mask=validate_mask()
    require(mask==[4,5,6,7],'Require ordered physical GPUs4/5/6/7')
    check_environment(os.environ)
    freeze=read(ROOT/'artifacts/native_quality_extension_contract_v1.json')
    for relative,digest in freeze['source_sha256'].items():
        require(sha(ROOT/relative)==digest,'Frozen native-extension source changed: '+relative)
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:False'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
    os.environ['SPT_NOENV']='1'
    from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
    configure_native_imports()
    import transformers
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    info=MODELS[args.size]
    require(transformers.__version__=='5.16.1','Use the pinned native tokenizer version')
    tok=AutoTokenizer.from_pretrained(info['path'],trust_remote_code=True)
    require(tok.eos_token_id==0,'Native tokenizer EOS differs')
    protocol_path=ROOT/'artifacts/baseline_extension_protocol_v1.json'
    protocol=read(protocol_path)
    rows=protocol['rows']
    require(len(rows)==len({r['id'] for r in rows})==64,'Require the frozen64-request extension')
    require(collections.Counter(r['dataset'] for r in rows)=={'gsm8k':32,'math500':32},
        'Native extension domain counts differ')
    out=Path(os.environ['LOOPED_KV_OUTPUT'])/('native_extension_'+args.size)
    out.mkdir(parents=True,exist_ok=False)
    config=dict(model=str(info['path']),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=4,
        hf_overrides={'total_ut_steps':4},seed=20261019,max_model_len=10240,max_num_seqs=8,
        max_num_batched_tokens=10240,enable_prefix_caching=False,enable_chunked_prefill=False,
        enable_expert_parallel=False,kv_cache_memory_bytes=20*1024**3,enforce_eager=False,
        async_scheduling=True,distributed_executor_backend='mp',disable_log_stats=False,
        worker_extension_cls='poc_looped_multigpu_kv_placement.worker_probe.KVWorkerProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    write(out/'engine_config.json',config)
    llm=LLM(**config)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    require(len(ownership)==4 and all(r['cache_slots']==4*info['layers'] for r in ownership),
        'Depth-indexed ownership differs')
    require({r['GPU_uuid'].removeprefix('GPU-') for r in ownership}==
        {UUIDS[GPUS.index(g)] for g in mask},'Physical ownership differs')
    require(all(all(g['Q_heads']==4 and g['KV_heads']==4 for g in r['geometry']) for r in ownership),
        'Native TP4 head ownership differs')
    require(all(r['world']==4 and r['physical_layers']==info['layers'] and
        all(g['loop_cache_modules']==4 for g in r['geometry']) for r in ownership),
        'Native TP4 physical layers or recurrent banks differ')
    write(out/'ownership.json',ownership)
    llm.generate([{'prompt_token_ids':tok.encode('Q: What is 1 + 1?\nA:')}],
        SamplingParams(temperature=0,max_tokens=4),use_tqdm=False)
    def expired(signum,frame):
        raise TimeoutError('Frozen native-extension request exceeded1200seconds')
    signal.signal(signal.SIGALRM,expired)
    for row in rows:
        ids=tok.encode(row['prompt'],add_special_tokens=False)
        require(len(ids)+8192<=10240,'Frozen prompt exceeds the declared context budget')
        signal.alarm(1200)
        before=time.perf_counter()
        try:
            value=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=8192,
                ignore_eos=False,stop=row.get('stop_strings',[]),include_stop_str_in_output=True),use_tqdm=False)[0]
            elapsed=(time.perf_counter()-before)*1000
        finally:
            signal.alarm(0)
        answer=value.outputs[0]
        tokens=list(answer.token_ids)
        reason=finish_reason(answer.finish_reason,answer.stop_reason,tok.eos_token_id)
        result=dict(id=row['id'],dataset=row['dataset'],size=args.size,mode='fixed',runtime='native_vllm_async_extension',
            TP=4,revision=info['revision'],physical_gpus=mask,prompt_ids=ids,output_ids=tokens,
            text=tok.decode(tokens,skip_special_tokens=False),native_api_text=answer.text,
            generated_tokens=len(tokens),finish_reason=reason,native_finish_reason=answer.finish_reason,
            native_stop_reason=answer.stop_reason,EOS=tok.eos_token_id,cap=8192,temperature=0,
            source_protocol_sha256=sha(protocol_path),extension_contract_sha256=sha(ROOT/'artifacts/native_quality_extension_contract_v1.json'),
            wall_ms=elapsed,metrics=None if value.metrics is None else dataclasses.asdict(value.metrics),
            timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),evidence_class='QUALITY_ONLY',
            scope='Existing native baseline on previously frozen extension IDs. No candidate or held-out confirmation.')
        with gzip.open(out/(row['id'].replace('/','_')+'.json.gz'),'xt',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,allow_nan=False)
        print('NATIVE_EXTENSION',args.size,row['id'],len(tokens),reason,flush=True)
    write(out/'status.json',dict(success=True,N=64,size=args.size,TP=4,mode='fixed',physical_gpus=mask,
        evidence_class='QUALITY_ONLY',candidate='NONE',original_held_out='UNTOUCHED'))


if __name__=='__main__':
    main()
