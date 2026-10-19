"""Natural native fixed-R4 baseline. No candidate transformation is enabled."""
import argparse
import dataclasses
import datetime as dt
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
from poc_ouro_parallelism_deep_discovery.contract import *
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
configure_native_imports()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',choices=MODELS,required=True)
    parser.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    parser.add_argument('--async-scheduling',action='store_true',
        help='Check an existing native option on frozen calibration items. This is not a candidate method.')
    parser.add_argument('--runtime-label',default='',
        help='Separate a repeat from earlier accuracy denominators. This changes metadata only.')
    args=parser.parse_args()
    mask=validate_mask();assert len(mask)==args.tp
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    info=MODELS[args.size];tok=AutoTokenizer.from_pretrained(info['path'],trust_remote_code=True)
    protocol=read(ROOT/'artifacts/probe_protocol.json')
    rows=[r for r in protocol['rows'] if r['campaign_partition']=='calibration']
    assert len(rows)==16
    runtime='native_vllm_async' if args.async_scheduling else 'native_vllm'
    if args.runtime_label:
        assert args.runtime_label.replace('_','').isalnum()
        runtime+='_'+args.runtime_label
    suffix='_async' if args.async_scheduling else ''
    out=Path(os.environ['LOOPED_KV_OUTPUT'])/f'{args.size}_native_fixed_TP{args.tp}{suffix}'
    out.mkdir(parents=True,exist_ok=False)
    config=dict(model=str(info['path']),trust_remote_code=True,dtype='bfloat16',
        tensor_parallel_size=args.tp,hf_overrides={'total_ut_steps':4},seed=20261019,
        max_model_len=10240,max_num_seqs=8,max_num_batched_tokens=10240,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        kv_cache_memory_bytes=20*1024**3,enforce_eager=False,async_scheduling=args.async_scheduling,
        distributed_executor_backend='mp',disable_log_stats=False,
        worker_extension_cls='poc_looped_recurrence_tp_4models.worker_probe.FourModelProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    write(out/'engine_config.json',config);llm=LLM(**config)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(ownership)==args.tp and all(r['cache_slots']==info['layers']*4 for r in ownership)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership}=={UUIDS[GPUS.index(g)] for g in mask}
    write(out/'ownership.json',ownership)
    llm.generate([{'prompt_token_ids':tok.encode('Q: What is 1 + 1?\nA:')}],
        SamplingParams(temperature=0,max_tokens=4),use_tqdm=False)
    for row in rows:
        ids=tok.encode(row['prompt'],add_special_tokens=False)
        before=time.perf_counter()
        value=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,
            max_tokens=protocol['quality_cap'],ignore_eos=False,stop=row.get('stop_strings',[]),
            include_stop_str_in_output=True),use_tqdm=False)[0]
        elapsed=(time.perf_counter()-before)*1000;answer=value.outputs[0]
        native_stop=answer.finish_reason
        # The pinned vLLM scheduler leaves stop_reason unset for native EOS.
        # Explicit text/token stops populate that field instead.
        reason='length' if native_stop=='length' else 'eos' if answer.stop_reason in [None,tok.eos_token_id] else 'stop'
        result=dict(id=row['id'],dataset=row['dataset'],size=args.size,mode='fixed',runtime=runtime,
            async_scheduling=args.async_scheduling,
            TP=args.tp,model=info['id'],revision=info['revision'],physical_gpus=mask,
            prompt_ids=ids,output_ids=list(answer.token_ids),text=answer.text,
            generated_tokens=len(answer.token_ids),finish_reason=reason,
            native_finish_reason=native_stop,native_stop_reason=answer.stop_reason,
            metrics=None if value.metrics is None else dataclasses.asdict(value.metrics),
            wall_ms=elapsed,timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            EOS=tok.eos_token_id,cap=protocol['quality_cap'],temperature=0,
            source_protocol_sha256=sha(ROOT/'artifacts/probe_protocol.json'),
            label='QUALITY_ONLY native baseline pilot. Natural greedy output. No candidate speedup.',
            parity_note='Native EOS and stop-token exposure can differ from staged-reference output buffers. Raw IDs are retained.')
        write(out/(row['id'].replace('/','_')+'.json'),result)
        print('NATIVE_QUALITY',args.size,args.tp,row['id'],len(answer.token_ids),reason,flush=True)
    write(out/'status.json',dict(success=True,N=16,size=args.size,TP=args.tp,mode='fixed',
        runtime=runtime,async_scheduling=args.async_scheduling,
        quality_scope='Frozen calibration only. No held-out or candidate quality test.'))

if __name__=='__main__':main()
