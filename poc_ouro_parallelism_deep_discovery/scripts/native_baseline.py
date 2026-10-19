"""Fresh optimized fixed-depth TP baselines. Controlled work is not task accuracy."""
import argparse
import dataclasses
import os
import signal
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
from poc_ouro_parallelism_deep_discovery.contract import *
# The verified native custom All-Reduce uses legacy CUDA IPC for graph buffers.
# The reduced red/green test rejects expandable allocation on this runtime.
# Set the scoped allocator before importing Torch or vLLM.
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
configure_native_imports()

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    p.add_argument('--R',type=int,choices=[1,4],required=True)
    p.add_argument('--fuse-allreduce-rms',action='store_true',
        help='Generic baseline-strength control. This is not a new looped architecture.')
    p.add_argument('--async-scheduling',action='store_true',
        help='Generic native asynchronous scheduler control, not a new recurrent architecture.')
    p.add_argument('--weight-storage-control',choices=['alias','copies'],
        help='Allocate four identical block copies in both arms. Change only execution pointer reuse. Not a candidate method.')
    p.add_argument('--isolate-compile-cache',action='store_true',
        help='Isolate the per-job vLLM index cache only. Global AOT isolation requires --isolate-all-compile-caches.')
    p.add_argument('--isolate-all-compile-caches',action='store_true',
        help='Also isolate VLLM_CACHE_ROOT and lower-level compiler paths. The global AOT model cache otherwise remains shared.')
    args=p.parse_args();mask=validate_mask();assert len(mask)==args.tp
    output=Path(os.environ['LOOPED_KV_OUTPUT'])/f'{args.size}_R{args.R}_TP{args.tp}'
    output.mkdir(parents=True,exist_ok=False)
    if args.isolate_all_compile_caches:
        scoped={name:str(output/directory) for name,directory in [
            ('VLLM_CACHE_ROOT','vllm_cache'),('TORCHINDUCTOR_CACHE_DIR','inductor_cache'),('TRITON_CACHE_DIR','triton_cache')]}
        for name,value in scoped.items():
            assert not Path(value).exists()
            os.environ[name]=value
        write(output/'cache_environment.json',dict(before_vllm_import=True,variables=scoped,
            scope='Per-job cache roots. Actual compiled storage arguments still require inspection.'))
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    info=MODELS[args.size];tok=AutoTokenizer.from_pretrained(str(info['path']),trust_remote_code=True)
    protocol=read(ROOT/'artifacts/probe_protocol.json')
    cases=[next(r for r in protocol['rows'] if r['dataset']==domain and r['campaign_partition']=='calibration')
           for domain in ['gsm8k','math500']]
    teacher=(tok.encode(' The answer is 42.\n',add_special_tokens=False)*64)[:128]
    inputs=[]
    for row in cases:
        base=tok.encode(row['prompt'],add_special_tokens=False)
        inputs.append({'id':row['id'],'prompt_ids':{str(N):(base*((N+len(base)-1)//len(base)))[:N]
                      for N in [1024,4096,8192]},'teacher_ids':teacher})
    write(output/'workload.json',dict(inputs=inputs,
        scope='Repeated prompt tokens create controlled hardware shapes. This is not long-context reasoning quality.',
        frozen_before_engine=True,output_count=128,ignore_EOS=True))
    config=dict(model=str(info['path']),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=args.tp,
        hf_overrides={'total_ut_steps':args.R},seed=20261019,max_model_len=10240,max_num_seqs=8,
        max_num_batched_tokens=10240,enable_prefix_caching=False,enable_chunked_prefill=False,
        enable_expert_parallel=False,kv_cache_memory_bytes=20*1024**3,
        enforce_eager=False,async_scheduling=False,distributed_executor_backend='mp',
        disable_log_stats=False,
        worker_extension_cls='poc_looped_recurrence_tp_4models.worker_probe.FourModelProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    if args.fuse_allreduce_rms:
        assert args.tp>1
        config['compilation_config']['pass_config']={'fuse_allreduce_rms':True}
    if args.async_scheduling:config['async_scheduling']=True
    if args.isolate_compile_cache or args.isolate_all_compile_caches:
        cache=output/'compile_cache'
        assert not cache.exists()
        config['compilation_config']['cache_dir']=str(cache)
    if args.weight_storage_control:
        assert args.R==4
        os.environ['OURO_CAMPAIGN_STORAGE_CONTROL']=args.weight_storage_control
        config['worker_extension_cls']='poc_ouro_parallelism_deep_discovery.worker_probe.CampaignProbe'
    write(output/'engine_config.json',config)
    llm=LLM(**config)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(ownership)==args.tp
    assert all(r['cache_slots']==info['layers']*args.R for r in ownership)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership}=={UUIDS[GPUS.index(g)] for g in mask}
    assert all(all(g['Q_heads']==16//args.tp and g['KV_heads']==16//args.tp for g in r['geometry']) for r in ownership)
    write(output/'ownership.json',ownership)
    if args.weight_storage_control:
        write(output/'storage_control.json',llm.collective_rpc('storage_control_evidence',timeout=120))
    def generate(row,N,count=128):
        llm.collective_rpc('teacher',timeout=120,args=(teacher[:count],))
        before=time.perf_counter()
        value=llm.generate([{'prompt_token_ids':row['prompt_ids'][str(N)]}],
            SamplingParams(temperature=0,max_tokens=count,ignore_eos=True),use_tqdm=False)[0]
        elapsed=(time.perf_counter()-before)*1000
        assert list(value.outputs[0].token_ids)==teacher[:count]
        metrics=None if value.metrics is None else dataclasses.asdict(value.metrics)
        return dict(wall_ms=elapsed,metrics=metrics,output_ids=list(value.outputs[0].token_ids),
            finish_reason=value.outputs[0].finish_reason,scope='Controlled-work live E2E. Forced teacher tokens, not task accuracy.')
    for N in [1024,4096,8192]:
        for _ in range(3):generate(inputs[0],N,16)
        for i,row in enumerate(inputs):
            for repeat in range(5):
                result=generate(row,N)
                write(output/f'P{N}_request{i}_repeat{repeat}.json',dict(size=args.size,R=args.R,TP=args.tp,
                    physical_gpus=mask,prompt=N,request_id=row['id'],repeat=repeat,**result))
            print('NATIVE_BASELINE',args.size,args.R,args.tp,N,i,flush=True)
    # Numerical capture is a separate lane. It is never the clean timing source.
    llm.collective_rpc('start_parity_capture',timeout=120)
    generated=generate(inputs[0],1024,4)
    evidence=llm.collective_rpc('finish_parity_capture',timeout=120,args=(f'base_{args.size}_R{args.R}_TP{args.tp}',1024))
    write(output/'parity_capture.json',dict(generated=generated,evidence=evidence,
        reference_status='Native samples only. Independent reference comparison remains pending.'))
    write(output/'status.json',dict(success=True,size=args.size,R=args.R,TP=args.tp,physical_gpus=mask,
        benchmark='Optimized fixed-depth native vLLM, graph enabled, controlled input and output work.',
        quality='NOT_RUN',adaptive='NOT_IMPLEMENTED_BY_THIS_NATIVE_PATH'))

if __name__=='__main__':main()
