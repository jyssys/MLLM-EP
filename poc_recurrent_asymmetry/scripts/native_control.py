"""Audit the installed optimized TP baseline. No native model implementation."""
import argparse
import dataclasses
import os
import statistics
import time
import numpy as np
from poc_recurrent_asymmetry.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    parser.add_argument('--allocator',choices=['guard_default','cuda_malloc'],default='guard_default')
    parser.add_argument('--reference-directory',type=Path,default=ROOT/'outputs/physical_v1/physical');args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,GPUS[:args.tp]))
    os.environ.update(POC4_FAMILY='native',POC4_MODEL='ouro_14',POC4_R='4')
    if args.allocator=='cuda_malloc':
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        os.environ.pop('PYTORCH_ALLOC_CONF',None)
    from vllm import LLM,SamplingParams
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=args.tp,
        max_model_len=16640,max_num_batched_tokens=16640,max_num_seqs=8,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        kv_cache_memory_bytes=32*1024**3,enforce_eager=False,async_scheduling=False,disable_log_stats=False,
        distributed_executor_backend='mp',worker_extension_cls='poc_recurrent_asymmetry.native_probe.NativeProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]},seed=20261017)
    limits=dict(logits_max_abs=.5,logits_mean_abs=.05,KV_relative_L2=.02,
        natural_prefix_tokens=64,scope='Frozen cross-backend diagnostic limits, not a full quality-equivalence claim.')
    write(output/'engine_contract.json',dict(kwargs=kwargs,reference_limits=limits,allocator=args.allocator))
    llm=LLM(**kwargs)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(ownership)==args.tp and all(r['physical_layers']==24 and r['cache_slots']==96 for r in ownership)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership}==set(UUIDS[:args.tp])
    write(output/'ownership.json',ownership)
    manifest=read(ROOT/'metrics/frozen_workload.json');requests=manifest['requests']
    def run(ids,count=32,teacher=True):
        llm.collective_rpc('teacher',timeout=120,args=([13]*count if teacher else None,))
        start=time.perf_counter()
        result=llm.generate([dict(prompt_token_ids=ids)],SamplingParams(temperature=0,max_tokens=count,
            ignore_eos=teacher),use_tqdm=False)[0]
        wall=(time.perf_counter()-start)*1000
        tokens=list(result.outputs[0].token_ids)
        if teacher:assert tokens==[13]*count
        assert result.metrics is not None,'Request timing needs disable_log_stats=False'
        return dict(ids=tokens,text=result.outputs[0].text,wall_ms=wall,
            metrics=dataclasses.asdict(result.metrics),finish_reason=result.outputs[0].finish_reason)
    comparisons=[]
    for P in [1024,16384]:
        llm.collective_rpc('start_parity_capture',timeout=120)
        result=run(requests[0]['prompts'][str(P)],count=4)
        evidence=llm.collective_rpc('finish_parity_capture',timeout=120,args=(f'parity_P{P}',P))
        reference=np.load(args.reference_directory/f'native_reference_P{P}.npz')
        for rank,row in enumerate(evidence):
            arrays=np.load(row['path'])
            logits=np.abs(arrays['logits']-reference['logits'])
            native_KV=[];gold_KV=[]
            import re
            for index,name in enumerate(row['KV_names']):
                slot=int(re.search(r'layers\.(\d+)\.',name).group(1))
                heads=16//args.tp;begin=rank*heads
                native_KV.append(arrays['KV'][index])
                gold_KV.append(reference['KV'][slot,:,begin:begin+min(2,heads),:])
            a=np.array(native_KV,dtype=np.float64);b=np.array(gold_KV,dtype=np.float64)
            relative=float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1e-12))
            comparisons.append(dict(P=P,rank=rank,logits_max_abs=float(logits.max()),
                logits_mean_abs=float(logits.mean()),KV_relative_L2=relative,
                teacher_top1_matches=int((arrays['logits'].argmax(-1)==reference['logits'].argmax(-1)).sum()),n_positions=4,
                pass_numerical=float(logits.max())<=limits['logits_max_abs'] and float(logits.mean())<=limits['logits_mean_abs']
                    and relative<=limits['KV_relative_L2']))
        write(output/f'parity_P{P}.json',dict(result=result,evidence=evidence))
    prior=read(PROJECT/'poc_async_recurrent_pp/outputs/baseline_v1/reference_parity.json')['natural']
    natural=[]
    for row in prior[:2]:
        result=run(row['prompt_ids'],count=64,teacher=False)
        natural.append(dict(id=row['id'],result=result,reference_ids=row['official_ids'][:64],
            token_prefix_equal=result['ids']==row['official_ids'][:64]))
    numerical=all(row['pass_numerical'] for row in comparisons)
    parity=all(row['token_prefix_equal'] for row in natural)
    write(output/'fidelity.json',dict(numerical_limits=limits,comparisons=comparisons,natural=natural,
        bounded_qualification=numerical and parity,scope='Selected written KV, full teacher logits, and two natural prefixes. Not full quality equivalence.'))
    # Baseline timing is retained as diagnostic evidence even if qualification is partial.
    # A partial baseline must not justify an exact architecture GO decision.
    clean=[]
    for P in manifest['contexts']:
        for _ in range(3):run(requests[0]['prompts'][str(P)])
        for row in requests:
            for repeat in range(5):
                result=run(row['prompts'][str(P)])
                metrics=result['metrics']
                record=dict(P=P,TP=args.tp,request_id=row['id'],repeat=repeat,outputs=32,
                    wall_ms=result['wall_ms'],metrics=metrics,
                    TTFT_ms=(metrics['first_token_ts']-metrics['scheduled_ts'])*1000,
                    TPOT_ms=(metrics['last_token_ts']-metrics['first_token_ts'])*1000/31,
                    bounded_reference_qualification=numerical and parity,
                    label='MEASURED_NATIVE_BASELINE' if numerical and parity else 'REFERENCE_PARTIAL_DIAGNOSTIC_ONLY')
                clean.append(record);write(output/f'P{P}_{row["id"].replace("/","_")}_repeat{repeat}.json',record)
            print('NATIVE',args.tp,P,row['id'],flush=True)
    clean_profile=[run(requests[0]['prompts']['16384'],count=4)['wall_ms'] for _ in range(3)]
    llm.collective_rpc('start_kernel_profile',timeout=120)
    result=run(requests[0]['prompts']['16384'],count=4)
    traces=llm.collective_rpc('finish_kernel_profile',timeout=120)
    write(output/'kernel_profile.json',dict(clean_wall_ms=clean_profile,profile_wall_ms=result['wall_ms'],traces=traces,
        observer_percent=(result['wall_ms']/statistics.median(clean_profile)-1)*100,
        scope='Separate actual native profile. Kernel service is not an E2E percentage.'))
    write(output/'status.json',dict(success=True,TP=args.tp,clean_samples=len(clean),
        reference_qualification='BOUNDED_PASS' if numerical and parity else 'REFERENCE_PARTIAL',
        no_candidate_engine=True))
    os._exit(0)


if __name__=='__main__':main()
