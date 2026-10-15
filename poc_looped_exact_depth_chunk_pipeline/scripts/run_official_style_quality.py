"""Unforced R4 natural quality. No candidate engine or teacher hook."""
import dataclasses
import os
import signal
import time
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=4,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=32768,max_num_seqs=8,
        max_num_batched_tokens=32768,gpu_memory_utilization=.85,seed=41015,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=False,disable_log_stats=False,distributed_executor_backend='mp',
        worker_extension_cls='poc_looped_multigpu_kv_placement.worker_probe.KVWorkerProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    questions=read(ROOT/'results/frozen_quality_questions.json')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    container_differences=[]
    for row in questions:
        actual=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True)
        # Transformers 5 returns BatchEncoding by default; 4 returns the ID list.
        actual=actual['input_ids'] if hasattr(actual,'keys') else actual
        rendered=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=False,add_generation_prompt=True)
        assert tokenizer.decode(row['prompt_ids'],skip_special_tokens=False)==rendered,'Frozen token IDs must decode to the exact official prompt'
        if actual!=row['prompt_ids']:
            container_differences.append(dict(dataset=row['dataset'],id=row['id'],runtime_retokenized_ids=actual,
                frozen_reference_ids=row['prompt_ids'],decoded_prompt_equal=True))
    write(output/'tokenizer_preflight.json',dict(all_frozen_prompt_strings_identical=True,checked=len(questions),
        tokenization_differences=container_differences,
        scope='Use frozen Transformers4.57 official fast-tokenizer IDs unchanged. Transformers5 retokenization is diagnostic only.'))
    write(output/'engine_contract.json',kwargs);llm=LLM(**kwargs)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(ownership)==4 and all(r['physical_layers']==48 and r['cache_slots']==192 for r in ownership)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership}==set(UUIDS)
    write(output/'ownership.json',ownership)
    def watchdog(signum,frame):raise TimeoutError('Natural generation wave exceeded 1200 seconds')
    signal.signal(signal.SIGALRM,watchdog)
    def generate_wave(wave,cap,temperature,top_p):
        params=[SamplingParams(temperature=temperature,top_p=top_p,top_k=-1,seed=seed,
            max_tokens=cap,ignore_eos=False,repetition_penalty=1.) for row,seed in wave]
        signal.alarm(1200);start=time.perf_counter_ns()
        try:values=llm.generate([{'prompt_token_ids':row['prompt_ids']} for row,seed in wave],params,use_tqdm=False)
        finally:signal.alarm(0)
        wall=(time.perf_counter_ns()-start)/1e6
        return [dict(dataset=row['dataset'],problem_id=row['id'],index=row['index'],seed=seed,cap=cap,
            temperature=temperature,top_p=top_p,top_k=-1,prompt_sha256=row['prompt_sha256'],prompt_tokens=len(row['prompt_ids']),
            generated_ids=list(value.outputs[0].token_ids),text=value.outputs[0].text,
            generated_tokens=len(value.outputs[0].token_ids),finish_reason=value.outputs[0].finish_reason,
            stop_reason=value.outputs[0].stop_reason,metrics=dataclasses.asdict(value.metrics),wave_wall_ms=wall,
            total_positions=len(row['prompt_ids'])+len(value.outputs[0].token_ids),
            forced=False,manually_reviewed=False) for (row,seed),value in zip(wave,values)]
    # Two same-seed same-batch repeats establish reproducibility before scoring.
    smoke=[(questions[0],SEEDS[0]),(questions[1],SEEDS[0])]
    a=generate_wave(smoke,256,1.,.7);b=generate_wave(smoke,256,1.,.7)
    write(output/'seed_reproducibility.json',dict(first=a,second=b,
        exact_token_reproducibility=all(x['generated_ids']==y['generated_ids'] for x,y in zip(a,b)),
        scope='Two short natural calibration cases. Not benchmark quality.'))
    assert all(x['generated_ids']==y['generated_ids'] for x,y in zip(a,b)),'Fix seed semantics before scoring'
    progress=[]
    for dataset,lane,temp,p,seeds in [('aime24','sampling',1.,.7,SEEDS),('aime24','greedy',0.,1.,[SEEDS[0]]),
                                      ('math100','greedy',0.,1.,[SEEDS[0]])]:
        chosen=[r for r in questions if r['dataset']==dataset]
        opportunities=[(row,seed) for seed in seeds for row in chosen]
        for cap in [4096,8192]:
            records=[]
            for start in range(0,len(opportunities),8):
                path=output/f'{dataset}_{lane}_cap{cap}_wave{start//8:02d}.json'
                assert not path.exists(),'Preserve all original quality outputs'
                values=generate_wave(opportunities[start:start+8],cap,temp,p)
                write(path,values);records.extend(values)
                progress.append(dict(dataset=dataset,lane=lane,cap=cap,completed=len(records),total=len(opportunities)))
                write(output/'progress.json',progress)
                print(dataset,lane,cap,len(records),'/',len(opportunities),flush=True)
            capped=sum(r['finish_reason']=='length' for r in records)
            write(output/f'{dataset}_{lane}_cap{cap}_summary.json',dict(generations=len(records),capped=capped,
                cap_fraction=capped/len(records),next_cap=8192 if cap==4096 and capped/len(records)>=.1 else None))
            if capped/len(records)<.1 or cap==8192:break
    write(output/'status.json',dict(success=True,scope='Native published-style public evaluation. No method speedup.'))
    os._exit(0)


if __name__=='__main__':main()
