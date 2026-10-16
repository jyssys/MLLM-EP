"""Preregistered natural evaluation. No output forcing or score-based arm selection."""
import argparse
import dataclasses
import os
import signal
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    from vllm import LLM,SamplingParams
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    lock=read(ROOT/'protocol_lock.json')
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=4,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=32768,
        max_num_seqs=4,max_num_batched_tokens=32768,gpu_memory_utilization=.85,seed=41015,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=False,disable_log_stats=False,distributed_executor_backend='mp',
        worker_extension_cls='poc_looped_multigpu_kv_placement.worker_probe.KVWorkerProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4]})
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    write(output/'effective_stop_contract.json',dict(renderer_eos_token_id=llm.llm_engine.renderer.get_eos_token_id(),
        ignore_eos=False,additional_stop_strings=[],quality_lane=True))
    owners=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(owners)==4 and all(r['cache_slots']==192 for r in owners)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in owners}==set(UUIDS)
    write(output/'ownership.json',owners)
    questions=read(ROOT/'results/frozen_questions.json')
    def timeout(signum,frame):
        raise TimeoutError('Natural wave exceeded the bounded 1800-second watchdog')
    signal.signal(signal.SIGALRM,timeout)
    def generate(wave,cap,arm,temperature,top_p):
        params=[SamplingParams(temperature=temperature,top_p=top_p,top_k=-1,seed=seed,
            max_tokens=min(cap,32768-len(row['arms'][arm]['ids'])),ignore_eos=False,
            repetition_penalty=1.,stop=[]) for row,seed in wave]
        signal.alarm(1800)
        start=time.perf_counter_ns()
        try:
            values=llm.generate([dict(prompt_token_ids=row['arms'][arm]['ids']) for row,seed in wave],params,use_tqdm=False)
        finally:
            signal.alarm(0)
        wall=(time.perf_counter_ns()-start)/1e6
        assert all(list(v.prompt_token_ids)==row['arms'][arm]['ids'] for (row,seed),v in zip(wave,values)), 'Native input IDs differ from the frozen prompt'
        return [dict(dataset=row['dataset'],problem_id=row['id'],index=row['index'],arm=arm,
            seed=seed,cap=cap,effective_cap=params[i].max_tokens,temperature=temperature,top_p=top_p,top_k=-1,
            prompt_sha256=row['arms'][arm]['text_sha256'],prompt_tokens=len(row['arms'][arm]['ids']),
            generated_ids=list(v.outputs[0].token_ids),text=v.outputs[0].text,
            generated_tokens=len(v.outputs[0].token_ids),finish_reason=v.outputs[0].finish_reason,
            stop_reason=v.outputs[0].stop_reason,metrics=dataclasses.asdict(v.metrics),wave_wall_ms=wall,
            total_positions=len(row['arms'][arm]['ids'])+len(v.outputs[0].token_ids),forced=False,
            manually_reviewed=False,protocol_lock_sha256=sha(ROOT/'protocol_lock.json'))
            for i,((row,seed),v) in enumerate(zip(wave,values))]
    if args.diagnostic:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
        content='Solve: If 2x + 3 = 11, what is x?'
        text=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=False,
            add_generation_prompt=True,enable_thinking=True)
        ids=tokenizer.encode(text,add_special_tokens=False)
        sanity=dict(id='official_card_arithmetic',dataset='sanity',index=-1,
            arms={'P1':dict(ids=ids,text_sha256=hashlib.sha256(text.encode()).hexdigest())})
        write(output/'arithmetic_sanity.json',generate([(sanity,SEEDS[0])],8192,'P1',0.,1.))
        for arm in ['P0','P1']:
            for start in range(0,8,4):
                path=output/f'diagnostic_{arm}_cap8192_wave{start//4:02d}.json'
                assert not path.exists()
                records=generate([(r,SEEDS[0]) for r in questions[start:start+4]],8192,arm,1.,.7)
                write(path,records)
                print('diagnostic',arm,start+len(records),flush=True)
        repeated=generate([(r,SEEDS[0]) for r in questions[:4]],8192,'P1',1.,.7)
        original=read(output/'diagnostic_P1_cap8192_wave00.json')
        comparisons=[dict(id=a['problem_id'],exact_equal=a['generated_ids']==b['generated_ids'],
            first_divergence=next((i for i,(u,v) in enumerate(zip(a['generated_ids'],b['generated_ids'])) if u!=v),None))
            for a,b in zip(original,repeated)]
        write(output/'diagnostic_same_config_long_repro.json',dict(repeated=repeated,comparisons=comparisons,
            PASS=all(r['exact_equal'] for r in comparisons)))
        assert all(r['exact_equal'] for r in comparisons),'Resolve same-config long sampling reproducibility before full quality scoring'
    else:
        sanity=[(row,SEEDS[0]) for row in questions[:4]]
        first=generate(sanity,256,'P1',1.,.7)
        second=generate(sanity,256,'P1',1.,.7)
        reproducible=all(a['generated_ids']==b['generated_ids'] for a,b in zip(first,second))
        write(output/'seed_reproducibility.json',dict(first=first,second=second,PASS=reproducible))
        assert reproducible,'Resolve seed reproducibility before full scoring'
        progress=[]
        for dataset,lane,temp,p,seeds in [('aime24','sampling',1.,.7,SEEDS),
            ('aime24','greedy',0.,1.,[SEEDS[0]]),('math100','greedy',0.,1.,[SEEDS[0]])]:
            chosen=[r for r in questions if r['dataset']==dataset]
            opportunities=[(row,seed) for seed in seeds for row in chosen]
            for cap in lock['quality']['caps']:
                records=[]
                for start in range(0,len(opportunities),4):
                    path=output/f'{dataset}_{lane}_cap{cap}_wave{start//4:02d}.json'
                    assert not path.exists(),'Do not overwrite an earlier raw generation'
                    values=generate(opportunities[start:start+4],cap,'P1',temp,p)
                    write(path,values)
                    records.extend(values)
                    progress.append(dict(dataset=dataset,lane=lane,cap=cap,completed=len(records),total=len(opportunities)))
                    write(output/'progress.json',progress)
                    print(dataset,lane,cap,len(records),'/',len(opportunities),flush=True)
                capped=sum(r['finish_reason']=='length' for r in records)
                write(output/f'{dataset}_{lane}_cap{cap}_summary.json',dict(total=len(records),capped=capped,
                    cap_fraction=capped/len(records),collection_complete=len(records)==len(opportunities)))
                if capped/len(records)<.1:
                    break
    write(output/'status.json',dict(success=True,diagnostic=args.diagnostic))
    os._exit(0)

if __name__=='__main__':
    main()
