"""Natural prompt checks and a red-capable cross-cap regression probe."""
import dataclasses
import os
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,5,6,7'
    from vllm import LLM, SamplingParams
    output = Path(os.environ['LOOPED_KV_OUTPUT'])
    kwargs = dict(model=str(MODEL), trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=4,
        hf_overrides={'total_ut_steps':4, 'early_exit_threshold':1.}, max_model_len=32768,
        max_num_seqs=8, max_num_batched_tokens=32768, gpu_memory_utilization=.85,
        seed=41015, enable_prefix_caching=False, enable_chunked_prefill=False,
        enable_expert_parallel=False, async_scheduling=False, enforce_eager=False,
        distributed_executor_backend='mp', disable_log_stats=False,
        worker_extension_cls='poc_looped_multigpu_kv_placement.worker_probe.KVWorkerProbe',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8]})
    write(output/'engine_contract.json', kwargs)
    llm = LLM(**kwargs)
    ownership = llm.collective_rpc('inspect_ownership', timeout=120)
    assert len(ownership)==4 and all(x['cache_slots']==192 for x in ownership)
    assert {x['GPU_uuid'].removeprefix('GPU-') for x in ownership} == set(UUIDS)
    write(output/'ownership.json', ownership)
    questions = read(ROOT/'results/frozen_questions.json')
    def wave(rows, cap, arm, temperature=0.):
        requests = [dict(prompt_token_ids=r['arms'][arm]['ids']) for r in rows]
        params = [SamplingParams(temperature=temperature, top_p=.7 if temperature else 1., top_k=-1,
            seed=SEEDS[0], max_tokens=cap, ignore_eos=False) for r in rows]
        start=time.perf_counter()
        values=llm.generate(requests, params, use_tqdm=False)
        wall=(time.perf_counter()-start)*1000
        return [dict(id=r['id'], dataset=r['dataset'], arm=arm, cap=cap, temperature=temperature,
            seed=SEEDS[0], ids=list(v.outputs[0].token_ids), text=v.outputs[0].text,
            finish_reason=v.outputs[0].finish_reason, stop_reason=v.outputs[0].stop_reason,
            prompt_tokens=len(v.prompt_token_ids), generated_tokens=len(v.outputs[0].token_ids),
            metrics=dataclasses.asdict(v.metrics) if v.metrics is not None else None,
            wave_wall_ms=wall) for r,v in zip(rows,values)]
    # Keep the previous B8 batch composition. Only the cap changes.
    old_wave = None
    for path in sorted((PREVIOUS/'results/quality_native_v3').glob('math100_greedy_cap4096_wave*.json')):
        previous=read(path)
        if any(x['problem_id']=='test/algebra/1339.json' for x in previous):
            ids=[x['problem_id'] for x in previous]
            old_wave=[next(r for r in questions if r['id']==i) for i in ids]
            break
    assert old_wave is not None
    result=[]
    for cap in [256,512]:
        for repeat in range(2):
            records=wave(old_wave,cap,'P0')
            write(output/f'regression_cap{cap}_repeat{repeat}.json',records)
            result.append((cap,repeat,records))
    comparisons=[]
    for first,second,label in [(result[0],result[1],'identical_cap256'),
        (result[2],result[3],'identical_cap512'),(result[0],result[2],'cap256_vs512')]:
        for a,b in zip(first[2],second[2]):
            length=min(len(a['ids']),len(b['ids']))
            divergence=next((i for i in range(length) if a['ids'][i]!=b['ids'][i]),None)
            comparisons.append(dict(test=label,id=a['id'],first_divergence=divergence,
                common_prefix_equal=divergence is None,compared_tokens=length))
    write(output/'regression_comparisons.json',comparisons)
    for arm in ['P0','P1']:
        for repeat in range(2):
            write(output/f'{arm}_short_repeat{repeat}.json',wave(questions[:8],256,arm))
        write(output/f'{arm}_long.json',wave(questions[:2],2048,arm))
    write(output/'status.json',dict(success=True,scope='Diagnostics only. No benchmark accuracy or full fidelity PASS.'))
    os._exit(0)

if __name__=='__main__':
    main()
