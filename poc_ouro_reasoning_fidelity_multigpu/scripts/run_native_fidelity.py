"""Native TP1/2/4 teacher-aligned and independent natural probes."""
import argparse
import dataclasses
import os
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tp',type=int,choices=[1,2,4],required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(map(str,GPUS[:args.tp]))
    from vllm import LLM, SamplingParams
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    kwargs=dict(model=str(MODEL), trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=args.tp,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=32768,
        max_num_seqs=1,max_num_batched_tokens=32768,gpu_memory_utilization=.85,seed=41015,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=False,distributed_executor_backend='mp',disable_log_stats=False,
        worker_extension_cls='poc_ouro_reasoning_fidelity_multigpu.worker_probe.FidelityProbe',
        compilation_config={'cudagraph_capture_sizes':[1]})
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    write(output/'effective_stop_contract.json',dict(renderer_eos_token_id=llm.llm_engine.renderer.get_eos_token_id(),
        ignore_eos_natural=False,additional_stop_strings=[],teacher_ignore_eos=True,
        scope='Teacher forcing cannot supply quality evidence. Natural lanes use the native renderer EOS.'))
    owners=llm.collective_rpc('inspect_ownership',timeout=120)
    assert all(r['physical_layers']==48 and r['cache_slots']==192 for r in owners)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in owners} == set(UUIDS[:args.tp])
    write(output/'ownership.json',owners)
    write(output/'effective_contract.json',llm.collective_rpc('audit_effective_contract'))
    inputs=read(ROOT/'results/fidelity_inputs.json')+read(ROOT/'results/fidelity_additional_inputs.json')
    for row in inputs:
        label='teacher_'+file_id(row['id'])
        llm.collective_rpc('teacher',args=(row['teacher_ids'],))
        llm.collective_rpc('fidelity_start',args=(label,len(row['prompt_ids']),[0,1,4,32,128,1024]))
        llm.generate([dict(prompt_token_ids=row['prompt_ids'])],SamplingParams(temperature=0.,max_tokens=1025,ignore_eos=True),use_tqdm=False)
        records=llm.collective_rpc('fidelity_finish',timeout=120)
        write(output/f'{label}.json',dict(id=row['id'],evidence=records))
        llm.collective_rpc('teacher',args=(None,))
        print('teacher',row['id'],flush=True)
    row=inputs[0]
    label='teacher_repeat_'+file_id(row['id'])
    llm.collective_rpc('teacher',args=(row['teacher_ids'],))
    llm.collective_rpc('fidelity_start',args=(label,len(row['prompt_ids']),[0,1,4,32,128,1024]))
    llm.generate([dict(prompt_token_ids=row['prompt_ids'])],SamplingParams(temperature=0.,max_tokens=1025,ignore_eos=True),use_tqdm=False)
    write(output/f'{label}.json',dict(id=row['id'],evidence=llm.collective_rpc('fidelity_finish',timeout=120)))
    llm.collective_rpc('teacher',args=(None,))
    completion_inputs=read(ROOT/'results/fidelity_completion_inputs.json')
    for cap,rows in [(256,inputs),(2048,inputs[:2]),(8192,completion_inputs)]:
        for row in rows:
            repeats=2 if cap==256 else 1
            for repeat in range(repeats):
                start=time.perf_counter()
                values=llm.generate([dict(prompt_token_ids=row['prompt_ids'])],
                    SamplingParams(temperature=0.,top_p=1.,top_k=-1,seed=SEEDS[0],max_tokens=cap,ignore_eos=False),use_tqdm=False)
                v=values[0];o=v.outputs[0]
                write(output/f'natural_{file_id(row["id"])}_cap{cap}_repeat{repeat}.json',dict(id=row['id'],
                    cap=cap,repeat=repeat,ids=list(o.token_ids),text=o.text,finish_reason=o.finish_reason,
                    stop_reason=o.stop_reason,wall_ms=(time.perf_counter()-start)*1000,
                    metrics=dataclasses.asdict(v.metrics) if v.metrics is not None else None,
                    timing_scope='Uninstrumented B1 natural probe. Not full benchmark.'))
        print('natural cap',cap,flush=True)
    sampling_records=[]
    for row in completion_inputs:
        for repeat in range(2):
            v=llm.generate([dict(prompt_token_ids=row['prompt_ids'])],
                SamplingParams(temperature=1.,top_p=.7,top_k=-1,seed=SEEDS[0],max_tokens=512,ignore_eos=False),use_tqdm=False)[0]
            o=v.outputs[0]
            record=dict(id=row['id'],seed=SEEDS[0],repeat=repeat,temperature=1.,top_p=.7,top_k=-1,cap=512,
                ids=list(o.token_ids),text=o.text,finish_reason=o.finish_reason,stop_reason=o.stop_reason)
            sampling_records.append(record)
    write(output/'controlled_sampling.json',sampling_records)
    write(output/'status.json',dict(success=True))
    os._exit(0)

if __name__=='__main__':
    main()
