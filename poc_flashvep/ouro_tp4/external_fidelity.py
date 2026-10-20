"""Try the installed native Ouro TP4 without changing that environment."""
import dataclasses
import os
import time
from .contract import *


def main():
    mask()
    from vllm import LLM,SamplingParams
    import torch
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=4,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=4096,max_num_seqs=4,
        max_num_batched_tokens=4096,gpu_memory_utilization=.80,kv_cache_memory_bytes=4*2**30,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=True,distributed_executor_backend='mp',
        worker_extension_cls='poc_flashvep.ouro_tp4.external_probe.ExternalProbe',disable_log_stats=True)
    write(ROOT/'external_engine_contract.json',kwargs)
    start=time.perf_counter();llm=LLM(**kwargs)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(ownership)==4 and all(r['physical_layers']==24 and r['cache_slots']==96 and r['world']==4 for r in ownership)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership}==set(UUIDS)
    write(ROOT/'external_ownership.json',ownership)
    write(ROOT/'external_setup.json',dict(wall_s=time.perf_counter()-start,scope='Initialization, not serving latency'))
    for P,O in [(128,8),(512,32),(2048,32)]:
        gold=next(iter(torch.load(ROOT/f'.tensor_cache/native_P{P}_O{O}.pt',weights_only=False).values()))
        llm.collective_rpc('teacher',args=(gold['ids'],))
        llm.collective_rpc('start_tp_reference')
        values=llm.generate([dict(prompt_token_ids=gold['input_ids'])],
            SamplingParams(temperature=0.,max_tokens=O,ignore_eos=True),use_tqdm=False)
        evidence=llm.collective_rpc('finish_tp_reference',args=(P,O,f'external_P{P}_O{O}'))
        write(ROOT/'correctness'/f'external_P{P}_O{O}.json',dict(P=P,O=O,evidence=evidence,
            teacher_ids=list(values[0].outputs[0].token_ids),teacher_exact=list(values[0].outputs[0].token_ids)==gold['ids']))
        llm.collective_rpc('teacher',args=(None,))
        natural=llm.generate([dict(prompt_token_ids=gold['input_ids'])],
            SamplingParams(temperature=0.,max_tokens=O,ignore_eos=False),use_tqdm=False)[0]
        write(ROOT/'correctness'/f'external_natural_P{P}_O{O}.json',dict(P=P,O=O,ids=list(natural.outputs[0].token_ids),
            native_ids=gold['ids'],token_exact=list(natural.outputs[0].token_ids)==gold['ids'],
            finish_reason=natural.outputs[0].finish_reason,scope='Natural greedy correctness. Not a performance or accuracy benchmark.'))
        print('EXTERNAL_FIDELITY_CAPTURED',P,O,flush=True)
    write(ROOT/'external_completion.json',dict(success=True,performance_matrix='NOT_RUN until numerical comparisons pass'))
    llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
