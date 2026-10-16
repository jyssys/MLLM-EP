"""Test two ranked calibration hypotheses on one permitted GPU."""
import os
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    from vllm import LLM,SamplingParams
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=32768,
        max_num_seqs=1,max_num_batched_tokens=32768,gpu_memory_utilization=.85,seed=41015,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=True,disable_log_stats=False,distributed_executor_backend='mp',
        compilation_config={'mode':0},
        worker_extension_cls='poc_ouro_reasoning_fidelity_multigpu.norm_diagnostic.NormDiagnosticProbe')
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    ownership=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(ownership)==1 and ownership[0]['GPU_uuid'].removeprefix('GPU-')==UUIDS[2]
    write(output/'ownership.json',ownership)
    weights=llm.collective_rpc('check_loaded_weights',timeout=300)
    write(output/'weight_identity.json',weights)
    assert weights[0]['all_equal']
    row=read(ROOT/'results/fidelity_inputs.json')[0]
    assert row['split']=='calibration' and row['id']=='60'
    for mode in ['native','hf_operator']:
        write(output/f'norm_mode_{mode}.json',llm.collective_rpc('norm_mode',args=(mode,)))
        llm.collective_rpc('teacher',args=(row['teacher_ids'],))
        llm.collective_rpc('fidelity_start',args=(mode,len(row['prompt_ids']),[0,128,1024]))
        llm.generate([dict(prompt_token_ids=row['prompt_ids'])],
            SamplingParams(temperature=0.,max_tokens=1025,ignore_eos=True),use_tqdm=False)
        write(output/f'{mode}.json',llm.collective_rpc('fidelity_finish',timeout=120))
    llm.collective_rpc('teacher',args=(None,))
    llm.collective_rpc('norm_mode',args=('native',))
    write(output/'status.json',dict(success=True,
        scope='Calibration-only eager operator ablation on GPU6. Not the GPU4 primary TP1 baseline or a method.'))
    os._exit(0)

if __name__=='__main__':
    main()
