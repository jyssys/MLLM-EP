"""Measure actual logits at the first divergent natural token. No quality tuning."""
import argparse
import os
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',required=True,choices=['hf','native'])
    args=parser.parse_args()
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    inputs=read(ROOT/'results/calibration_divergent_prefixes.json')
    if args.backend=='hf':
        assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        assert transformers.__version__=='4.57.1'
        model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,
            device_map='cuda:0',attn_implementation='sdpa').eval()
        with torch.inference_mode():
            for row in inputs:
                past=None;inp=torch.tensor([row['prompt_ids']],device='cuda')
                for step in range(row['step']+1):
                    value=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                    past=value.past_key_values
                    inp=torch.tensor([[row['teacher_ids'][step]]],device='cuda')
                np.savez_compressed(output/f'first_divergence_{row["id"]}.npz',logits=value.logits[:,-1,:].float().cpu().numpy())
                write(output/f'first_divergence_{row["id"]}.json',dict(id=row['id'],step=row['step'],
                    current_top1=int(value.logits[:,-1,:].argmax(-1)),prior_top1=row['reference_next'],
                    cache_lengths=[past.get_seq_length(i) for i in range(192)],
                    GPU_uuid=str(torch.cuda.get_device_properties(0).uuid)))
        write(output/'status.json',dict(success=True,scope='Calibration first-divergence logits. Not benchmark accuracy.'))
        return
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    from vllm import LLM,SamplingParams
    kwargs=dict(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=32768,
        max_num_seqs=1,max_num_batched_tokens=32768,gpu_memory_utilization=.85,seed=41015,
        enable_prefix_caching=False,enable_chunked_prefill=False,enable_expert_parallel=False,
        async_scheduling=False,enforce_eager=False,disable_log_stats=False,distributed_executor_backend='mp',
        worker_extension_cls='poc_ouro_reasoning_fidelity_multigpu.worker_probe.FidelityProbe',
        compilation_config={'cudagraph_capture_sizes':[1]})
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    write(output/'ownership.json',llm.collective_rpc('inspect_ownership',timeout=120))
    for row in inputs:
        label='first_divergence_'+row['id']
        reference=read(ROOT/f'results/hf457_fidelity_v1/natural_{row["id"]}_cap256.json')
        assert len(reference['ids'])==256
        assert reference['ids'][:row['step']+1]==row['teacher_ids']
        llm.collective_rpc('teacher',args=(reference['ids'],))
        llm.collective_rpc('fidelity_start',args=(label,len(row['prompt_ids']),[row['step']]))
        llm.generate([dict(prompt_token_ids=row['prompt_ids'])],
            SamplingParams(temperature=0.,top_p=1.,top_k=-1,seed=SEEDS[0],max_tokens=256,ignore_eos=True),use_tqdm=False)
        write(output/f'{label}.json',llm.collective_rpc('fidelity_finish',timeout=120))
    llm.collective_rpc('teacher',args=(None,))
    write(output/'status.json',dict(success=True,scope='Calibration first-divergence logits. Not benchmark accuracy.'))
    os._exit(0)

if __name__=='__main__':
    main()
