"""Same native vLLM build and eager FA3 backend for TP1 versus TP4."""
import argparse
from .contract import *

def main():
    import torch
    from vllm import LLM,SamplingParams
    p=argparse.ArgumentParser();p.add_argument('--tp',type=int,choices=[1,4],required=True);a=p.parse_args()
    llm=LLM(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=a.tp,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=8192,max_num_seqs=4,
        max_num_batched_tokens=8192,kv_cache_memory_bytes=8*2**30,enable_prefix_caching=False,
        enable_chunked_prefill=False,enforce_eager=True,async_scheduling=False,
        distributed_executor_backend='mp',worker_extension_cls='poc_flashvep.ouro_8arm.native_probe.Probe',disable_log_stats=True)
    write(ROOT/'numerical'/f'vLLM_TP{a.tp}_ownership.json',llm.collective_rpc('inspect_ownership'))
    outputs=[]
    for row in read(ROOT/'numerical/hf_manifest.json'):
        ref=torch.load(row['path'],weights_only=False,map_location='cpu')
        llm.collective_rpc('teacher',args=(ref['ids'],));llm.collective_rpc('start_trace')
        y=llm.generate([dict(prompt_token_ids=ref['input_ids'])],SamplingParams(temperature=0.,max_tokens=row['O'],ignore_eos=True),use_tqdm=False)
        result=llm.collective_rpc('finish_trace',args=(row['label'],a.tp))
        outputs.append(dict(label=row['label'],capture=result,output_ids=list(y[0].outputs[0].token_ids)))
        print('NATIVE_NUMERICAL',a.tp,row['label'],flush=True)
    write(ROOT/'numerical'/f'vLLM_TP{a.tp}_manifest.json',outputs)
    llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
