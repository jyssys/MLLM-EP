"""Use official generation with the original full cache and EOS policy."""
import os
import signal
import torch
from poc_looped_multigpu_kv_placement.contract import *


def main():
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in GPUS
    from transformers import AutoModelForCausalLM,AutoTokenizer
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    model=AutoModelForCausalLM.from_pretrained(model_path('huginn'),trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',low_cpu_mem_usage=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path('huginn'),trust_remote_code=True)
    frozen=json.loads((ROOT/'results/frozen_quality.json').read_text())
    def expire(signum,frame):raise TimeoutError('Natural quality case watchdog300s')
    signal.signal(signal.SIGALRM,expire)
    for row in [r for r in frozen['cases'] if r['model']=='huginn']:
        torch.manual_seed(41015);torch.cuda.manual_seed_all(41015)
        ids=torch.tensor([row['prompt_token_ids']],device='cuda')
        signal.alarm(300)
        with torch.inference_mode():
            value=model.generate(ids,max_new_tokens=512,do_sample=False,num_steps=16,use_cache=True,
                temperature=None,top_k=None,top_p=None,return_dict_in_generate=True)
        signal.alarm(0);tokens=value.sequences[0,ids.shape[1]:].cpu().tolist()
        write(output/f'{row["request_id"]}.json',dict(request_id=row['request_id'],output_ids=tokens,
            text=tokenizer.decode(tokens,skip_special_tokens=False),output_length=len(tokens),
            native_EOS=model.generation_config.eos_token_id,truncated=len(tokens)==512,
            cache_policy='full',num_steps=16,physical_GPU_mask=os.environ['CUDA_VISIBLE_DEVICES'],
            scope='Natural calibration sanity, not benchmark accuracy. No code execution.'))
    write(output/'status.json',dict(success=True))


if __name__=='__main__':main()
