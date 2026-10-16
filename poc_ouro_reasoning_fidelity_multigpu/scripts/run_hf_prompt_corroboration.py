"""Two predeclared AIME cases under both prompt arms. No serving speed claim."""
import gc
import os
import signal
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    assert transformers.__version__=='4.57.1'
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    assert model.config.total_ut_steps==4 and model.config.early_exit_threshold==1.
    frozen=read(ROOT/'results/frozen_questions.json')
    chosen=[next(r for r in frozen if r['id']==i and r['dataset']=='aime24') for i in ['60','61']]
    write(output/'reference_contract.json',dict(transformers=transformers.__version__,torch=torch.__version__,
        checkpoint=str(MODEL),revision=REVISION,R=4,exit_at_step=3,EOS=2,attention='SDPA',
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),physical_gpu=4,
        lock_sha256=sha(ROOT/'hf_corroboration_lock.json'),
        scope='Reference completion diagnostic. Wall time is not optimized native serving performance.'))
    def alarm(signum,frame):
        raise TimeoutError('Reference completion exceeded its 2700-second watchdog')
    signal.signal(signal.SIGALRM,alarm)
    records=[]
    for arm in ['P0','P1']:
        for row in chosen:
            torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
            inputs=torch.tensor([row['arms'][arm]['ids']],device='cuda')
            signal.alarm(2700)
            start=time.perf_counter()
            try:
                with torch.inference_mode():
                    values=model.generate(inputs,max_new_tokens=8192,do_sample=True,temperature=1.,top_p=.7,
                        top_k=0,eos_token_id=2,pad_token_id=2,exit_at_step=3,return_dict_in_generate=True)
            finally:
                signal.alarm(0)
            wall=(time.perf_counter()-start)*1000
            ids=values.sequences[0,len(row['arms'][arm]['ids']):].tolist()
            cache=values.past_key_values
            assert len(cache.key_cache)==192
            record=dict(dataset='aime24',problem_id=row['id'],arm=arm,prompt_ids=row['arms'][arm]['ids'],
                seed=SEEDS[0],cap=8192,generated_ids=ids,generated_tokens=len(ids),
                text=tokenizer.decode(ids,skip_special_tokens=False),
                finish_reason='stop' if ids[-1:]==[2] else 'length',reference_wall_ms=wall,
                cache_slots=len(cache.key_cache),cache_lengths=[cache.get_seq_length(i) for i in range(192)],
                EOS=2,forced=False,temperature=1.,top_p=.7,top_k=0,
                scope='Two predeclared overlapping AIME problems. No full-benchmark or serving-latency claim.')
            records.append(record)
            write(output/'reference_completions.json',records)
            print('HF reference corroboration',arm,row['id'],len(ids),record['finish_reason'],flush=True)
            del cache,values,inputs
            gc.collect();torch.cuda.empty_cache()
    from poc_ouro_reasoning_fidelity_multigpu.hf_threshold_audit import audit
    signal.alarm(600)
    try:
        audit(model,output)
    finally:
        signal.alarm(0)
    write(output/'status.json',dict(success=True,completed_records=len(records),
        independent_problems=2,peak_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    main()
