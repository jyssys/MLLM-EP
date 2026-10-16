"""Official-card arithmetic sanity. This is not benchmark accuracy."""
import os
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer,aime_integer

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5'
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    assert transformers.__version__=='4.57.1'
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    ids=read(ROOT/'results/model_tokenizer_audit.json')['card_example_rendering']['P1']['ids']
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    inputs=torch.tensor([ids],device='cuda')
    with torch.inference_mode():
        value=model.generate(inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=2048,
            do_sample=False,eos_token_id=2,pad_token_id=2,exit_at_step=3,return_dict_in_generate=True)
    generated=value.sequences[0,len(ids):].tolist()
    text=tokenizer.decode(generated,skip_special_tokens=False)
    answer,rule=final_answer(text)
    # The short model-card example does not request a boxed answer.
    # Keep unparseable output unscored rather than changing the benchmark parser.
    cache=value.past_key_values
    write(output/'arithmetic_sanity.json',dict(prompt_ids=ids,ids=generated,text=text,
        finish_reason='stop' if generated and generated[-1]==2 else 'length',expected_answer=4,
        parsed_answer=aime_integer(answer),parse_rule=rule,
        parsed_correct=aime_integer(answer)==4,cache_class=type(cache).__name__,
        cache_slots=len(cache.key_cache),cache_lengths=[cache.get_seq_length(i) for i in range(192)],
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
        scope='One official-card arithmetic diagnostic, not benchmark accuracy.'))
    write(output/'status.json',dict(success=True))

if __name__=='__main__':
    main()
