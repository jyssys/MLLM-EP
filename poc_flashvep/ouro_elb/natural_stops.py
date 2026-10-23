"""Small natural-stop lifecycle probes. Not benchmark quality evidence."""
import gc
import torch
from transformers import AutoTokenizer
from .common import MODEL,write
from .experiments import new_engine

def run_job(model,job,target):
    from vllm_rlt import SamplingParams
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    prompts=['What is 2 + 2? Answer with one number.','Say hello in one short sentence.']
    results=[];references={}
    for D in (1,2,4):
        e=new_engine(model,2,D,job)
        for index,text in enumerate(prompts):
            ids=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)
            if hasattr(ids,'keys'):ids=ids['input_ids']
            ids=list(map(int,ids))
            rid='same-reused-id'
            e.add_request(rid,ids,SamplingParams(max_tokens=64,max_loops=4,min_loops=4,exit_threshold=1.,temperature=0.,ignore_eos=False))
            final=None;kind=None
            while e.has_unfinished_requests():
                for out in e.step():
                    final=out
                    if out.finished:kind=out.finish_reason
                    elif out.token_ids and out.token_ids[-1]==2:
                        e.abort_request(rid);kind='native_chat_end_frontend_stop'
            assert final and all(d==4 for d in final.exit_depths)
            if D==1:references[index]=list(final.token_ids)
            row=dict(D=D,prompt=text,input_ids=ids,output_ids=list(final.token_ids),text=tokenizer.decode(final.token_ids),
                stop=kind,depths=list(final.exit_depths),token_exact_to_D1=references[index]==list(final.token_ids),
                cache_released=e.cache_manager.num_used_blocks==0,scope='NATURAL_STOP_LIFECYCLE_NOT_GSM8K_MATH_QUALITY')
            write(target/f'natural_stop_D{D}_p{index}.json',row);results.append(row)
        e.close();del e;gc.collect();torch.cuda.empty_cache()
    return dict(results=results)
