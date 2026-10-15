"""Official pretrained Huginn forward/cache, single GPU only. No TP emulator."""
import argparse
import gzip
import os
import signal
import time
import torch
from poc_looped_multigpu_kv_placement.contract import *


def cache_record(cache):
    storages={};logical=0;slots=[]
    for index in sorted(cache.key_cache):
        components=[]
        for kind,values in [('K',cache.key_cache[index]),('V',cache.value_cache[index])]:
            for tensor in values.values():
                logical+=tensor.numel()*tensor.element_size()
                storages[tensor.untyped_storage().data_ptr()]=tensor.untyped_storage().nbytes()
            examples=[]
            for position in sorted(set([min(values),max(values)])):
                tensor=values[position]
                examples.append(dict(position=position,shape=list(tensor.shape),stride=list(tensor.stride()),
                    storage_ptr=tensor.untyped_storage().data_ptr(),offset=tensor.storage_offset(),
                    dtype=str(tensor.dtype),logical_bytes=tensor.numel()*tensor.element_size(),
                    storage_bytes=tensor.untyped_storage().nbytes()))
            components.append(dict(kind=kind,positions=len(values),examples=examples))
        slots.append(dict(step_index=index,components=components))
    return dict(policy=cache.lookup_strategy,slot_count=len(cache.key_cache),slots=slots,
        logical_unique_elements_bytes=logical,actual_unique_storage_bytes=sum(storages.values()),
        storage_count=len(storages),seen_tokens=cache.get_seq_length(),
        limitation='Unique allocation capacity is not measured HBM traffic or bandwidth.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--limit',type=int,default=8);parser.add_argument('--prefixes',type=int,nargs='+',default=[1024,4032])
    parser.add_argument('--steps',type=int,default=16);args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in [str(gpu) for gpu in GPUS]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model=AutoModelForCausalLM.from_pretrained(model_path('huginn'),trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',low_cpu_mem_usage=True).eval()
    cache_class=__import__(type(model).__module__,fromlist=['HuginnDynamicCache']).HuginnDynamicCache
    weights={p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}
    write(output/'loaded.json',dict(model=MODELS['huginn'],steps=args.steps,physical_GPU_mask=os.environ['CUDA_VISIBLE_DEVICES'],
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),weight_unique_bytes=sum(weights.values()),
        physical_blocks={key:len(model.transformer[key]) for key in ['prelude','core_block','coda']},
        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        head_count=model.config.num_attention_heads,KV_head_count=model.config.num_key_value_heads,
        state_init='Official default init_scale1. Reset seed41015 per request. No adaptive steps.',
        runner='Official Hugging Face forward and HuginnDynamicCache. Not an optimized multi-GPU serving baseline.'))
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:rows=[r for r in json.load(stream) if r['model']=='huginn'][:args.limit]
    def expired(signum,frame):raise TimeoutError('Huginn per-request watchdog300s')
    signal.signal(signal.SIGALRM,expired)
    def generate(ids,policy='full',tokens=64,events=False):
        torch.manual_seed(41015);torch.cuda.manual_seed_all(41015)
        cache=cache_class(lookup_strategy=policy);x=torch.tensor([ids],device='cuda',dtype=torch.long)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        if events:marks=[torch.cuda.Event(enable_timing=True) for _ in range(tokens+1)]
        start=time.perf_counter_ns()
        with torch.inference_mode():
            value=model(x,use_cache=True,past_key_values=cache,num_steps=args.steps,
                cache_position=torch.arange(len(ids),device='cuda'),output_details={'return_logits':True,'return_latents':False,'return_head':False,'return_stats':False})
            next_id=value.logits[:,-1:].argmax(-1);output_ids=[next_id];del value
            torch.cuda.synchronize();prefill_end=time.perf_counter_ns()
            if events:marks[0].record()
            for step in range(tokens-1):
                value=model(next_id,use_cache=True,past_key_values=cache,num_steps=args.steps,
                    cache_position=torch.tensor([len(ids)+step],device='cuda'),
                    output_details={'return_logits':True,'return_latents':False,'return_head':False,'return_stats':False})
                next_id=value.logits[:,-1:].argmax(-1);output_ids.append(next_id);del value
                if events:marks[step+1].record()
            torch.cuda.synchronize();end=time.perf_counter_ns()
        result=dict(output_ids=torch.cat(output_ids,dim=1)[0].cpu().tolist(),prefill_ms=(prefill_end-start)/1e6,
            decode_ms=(end-prefill_end)/1e6,E2E_ms=(end-start)/1e6,TPOT_ms=(end-prefill_end)/1e6/(tokens-1),
            memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            cache=cache_record(cache),loop_count=args.steps,
            token_intervals_ms=[marks[i].elapsed_time(marks[i+1]) for i in range(tokens-1)] if events else None)
        return result
    if args.smoke:
        ids=rows[0]['prefix_token_ids']['1024'][:32]
        signal.alarm(300)
        a=generate(ids,tokens=4);b=generate(ids,tokens=4,events=True)
        assert a['output_ids']==b['output_ids']
        write(output/'smoke_clean.json',a);write(output/'smoke_events.json',b)
        write(output/'cache_policy_secondary.json',generate(ids,policy='compress-s16',tokens=4))
        signal.alarm(0)
    else:
        for p in args.prefixes:
            for _ in range(3):generate(rows[0]['prefix_token_ids'][str(p)],tokens=4)
            for row in rows:
                for repeat in range(3):
                    signal.alarm(300);value=generate(row['prefix_token_ids'][str(p)]);signal.alarm(0)
                    write(output/f'P{p}.{row["request_id"]}.repeat{repeat}.json',dict(prefix=p,request_id=row['request_id'],repeat=repeat,**value))
                    print(p,row['request_id'],repeat,round(value['E2E_ms'],2),flush=True)
    write(output/'status.json',dict(success=True));os._exit(0)


if __name__=='__main__':main()
