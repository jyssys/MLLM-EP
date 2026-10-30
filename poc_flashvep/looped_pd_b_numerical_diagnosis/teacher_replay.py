"""Teacher-forced native operators on real failure histories, without PD timing."""
import argparse
import gc
import json
import os
import torch
from .common import ROOT,MODELS,MODEL_PINS,setup,write
from .linear_probe import metrics


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',required=True)
    parser.add_argument('--failure-manifest',default='configs/failure_manifest.json')
    parser.add_argument('--sample',type=int,required=True)
    parser.add_argument('--sizes',default='1,2,3,4')
    parser.add_argument('--variant',default='original')
    parser.add_argument('--name',required=True)
    parser.add_argument('--track-all',action='store_true')
    parser.add_argument('--prefill-chunk',type=int,default=0)
    parser.add_argument('--kv-hash',action='store_true')
    args=parser.parse_args()
    setup()
    dest=ROOT/'raw'/args.name
    if (dest/'result.json').exists(): print('REUSE',args.name); return
    dest.mkdir(parents=True,exist_ok=True)
    case=next(r for r in json.loads((ROOT/args.failure_manifest).read_text())
              if r['label']==args.case and r['sample_id']==args.sample)
    model_name=case['left_row']['model']
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    os.environ['B_NUMERICAL_VARIANT']=args.variant
    os.environ['B_NUMERICAL_RUN']=args.name
    from .patches import install
    install()
    from vllm_rlt.models import AutoModelForCausalLM
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.layers.normalization import RMSNorm
    model=AutoModelForCausalLM.from_pretrained(MODELS[model_name],device='cuda',dtype=torch.bfloat16)
    config=model.config
    prompt=case['left_row']['input_ids']
    # Same token history in every geometry, including the input which predicts the first divergent output.
    history=case['left_row']['output_ids'][:case['first_divergence']]
    depth=MODEL_PINS[model_name][2]
    seeds=case['left_row']['seed']
    tracked={}
    comparison=[]
    outputs=[]
    kv_hashes=[]
    context={}
    isolated=[]
    occurrences={}
    hooked=[]
    roles=('.q_proj','.k_proj','.v_proj','.o_proj','.gate_proj','.up_proj','.down_proj',
           '.Wqkv','.attn.proj','.mlp.fc','.mlp.proj','.adapter','lm_head')
    def observe(name,module,inputs,out):
        if not context.get('track'): return
        index=context['target_row']
        if index>=len(out): return
        base_key=(context['phase'],context['step'],context.get('depth',0),name)
        occurrence_key=(context['size'],base_key)
        occurrence=occurrences.get(occurrence_key,0)
        occurrences[occurrence_key]=occurrence+1
        key=(*base_key,occurrence)
        full_first_block=(context['phase']=='prompt' and context.get('depth')==0 and
                          (name.startswith('model.layers.0.') or name.startswith('transformer.prelude.')
                           or name.startswith('attention_kernel.layer_0') or name.startswith('attention_kernel.layer_1')))
        full_small_chunk=context['phase']=='prompt' and context.get('sequence_length',len(prompt))<=16
        selection=slice(0,context.get('sequence_length',len(prompt))) if full_first_block or full_small_chunk else index
        x=inputs[0][selection].detach()
        y=out[selection].detach()
        if context['size']==1:
            tracked[key]=(x.cpu().clone(),y.cpu().clone())
        elif key in tracked:
            a,b=tracked[key]
            value=dict(phase=key[0],step=key[1],depth=key[2],module=name,occurrence=occurrence,rows=context['size'],
                       input=metrics(a.to(x.device),x),output=metrics(b.to(y.device),y),
                       full_first_prompt_sequence=full_first_block,chunk_start=context.get('chunk_start'))
            comparison.append(value)
            if value['input']['exact'] and not value['output']['exact'] and len(isolated)<32 and a.numel()<1000000:
                # Preserve the exact common operand and both raw operator outputs.
                isolated.append(dict(key=key,rows=context['size'],input=a,one=b,many=y.cpu().clone()))
    for name,module in model.named_modules():
        if (isinstance(module,torch.nn.Linear) and any(name.endswith(r) for r in roles)) or isinstance(module,RMSNorm):
            hooked.append(module.register_forward_hook(lambda m,i,o,n=name: observe(n,m,i,o)))
    attend=KVCacheManager._attend_prepared
    def traced_attend(cache,layer,batch,q):
        value=attend(cache,layer,batch,q)
        observe(f'attention_kernel.layer_{layer}',None,(q,),value)
        return value
    KVCacheManager._attend_prepared=traced_attend
    kv_write=KVCacheManager._write_prepared
    def traced_write(cache,layer,batch,k,v):
        observe(f'KV_write_K.layer_{layer}',None,(k,),k)
        observe(f'KV_write_V.layer_{layer}',None,(v,),v)
        return kv_write(cache,layer,batch,k,v)
    KVCacheManager._write_prepared=traced_write
    for size in map(int,args.sizes.split(',')):
        context['size']=size
        token_capacity=len(prompt)+len(history)+2
        pages=(token_capacity+15)//16
        blocks=pages*size*depth
        bytes_needed=blocks*config.num_hidden_layers*16*config.num_key_value_heads*config.head_dim*4
        free,_=torch.cuda.mem_get_info()
        if bytes_needed>free*.75:
            outputs.append(dict(rows=size,status='NOT_RUN_HBM',estimated_KV_bytes=bytes_needed,free_bytes=free))
            continue
        cache=KVCacheManager(config.num_hidden_layers,config.num_key_value_heads,config.head_dim,
             blocks,16,depth,device='cuda',dtype=torch.bfloat16,backend='flash_attn_4',
             recurrent_layers=getattr(model,'recurrent_kv_layers',None))
        ids=[f'case_{i}' for i in range(size)]
        for rid in ids:
            if not cache.allocate(rid,token_capacity): raise RuntimeError('KV allocation failed')
        generators=[torch.Generator(device='cuda').manual_seed(seeds) for _ in ids]
        plen=len(prompt)
        tokens=torch.tensor(prompt*size,device='cuda')
        reqs=[rid for rid in ids for _ in prompt]
        positions=list(range(plen))*size
        pre=cache._prepare_batch(reqs,[0]*len(reqs),positions,packed_prefill=True)
        if args.prefill_chunk and model_name!='ouro':
            raise ValueError('The chunk-tail isolator currently supports Ouro only')
        chunks=range(0,plen,args.prefill_chunk) if args.prefill_chunk else (0,)
        for left in chunks:
            current=prompt[left:left+args.prefill_chunk] if args.prefill_chunk else prompt
            width=len(current)
            tokens=torch.tensor(current*size,device='cuda')
            reqs=[rid for rid in ids for _ in current]
            positions=list(range(left,left+width))*size
            pre=cache._prepare_batch(reqs,[0]*len(reqs),positions,packed_prefill=True)
            context.update(phase='prompt',step=-1,depth=0,target_row=width-1,sequence_length=width,chunk_start=left,track=True)
            if getattr(model,'requires_boundary_kv',False):
                hidden=model.prelude_prepared(tokens,pre,cache,generators=[g for g in generators for _ in current])
            else: hidden=model.prelude(tokens)
            for d in range(depth):
                context['depth']=d
                kv=cache._prepare_batch(reqs,[d]*len(reqs),positions,packed_prefill=True)
                hidden,_=model.recurrent_prepared(hidden,kv,cache,compute_gate=False)
        if getattr(model,'requires_boundary_kv',False):
            # Populate earlier prompt coda KV. Final prompt coda is executed below.
            keep=[j for i in range(size) for j in range(i*plen,(i+1)*plen-1)]
            if keep:
                context['track']=False
                kv=cache._prepare_batch([reqs[i] for i in keep],[0]*len(keep),[positions[i] for i in keep],packed_prefill=True)
                model.coda_prepared(hidden[keep],kv,cache,compute_logits=False)
        hidden=hidden[torch.tensor([(i+1)*width-1 for i in range(size)],device='cuda')]
        if args.kv_hash:
            import hashlib
            entries=[]
            for d in range(depth):
                for layer in range(config.num_hidden_layers):
                    if d and layer not in getattr(model,'recurrent_kv_layers',range(config.num_hidden_layers)):continue
                    valid=plen-1 if layer in getattr(model,'coda_kv_layers',()) else plen
                    if not valid:continue
                    k,v=cache.read(layer,ids[0],d,length=valid)
                    entries.append(dict(depth=d,layer=layer,valid_tokens=valid,
                        key_sha256=hashlib.sha256(k.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                        value_sha256=hashlib.sha256(v.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()))
            kv_hashes.append(dict(rows=size,entries=entries,
                final_hidden_sha256=hashlib.sha256(hidden[0].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()))
        context.update(phase='coda',step=0,depth=depth,target_row=0,track=True,chunk_start=None)
        if getattr(model,'requires_boundary_kv',False):
            kv=cache._prepare_batch(ids,[0]*size,[plen-1]*size)
            logits=model.coda_prepared(hidden,kv,cache)
        else: logits=model.coda(hidden)
        for step,token in enumerate(history,1):
            position=plen-1+step
            context.update(phase='decode',step=step,depth=0,target_row=0,track=args.track_all or step in (1,len(history)))
            tokens=torch.full((size,),token,device='cuda',dtype=torch.long)
            if getattr(model,'requires_boundary_kv',False):
                kv=cache._prepare_batch(ids,[0]*size,[position]*size)
                hidden=model.prelude_prepared(tokens,kv,cache,generators=generators)
            else: hidden=model.prelude(tokens)
            for d in range(depth):
                context['depth']=d
                kv=cache._prepare_batch(ids,[d]*size,[position]*size)
                hidden,_=model.recurrent_prepared(hidden,kv,cache,compute_gate=False)
            context.update(phase='coda',depth=depth)
            if getattr(model,'requires_boundary_kv',False):
                kv=cache._prepare_batch(ids,[0]*size,[position]*size)
                logits=model.coda_prepared(hidden,kv,cache)
            else: logits=model.coda(hidden)
        top,indices=logits[0].float().topk(2)
        outputs.append(dict(rows=size,status='COMPLETE',top1=int(logits[0].argmax()),topk_first=int(indices[0]),topk_second=int(indices[1]),
                            margin=float(top[0]-top[1]),KV_bytes=bytes_needed,
                            all_clone_rows_logit_exact=all(torch.equal(logits[0],r) for r in logits)))
        print(args.name,'rows',size,'argmax',int(logits[0].argmax()),flush=True)
        del cache,hidden,logits,pre,kv
        gc.collect();torch.cuda.empty_cache()
    torch.save(isolated,dest/'isolated_same_operand_differences.pt')
    first={}
    for row in comparison:
        if not row['output']['exact']: first.setdefault(str(row['rows']),row)
    write(dest/'result.json',dict(case=args.case,sample=args.sample,variant=args.variant,
          forced_history=history,input_ids=prompt,prefill_chunk=args.prefill_chunk,outputs=outputs,first_observed_difference=first,
          comparisons=comparison,prefill_KV_hashes=kv_hashes,
          scope='Instrumented eager teacher-forced replay of actual failure prompts/history. Not clean PD reproduction. Only selected operations/steps traced.',
          torch=torch.__version__,CUDA=torch.version.cuda,evidence='MEASURED_OPERATOR_REPLAY',
          greedy_definition='torch.argmax returns the first maximal index. Topk ordering within a tie is not greedy argmax.'))


if __name__=='__main__': main()
