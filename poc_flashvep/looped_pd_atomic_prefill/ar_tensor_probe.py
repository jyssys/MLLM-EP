"""Compare actual native Qwen tensors on a common teacher-forced prefix."""
import hashlib
from .common import *


def main():
    cfg=config();folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_BATCH_INVARIANT']='1' if cfg.get('ar_batch_invariant') else '0'
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    import torch
    from vllm import LLM,SamplingParams
    from .ar_worker import llm_options
    torch.set_num_threads(1);engine=None;handles=[]
    try:
        original=json.loads((ROOT/'state/ar_numerical_debug_original_failures.json').read_text())
        failure=next(r for r in original[0]['failures'] if r['sample_id']==13)
        target=failure['common_teacher_forced_history'];length=len(target)
        pool=json.loads((ROOT/'data/test_qwen8_L8000.json').read_text())
        filler=pool[0]['input_ids']['qwen8']+target[-(length-8000):]
        assert len(filler)==length
        options=llm_options(cfg,'capacity');options.update(enforce_eager=True,max_model_len=length+2,max_num_batched_tokens=32768,max_num_seqs=4)
        options.pop('kv_transfer_config',None)
        llm=LLM(**options);engine=llm.llm_engine;model=engine.model_executor.driver_worker.worker.model_runner.model
        selected={'capture':False,'index':length-1,'batch_index':0};traces={};logits={}
        def vectors(value,index):
            if torch.is_tensor(value):
                if value.ndim>=2 and value.shape[0]>index:return value[index].detach().float().cpu().contiguous()
                return None
            if isinstance(value,(tuple,list)):
                return [vectors(v,index) for v in value if torch.is_tensor(v)]
            return None
        def hook(name):
            def capture(module,inputs,output):
                if not selected['capture']:return
                # Q/K norms flatten heads. Use the exact head range rather
                # than treating a head index as a token row.
                first=next((v for v in inputs if torch.is_tensor(v) and v.ndim>=2),None)
                index=selected['index']
                if first is None or first.shape[0]<=index:return
                if first.ndim==3:
                    input_value=first[index].detach().float().cpu().contiguous()
                    output_value=output[index].detach().float().cpu().contiguous() if torch.is_tensor(output) else vectors(output,index)
                else:input_value=vectors(inputs,index);output_value=vectors(output,index)
                traces.setdefault(selected['label'],[]).append(dict(operation=name,input=input_value,output=output_value))
            return capture
        endings=('qkv_proj','o_proj','gate_up_proj','down_proj','input_layernorm','post_attention_layernorm','q_norm','k_norm','attn','norm')
        for name,module in model.named_modules():
            if name.endswith(endings):handles.append(module.register_forward_hook(hook(name)))
        compute=model.compute_logits
        def compute_logits(*args,**kw):
            result=compute(*args,**kw)
            if selected['capture'] and result is not None:
                logits[selected['label']]=result[selected['batch_index']].detach().float().cpu().contiguous()
            return result
        model.compute_logits=compute_logits
        llm.generate([dict(prompt_token_ids=target)],SamplingParams(max_tokens=1,temperature=0,ignore_eos=True),use_tqdm=False)
        outputs={}
        for label,prompts,index,batch_index in [('alone',[target],length-1,0),('alone_repeat',[target],length-1,0),
            ('target_first',[target,filler],length-1,0),('target_second',[filler,target],2*length-1,1)]:
            selected.update(capture=True,label=label,index=index,batch_index=batch_index)
            result=llm.generate([dict(prompt_token_ids=p) for p in prompts],SamplingParams(max_tokens=1,temperature=0,ignore_eos=True,logprobs=2),use_tqdm=False)
            selected['capture']=False
            outputs[label]=list(result[batch_index].outputs[0].token_ids)
        def leaves(value):
            if torch.is_tensor(value):return [value]
            if isinstance(value,list):return [t for v in value for t in leaves(v)]
            return []
        def error(a,b):
            x=leaves(a);y=leaves(b)
            if len(x)!=len(y) or any(u.shape!=v.shape for u,v in zip(x,y)):return dict(comparable=False)
            if not x:return dict(comparable=False)
            u=torch.cat([t.flatten() for t in x]);v=torch.cat([t.flatten() for t in y]);d=u-v
            return dict(comparable=True,exact=bool(torch.equal(u,v)),exact_element_fraction=float((u==v).float().mean()),
                max_absolute_error=float(d.abs().max()),relative_L2_error=float(d.norm()/u.norm().clamp_min(1e-30)))
        comparisons=[]
        for label in ('alone_repeat','target_first','target_second'):
            a=traces['alone'];b=traces[label];assert [r['operation'] for r in a]==[r['operation'] for r in b],'Probe did not observe matching native operations'
            for x,y in zip(a,b):
                comparisons.append(dict(candidate=label,operation=x['operation'],input=error(x['input'],y['input']),output=error(x['output'],y['output'])))
        logit_comparisons={}
        for label,value in logits.items():
            top=torch.topk(value,2)
            logit_comparisons[label]=dict(error=error(logits['alone'],value),top_ids=top.indices.tolist(),top_values=top.values.tolist(),top1_top2_margin=float(top.values[0]-top.values[1]))
        torch.save(dict(traces=traces,logits=logits),folder/'selected_native_tensors.pt')
        write(folder/'tensor_comparison.json',dict(options=options,target_sample=13,original_divergent_index=failure['first_divergent_output_index'],
            common_prefix_tokens=length,input_ids=target,outputs=outputs,comparisons=comparisons,logits=logit_comparisons,
            evidence='MEASURED_NATIVE_EAGER_TEACHER_FORCED_PREFIX',
            scope='Original Qwen checkpoint and native modules. Selected target-token rows only. Cold full-prefix evaluation does not reproduce incremental Decode or identify the first difference over all token rows. Eager tracing is not clean Serving latency.'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error)));raise
    finally:
        for handle in handles:handle.remove()
        if engine is not None:
            close=getattr(engine.engine_core,'shutdown',None)
            if close:close()


if __name__=='__main__':main()
