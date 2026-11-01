"""Measure native projection geometry on captured real Qwen operands."""
from .common import *


def main():
    cfg=config();folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_BATCH_INVARIANT']='1' if cfg.get('ar_batch_invariant') else '0'
    write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
    import torch
    from vllm import LLM
    from .ar_worker import llm_options
    torch.set_num_threads(1);engine=None
    try:
        options=llm_options(cfg,'capacity');options.update(enforce_eager=True,max_num_seqs=4)
        llm=LLM(**options);engine=llm.llm_engine;model=engine.model_executor.driver_worker.worker.model_runner.model
        source=ROOT/'raw/ext_qwen_default_eager_teacher_prefix_probe_v1/selected_native_tensors.pt'
        capture=torch.load(source,map_location='cpu',weights_only=True)
        def first(value):
            if torch.is_tensor(value):return value
            if isinstance(value,list):
                for item in value:
                    result=first(item)
                    if result is not None:return result
            return None
        vectors={r['operation']:first(r['input']) for r in capture['traces']['alone']}
        final=next(r for r in capture['traces']['alone'] if r['operation']=='model.norm')
        vectors['LM_HEAD_DIAGNOSTIC']=first(final['output'])
        result=[]
        for name,vector in vectors.items():
            if not name.endswith(('qkv_proj','o_proj','gate_up_proj','down_proj','LM_HEAD_DIAGNOSTIC')):continue
            if '.layers.0.' not in name and '.layers.31.' not in name and name!='LM_HEAD_DIAGNOSTIC':continue
            module=model.lm_head if name=='LM_HEAD_DIAGNOSTIC' else model.get_submodule(name)
            x=vector.to(device='cuda',dtype=torch.bfloat16).reshape(1,-1)
            weight=module.weight
            def apply(value):
                if name=='LM_HEAD_DIAGNOSTIC':return model.logits_processor._apply_head(model.lm_head,value,None)
                out=module(value);return out[0] if isinstance(out,tuple) else out
            reference=apply(x).detach().float()
            for batch in (1,2,3,4,8,16,32,256):
                operand=x.repeat(batch,1);out=apply(operand).detach().float();diff=out[0]-reference[0]
                for _ in range(3):apply(operand)
                times=[]
                for _ in range(10):
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    a.record();apply(operand);b.record();b.synchronize();times.append(a.elapsed_time(b))
                top=torch.topk(out[0],2) if name=='LM_HEAD_DIAGNOSTIC' else None
                result.append(dict(operation=name,batch=batch,input_width=x.shape[-1],weight_shape=list(weight.shape),
                    exact=bool(torch.equal(out[0],reference[0])),exact_element_fraction=float((out[0]==reference[0]).float().mean()),
                    max_absolute_error=float(diff.abs().max()),relative_L2_error=float(diff.norm()/reference[0].norm().clamp_min(1e-30)),
                    CUDA_event_ms_mean=sum(times)/len(times),within_start_samples=len(times),
                    top_ids=top.indices.tolist() if top is not None else None,top1_top2_margin=float(top.values[0]-top.values[1]) if top is not None else None,
                    evidence='MEASURED_PROJECTION_MICROBENCHMARK',scope='Actual checkpoint weights and an actual captured operand. Repeated rows isolate GEMM M geometry. Not model E2E speedup.'))
        write(folder/'linear_geometry.json',dict(options=options,source=str(source),source_SHA256=digest(source),measurements=result))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',time=time.time(),error=repr(error)));raise
    finally:
        if engine is not None:
            close=getattr(engine.engine_core,'shutdown',None)
            if close:close()


if __name__=='__main__':main()
