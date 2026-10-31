"""Native vLLM dense-AR diagnostic. It is not the RLT execution engine."""
import sys
import gc
from .common import *

def main():
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    import torch
    from vllm import LLM,SamplingParams
    from ..looped_pd_abc.nvml_watch import GPUWatch
    cfg=json.loads(Path(sys.argv[1]).read_text());setup();folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    torch.set_num_threads(1);torch.cuda.set_device(0)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[cfg['gpus'][0]].removeprefix('GPU-')
    checkpoint=json.loads((ROOT/'configs/dense_ar_checkpoint.json').read_text())['path']
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start()
    try:
        llm=LLM(model=checkpoint,dtype='bfloat16',max_model_len=cfg.get('max_model_len',4096),max_num_seqs=cfg.get('slots',8),
            max_num_batched_tokens=cfg.get('token_budget',32768),enable_chunked_prefill=True,enable_prefix_caching=False,
            async_scheduling=False,gpu_memory_utilization=.65,seed=cfg['seed'],
            compilation_config={'cudagraph_capture_sizes':cfg.get('capture_sizes',[1,2,4,8])})
        engine=llm.llm_engine
        runner=engine.model_executor.driver_worker.worker.model_runner
        write(folder/'geometry.json',dict(unique_model_parameters=sum(p.numel() for p in runner.model.parameters()),
            weight_bytes=sum(p.numel()*p.element_size() for p in runner.model.parameters()),
            KV_dtype=str(runner.kv_cache_dtype),KV_blocks=runner.kv_cache_config.num_blocks,
            KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),dtype='BF16',
            scope='Native vLLM allocations. KV is paged. This model has no recurrent loop.'))
        write(folder/'engine.json',dict(engine='vLLM 0.26.0 native Qwen3',revision=QWEN_PIN[1],
            prefix_caching=False,chunked_prefill=True,async_scheduling=False,
            GPU=cfg['gpus'][0],uuid=str(torch.cuda.get_device_properties(0).uuid),
            context_limit=4096,scope='Dense AR diagnostic. Engine and kernels differ from Looped RLT.'))
        def run(job,tag):
            rows=json.loads((ROOT/'data'/job['dataset']).read_text())[:job['batch']]
            params=SamplingParams(max_tokens=job['cap'],temperature=0,ignore_eos=True,seed=cfg['seed'])
            for i,row in enumerate(rows):
                engine.add_request(f'{tag}_{i}',{'prompt_token_ids':row['input_ids']['qwen']},params)
            first={};events={};final={};torch.cuda.synchronize();t0=time.monotonic()
            first_outputs=engine.step();torch.cuda.synchronize();t1=time.monotonic()
            assert len(first_outputs)==len(rows),'The prefill step did not serve the complete cohort'
            for out in first_outputs:
                assert len(out.outputs[0].token_ids)==1,'The prefill step did not produce exactly one token'
                first[out.request_id]=time.monotonic();events[out.request_id]=[]
            while engine.has_unfinished_requests():
                for out in engine.step():
                    now=time.monotonic();events.setdefault(out.request_id,[]).append(now)
                    if out.finished:final[out.request_id]=list(out.outputs[0].token_ids)
            torch.cuda.synchronize();t2=time.monotonic()
            assert len(final)==len(rows) and all(len(v)==job['cap'] for v in final.values())
            return dict(tag=tag,job=job,model='qwen',batch=len(rows),input_ids=[r['input_ids']['qwen'] for r in rows],
                output_ids={k.rsplit('_',1)[1]:v for k,v in final.items()},
                prefill_plus_first_token_service_s=t1-t0,decode_rest_service_s=t2-t1,
                timed_decode_tokens_per_request=job['cap']-1,
                decode_ITL_s=[b-a for values in events.values() for a,b in zip(values,values[1:])],
                scope='P includes first sampling. All initial prompt tokens fit the budget. D excludes prefill and first token. Native vLLM supported chunking is enabled.',
                evidence='MEASURED_NATIVE_VLLM',time=time.time())
        for j,job in enumerate(cfg['jobs']):
            run(dict(job,cap=2),f'w{j}')
            baseline=None
            for repeat in range(job.get('repeats',3)):
                os.environ['ABC_MEASURED_ACTIVE']='1'
                value=run(job,f'j{j}r{repeat}')
                os.environ['ABC_MEASURED_ACTIVE']='0'
                if baseline is None:baseline=value['output_ids']
                assert value['output_ids']==baseline,'Repeated native vLLM output parity failed'
                value['repeat']=repeat;value['repeat_output_parity']=True
                append(folder/'cohorts.jsonl',value);print('AR_COHORT_COMPLETE',j,repeat,flush=True)
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:monitor.stop()

if __name__=='__main__':main()
