"""Measure native D batches after validated resident-prefix preparation.

Preparation is sequential and outside timed Decode. This avoids a huge B32
Prefill activation peak that is not part of a native PD Decode worker.
"""
from .common import *


def main():
    cfg=config();os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_BATCH_INVARIANT']='0'
    hashes=json.loads(os.environ.get('ATOMIC_SOURCE_HASHES','{}'))
    for path,expected in hashes.items():assert digest(path)==expected,'Source changed before prefix-capacity startup'
    folder=ROOT/'raw'/cfg['name'];write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    write(folder/'loaded_source.json',dict(time=time.time(),pid=os.getpid(),hashes=hashes,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    import torch
    from vllm import LLM,SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from .ar_worker import llm_options
    from ..looped_pd_abc.nvml_watch import GPUWatch
    torch.set_num_threads(1);hits={};original=KVCacheManager.get_computed_blocks
    def observed(self,request):
        value=original(self,request);hits[request.request_id.split('-',1)[0]]=value[1];return value
    KVCacheManager.get_computed_blocks=observed
    options=llm_options(cfg,'capacity');options.update(enable_prefix_caching=True)
    watch=GPUWatch(folder,cfg['gpus'],cfg['name']);watch.start();engine=None
    try:
        llm=LLM(**options);engine=llm.llm_engine;runner=engine.model_executor.driver_worker.worker.model_runner
        block=min(g.kv_cache_spec.block_size for g in runner.kv_cache_config.kv_cache_groups)
        write(folder/'geometry.json',dict(unique_model_parameters=sum(p.numel() for p in runner.model.parameters()),
            weight_bytes=sum(p.numel()*p.element_size() for p in runner.model.parameters()),KV_blocks=runner.kv_cache_config.num_blocks,
            KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),options=options,block_size=block,
            cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),physical_GPU=cfg['gpus'][0],model_revision=cfg['model_revision'],
            scope='Official prefix-cache preparation only in this D-only control. Main Serving still disables prefix caching. Timed D excludes suffix recomputation and first sampling.'))
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
        def prepare(job,tag):
            for i,row in enumerate(data[:job['batch']]):
                rid=f'{tag}_prepare_{i}'
                engine.add_request(rid,{'prompt_token_ids':row['input_ids']['qwen8']},SamplingParams(max_tokens=1,temperature=0,ignore_eos=True,seed=cfg['seed']+i))
                completed=False
                while engine.has_unfinished_requests():
                    for out in engine.step():
                        if out.finished:assert out.request_id==rid;completed=True
                assert completed
            torch.cuda.synchronize()
        def cohort(job,tag):
            prepare_start=time.monotonic();prepare(job,tag);cached={};outputs={};events={};cap=job['cap']
            for i,row in enumerate(data[:job['batch']]):
                rid=f'{tag}_decode_{i}'
                engine.add_request(rid,{'prompt_token_ids':row['input_ids']['qwen8']},SamplingParams(max_tokens=cap,temperature=0,ignore_eos=True,seed=cfg['seed']+i))
            first=engine.step();torch.cuda.synchronize();start=time.monotonic()
            assert len(first)==job['batch'] and all(len(o.outputs[0].token_ids)==1 for o in first),'Resident-prefix D boundary is incomplete'
            for out in first:
                cached[out.request_id]=hits[out.request_id]
                assert cached[out.request_id]>=cfg['length']-block,'Prepared Prompt KV did not remain resident'
                events[out.request_id]=[dict(time=start,count=1)]
            while engine.has_unfinished_requests():
                for out in engine.step():
                    events[out.request_id].append(dict(time=time.monotonic(),count=len(out.outputs[0].token_ids)))
                    if out.finished:outputs[out.request_id.rsplit('_',1)[1]]=list(out.outputs[0].token_ids)
            torch.cuda.synchronize();end=time.monotonic()
            assert len(outputs)==job['batch'] and all(len(v)==cap for v in outputs.values())
            return dict(job=job,batch=job['batch'],capacity_mode='PREPARED_D',prepared_D_rest_s=end-start,
                P_plus_first_sample_s=start-prepare_start,token_events=events,output_ids=outputs,output_cap=cap,
                valid_resident_prefix_tokens=cached,actual_token_budget=options['max_num_batched_tokens'],
                capacity_preparation='PREFIX_RESIDENT_RECOMPUTED_SUFFIX',evidence='MEASURED_NATIVE_PREFIX_PREPARED_D_ONLY',
                scope='Every cached prefix was actually computed by the original BF16 model. No synthetic KV. Final suffix and first sample are outside the D-rest timer. Prefix caching remains a separate preparation-path control.')
        for j,job in enumerate(cfg['jobs']):
            cohort(dict(job,cap=2),f'j{j}warm');reference=None
            for repeat in range(job.get('repeats',3)):
                value=cohort(job,f'j{j}r{repeat}')
                if reference is None:reference=value['output_ids']
                assert value['output_ids']==reference,'Repeated resident-prefix Decode output mismatch'
                value.update(status='COMPLETE',repeat=repeat,repeat_output_parity=True)
                append(folder/'cohorts.jsonl',value);print('COHORT_COMPLETE',cfg['name'],j,repeat,flush=True)
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if engine is not None:
            close=getattr(engine.engine_core,'shutdown',None)
            if close:close()
        watch.stop()


if __name__=='__main__':main()
