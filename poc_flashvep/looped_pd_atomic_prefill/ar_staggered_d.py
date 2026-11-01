"""Measure finite staggered Decode after genuine resident-prefix preparation."""
from .common import *


def main():
    cfg=config();folder=ROOT/'raw'/cfg['name']
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_BATCH_INVARIANT']='0'
    for path,expected in json.loads(os.environ.get('ATOMIC_SOURCE_HASHES','{}')).items():
        assert digest(path)==expected,'Source changed before prepared Decode startup'
    import torch
    from vllm import LLM,SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from .ar_worker import llm_options
    from ..looped_pd_abc.nvml_watch import GPUWatch
    torch.set_num_threads(1);hits={};get_blocks=KVCacheManager.get_computed_blocks
    def observed(self,request):
        result=get_blocks(self,request);hits[request.request_id.split('-',1)[0]]=result[1];return result
    KVCacheManager.get_computed_blocks=observed
    options=llm_options(cfg,'capacity');options['enable_prefix_caching']=True
    watch=GPUWatch(folder,cfg['gpus'],cfg['name']);watch.start();engine=None
    try:
        llm=LLM(**options);engine=llm.llm_engine;runner=engine.model_executor.driver_worker.worker.model_runner
        block=min(g.kv_cache_spec.block_size for g in runner.kv_cache_config.kv_cache_groups)
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
        write(folder/'geometry.json',dict(options=options,KV_blocks=runner.kv_cache_config.num_blocks,
            KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),
            cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),physical_GPU=cfg['gpus'][0],
            scope='Official resident-prefix preparation. Not the main NIXL Serving path. Cached final suffix recomputation remains inside this diagnostic.'))
        # Sequential original-model execution constructs all retained KV.
        for i,row in enumerate(data):
            rid=f'prepare_{i}'
            engine.add_request(rid,{'prompt_token_ids':row['input_ids']['qwen8']},
                SamplingParams(max_tokens=2,temperature=0,ignore_eos=True,seed=cfg['seed']+i))
            while engine.has_unfinished_requests():engine.step()
        torch.cuda.synchronize();start=time.monotonic();pending=0;first={};events={};completed={}
        while pending<len(data) or engine.has_unfinished_requests():
            now=time.monotonic()
            while pending<len(data) and now>=start+cfg['prepared_D_arrival_offsets'][pending]:
                rid=f'decode_{pending}';row=data[pending]
                engine.add_request(rid,{'prompt_token_ids':row['input_ids']['qwen8']},
                    SamplingParams(max_tokens=cfg['cap'],temperature=0,ignore_eos=True,seed=cfg['seed']+pending))
                append(folder/'prepared_D_release_events.jsonl',dict(index=pending,
                    scheduled=start+cfg['prepared_D_arrival_offsets'][pending],actual=time.monotonic()))
                pending+=1
            for out in engine.step():
                now=time.monotonic();rid=out.request_id;index=int(rid.rsplit('_',1)[1]);count=len(out.outputs[0].token_ids)
                first.setdefault(rid,now);events.setdefault(rid,[]).append(dict(time=now,count=count))
                assert hits[rid]>=cfg['length']-block,'Valid cached Prompt KV is absent'
                if not out.finished:continue
                arrival=start+cfg['prepared_D_arrival_offsets'][index]
                value=dict(sample_id=index,seed=cfg['seed']+index,input_ids=data[index]['input_ids']['qwen8'],
                    output_ids=list(out.outputs[0].token_ids),arrival=arrival,admission=arrival,first_token=first[rid],finish=now,
                    input_tokens=cfg['length'],output_tokens=count,ttft_ms=1000*(first[rid]-arrival),
                    tpot_ms=1000*(now-first[rid])/max(1,count-1),e2e_ms=1000*(now-arrival),token_events=events[rid],
                    valid_resident_prefix_tokens=hits[rid],evidence='MEASURED_PREFIX_PREPARED_STAGGERED_D_ONLY')
                assert count==cfg['cap'];append(folder/'prepared_D_requests.jsonl',value);completed[rid]=value
            assert time.monotonic()-start<cfg['timeout'],'Staggered Decode watchdog'
            if not engine.has_unfinished_requests():time.sleep(.0005)
        assert len(completed)==cfg['count']
        write(folder/'prepared_D_window.json',dict(start=start,end=time.monotonic(),
            last_arrival=start+cfg['prepared_D_arrival_offsets'][-1],scope='Finite prepared-KV cohort. Preparation excluded. Not sustainable capacity.'))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if engine is not None:
            close=getattr(engine.engine_core,'shutdown',None)
            if close:close()
        watch.stop()


if __name__=='__main__':main()
