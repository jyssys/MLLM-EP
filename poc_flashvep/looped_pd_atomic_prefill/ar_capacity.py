"""Native AR finite P-only service and valid-resident-KV Decode cohorts."""
import statistics
from .common import *


def main():
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    cfg=config();folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=True)
    os.environ['VLLM_BATCH_INVARIANT']='1' if cfg.get('ar_batch_invariant') else '0'
    write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    import torch
    from vllm import LLM,SamplingParams
    from ..looped_pd_abc.nvml_watch import GPUWatch
    from .ar_worker import llm_options
    torch.set_num_threads(1)
    mode=cfg['capacity_mode'];options=llm_options(cfg,'prefill' if mode=='P_ONLY' else 'capacity')
    options.pop('kv_transfer_config',None)
    # D preparation may need a larger batch budget. It is outside timed D service.
    if mode=='PREPARED_D':options['max_num_batched_tokens']=max(options['max_num_batched_tokens'],max(j['batch'] for j in cfg['jobs'])*cfg['length'])
    monitor=GPUWatch(folder,cfg['gpus'],cfg['name']);monitor.start();engine=None
    try:
        llm=LLM(**options);engine=llm.llm_engine;runner=engine.model_executor.driver_worker.worker.model_runner
        write(folder/'geometry.json',dict(unique_model_parameters=sum(p.numel() for p in runner.model.parameters()),
            weight_bytes=sum(p.numel()*p.element_size() for p in runner.model.parameters()),KV_blocks=runner.kv_cache_config.num_blocks,
            KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),options=options,
            scope='Native local capacity. Prepared D includes no P or KV-transfer time. Its preparation budget can differ from Serving.'))
        data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
        def cohort(job,tag):
            rows=data[:job['batch']];cap=1 if mode=='P_ONLY' else job['cap']
            final={};events={};params=SamplingParams(max_tokens=cap,temperature=0,ignore_eos=True,seed=cfg['seed'])
            for i,row in enumerate(rows):engine.add_request(f'{tag}_{i}',{'prompt_token_ids':row['input_ids']['qwen8']},params)
            torch.cuda.synchronize();start=time.monotonic();p_end=None
            first=engine.step();torch.cuda.synchronize();first_step_end=time.monotonic()
            if mode=='PREPARED_D':
                assert len(first)==len(rows) and all(len(o.outputs[0].token_ids)==1 for o in first),'Valid prepared-D boundary is incomplete'
                p_end=first_step_end
            for o in first:
                events.setdefault(o.request_id,[]).append(dict(time=time.monotonic(),count=len(o.outputs[0].token_ids)))
                if o.finished:final[o.request_id]=list(o.outputs[0].token_ids)
            while engine.has_unfinished_requests():
                for o in engine.step():
                    events.setdefault(o.request_id,[]).append(dict(time=time.monotonic(),count=len(o.outputs[0].token_ids)))
                    if o.finished:final[o.request_id]=list(o.outputs[0].token_ids)
            torch.cuda.synchronize();end=time.monotonic()
            assert len(final)==len(rows) and all(len(v)==cap for v in final.values())
            return dict(tag=tag,model='qwen8',batch=len(rows),job=job,input_ids=[r['input_ids']['qwen8'] for r in rows],
                output_ids={k.rsplit('_',1)[1]:v for k,v in final.items()},token_events=events,
                P_plus_first_sample_s=(first_step_end-start) if mode=='PREPARED_D' else end-start,
                prepared_D_rest_s=end-p_end if p_end is not None else None,
                output_cap=cap,capacity_mode=mode,actual_token_budget=options['max_num_batched_tokens'],
                evidence='MEASURED_NATIVE_LOCAL_CAPACITY',
                scope='P-only generates one first sample then retires. D generates cap-1 tokens from valid resident KV after the first sample.')
        for i,job in enumerate(cfg['jobs']):
            cohort(dict(job,cap=2),f'w{i}')
            reference=None
            for repeat in range(job.get('repeats',3)):
                value=cohort(job,f'j{i}r{repeat}')
                if reference is None:reference=value['output_ids']
                assert value['output_ids']==reference,'Repeated native AR capacity output mismatch'
                value.update(repeat=repeat,repeat_output_parity=True)
                append(folder/'cohorts.jsonl',value);print('COHORT_COMPLETE',cfg['name'],i,repeat,flush=True)
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if engine is not None:
            close=getattr(engine.engine_core,'shutdown',None)
            if close:close()
        monitor.stop()


if __name__=='__main__':main()
