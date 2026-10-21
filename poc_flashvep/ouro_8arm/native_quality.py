"""Real greedy task generations with native static TP1 or TP4. Not frozen feedback."""
import argparse
import time
import statistics
from .contract import *

def main():
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    p=argparse.ArgumentParser();p.add_argument('--tp',type=int,choices=[1,4],required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--label',required=True)
    p.add_argument('--cap',type=int,default=4096);p.add_argument('--nsys',action='store_true')
    p.add_argument('--extension400',action='store_true');a=p.parse_args()
    arm='S4' if a.tp==1 else 'T4';rows=dataset(not a.full)
    if a.extension400:rows=subset400(extension_only=True)
    if a.limit:rows=rows[:a.limit]
    path=ROOT/'raw'/a.label;path.mkdir(exist_ok=False)
    start=time.perf_counter()
    llm=LLM(model=str(MODEL),trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=a.tp,
        hf_overrides={'total_ut_steps':4,'early_exit_threshold':1.},max_model_len=8192,max_num_seqs=4,
        max_num_batched_tokens=8192,kv_cache_memory_bytes=(16 if a.tp==1 else 4)*2**30,gpu_memory_utilization=.8,
        enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
        worker_extension_cls='poc_flashvep.ouro_8arm.native_probe.Probe',disable_log_stats=True)
    tok=llm.get_tokenizer();write(path/'setup.json',dict(wall_s=time.perf_counter()-start,arm=arm,engine='native vLLM 0.26.0',
        graphs='Native default graph/compile mode',IDs=[r['id'] for r in rows],cap=a.cap,status='NUMERICALLY_UNQUALIFIED_EXPLORATORY' if a.tp==4 else 'BACKEND_CONTROL',
        ownership=llm.collective_rpc('inspect_ownership')))
    llm.generate(['Numerical warm-up: 1 + 1 ='],SamplingParams(temperature=0,max_tokens=8),use_tqdm=False)
    original=llm.llm_engine.step;times={}
    def step(*args,**kwargs):
        # LLM._add_request forces FINAL_ONLY. Override only frontend emission,
        # not model sampling, so TTFT is not mislabeled as completion time.
        for state in llm.llm_engine.output_processor.request_states.values():state.output_kind=RequestOutputKind.CUMULATIVE
        values=original(*args,**kwargs);now=time.perf_counter()
        for value in values:
            if value.outputs:
                count=len(value.outputs[0].token_ids)
                if count:times.setdefault(value.request_id,[]).append((count,now))
        return values
    llm.llm_engine.step=step
    for offset in range(0,len(rows),4):
        group=rows[offset:offset+4];times.clear()
        cohort_index=group[0].get('cohort400',offset//4)
        params=[SamplingParams(temperature=0,max_tokens=a.cap,stop=r['stop_strings'] or None,include_stop_str_in_output=True) for r in group]
        inputs=[dict(prompt_token_ids=r.get('input_ids') or tok.encode(r['prompt'],add_special_tokens=False)) for r in group]
        llm.collective_rpc('safety_start')
        if a.nsys:llm.collective_rpc('profile_begin')
        begin=time.perf_counter();output=llm.generate(inputs,params,use_tqdm=False);end=time.perf_counter()
        finite=llm.collective_rpc('safety_finish');assert all(finite),finite
        if a.nsys:llm.collective_rpc('profile_end')
        cohort=[]
        for row,prompt,result in zip(group,inputs,output):
            value=result.outputs[0];tt=times.get(result.request_id,[])
            assert tt and value.token_ids and all(0<=v<49152 for v in value.token_ids)
            first,last=tt[0][1],tt[-1][1];count=len(value.token_ids)
            record=dict(id=row['id'],dataset=row['dataset'],arm=arm,engine='vLLM_0.26.0',GPUs=[4] if a.tp==1 else [4,5,6,7],
                prompt_sha256=row['prompt_sha256'],prompt_ids=prompt['prompt_token_ids'],output_ids=list(value.token_ids),text=value.text,
                finish_reason='eos' if value.token_ids[-1]==tok.eos_token_id else value.finish_reason,
                native_finish_reason=value.finish_reason,stop_reason=value.stop_reason,output_tokens=count,mean_depth=4.,
                max_new_tokens=a.cap,evaluation_eligible=a.cap==4096,
                E2E_s=last-begin,TTFT_s=first-begin,TPOT_s=(last-first)/(count-1) if count>1 else None,
                cohort_makespan_s=end-begin,cohort_index=cohort_index,cohort_C=len(group),
                token_publications=[dict(tokens=n,t_s=t-begin) for n,t in tt],finite_logits=True,
                status='NUMERICALLY_UNQUALIFIED_EXPLORATORY' if a.tp==4 else 'BACKEND_CONTROL',
                lane='TASK_GENERATION_WITH_GPU_FINITE_OBSERVER. Not an observer-free hardware speedup.')
            write(path/(row['id'].replace('/','_')+'.json'),record);cohort.append(record['id'])
            print('QUALITY',arm,row['id'],count,value.finish_reason,flush=True)
        write(path/f'cohort_{cohort_index:03d}.json',dict(ids=cohort,makespan_s=end-begin,output_tokens=sum(len(r.outputs[0].token_ids) for r in output)))
    llm.llm_engine.step=original
    write(path/'after_ownership.json',llm.collective_rpc('inspect_ownership'))
    llm.llm_engine.engine_core.shutdown()
    write(path/'complete.json',dict(success=True,requests=len(rows),arm=arm))

if __name__=='__main__':main()
