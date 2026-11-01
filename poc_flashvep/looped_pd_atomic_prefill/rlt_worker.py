"""Drive qualified RLT Full-Prefill PD without a new migration policy."""
import sys
import collections
from .common import *


def serve(engine,inputs,args,run,warmup=False,contract=None):
    from vllm_rlt import SamplingParams
    cfg=contract or config();n=min(len(inputs),cfg.get('warmup_count',2)) if warmup else len(inputs)
    offsets=[0.]*n if warmup else cfg['arrival_offsets'];caps=[cfg.get('warmup_cap',2)]*n if warmup else cfg['request_caps']
    serial=cfg.get('calibration_serial') and not warmup
    folder=ROOT/'raw'/cfg['name'];start=time.monotonic();index=0;active={};out={};events={};first={};last={};admitted={};arrivals={}
    observation=start
    if cfg.get('nsys') and not warmup:
        import torch
        torch.cuda.nvtx.range_push('APD_MEASURED_WINDOW')
    try:
        while index<n or active:
            now=time.monotonic()
            while index<n and now>=start+offsets[index]:
                if serial and active:break
                rid=f'r{run}_q{index}';arrival=now if serial else start+offsets[index]
                params=SamplingParams(max_tokens=caps[index],temperature=0.,seed=cfg['seed']+index,
                    min_loops=cfg['depth'],max_loops=cfg['depth'],exit_threshold=1.,ignore_eos=True)
                send_start=time.monotonic();engine.add_request(rid,inputs[index],params)
                active[rid]=index;admitted[rid]=time.monotonic();arrivals[rid]=arrival;events[rid]=[]
                if not warmup:append(folder/'offered_requests.jsonl',dict(sample_id=index,prompt_id=cfg['prompt_ids'][index],
                    arrival=arrival,actual_send_start=send_start,actual_send_end=admitted[rid],admission=admitted[rid],cap=caps[index],seed=cfg['seed']+index))
                index+=1
                if serial:break
            for output in engine.step():
                now=time.monotonic();rid=output.request_id;i=active[rid];count=len(output.token_ids)
                previous=last.get(rid,(now,0))
                if count>previous[1]:
                    first.setdefault(rid,now);events[rid].append(dict(time=now,previous_time=previous[0],new_tokens=count-previous[1],first=previous[1]==0))
                    last[rid]=(now,count)
                if not output.finished:continue
                assert count==caps[i] and set(output.exit_depths)=={cfg['depth']},'Full-depth or output work changed'
                active.pop(rid)
                row=dict(model=cfg['model'],name=cfg['name'],arm=cfg['arm'],run=run,sample_id=i,warmup=warmup,
                    prompt_id=cfg['prompt_ids'][i],seed=cfg['seed']+i,input_ids=inputs[i],output_ids=list(output.token_ids),
                    exit_depths=list(output.exit_depths),finish_reason=str(output.finish_reason),input_tokens=len(inputs[i]),output_tokens=count,
                    arrival=arrivals[rid],admission=admitted[rid],first_token=first[rid],finish=now,
                    ttft_ms=1000*(first[rid]-arrivals[rid]),tpot_ms=1000*(now-first[rid])/max(1,count-1),e2e_ms=1000*(now-arrivals[rid]),
                    token_events=events[rid],chunk=cfg['chunk'],chunk_label=cfg['chunk_label'],token_budget=cfg['token_budget'],
                    recurrent_depth=cfg['depth'],atomic_prompt_requested=cfg['atomic_prompt'],gpus=cfg['gpus'],
                    generation='CONTROLLED_IGNORE_EOS',timing_scope='Scheduled arrival to coordinator delivery. Includes IPC, admission, and queue. No HTTP.',
                    evidence='MEASURED_HEAVY_CORRECTNESS' if cfg.get('state_probe') or cfg.get('verify_transfer') else 'MEASURED_PROFILE' if cfg.get('nsys') else 'MEASURED')
                append(folder/'requests.jsonl',row);out[i]=row
                print('DONE',cfg['name'],i,count,round(row['e2e_ms'],2),flush=True)
            now=time.monotonic()
            if not warmup and now-observation>=.5:
                observation=now
                phases=collections.Counter(p.phase for p in getattr(engine,'transfers',{}).values())
                append(folder/'queue_observations.jsonl',dict(time=now,delivered=index,completed=len(out),outstanding=len(active),
                    pending_phases=dict(phases),worker_credits={name:dict(role=p.role,slots=p.slots,reserved_blocks=p.reserved_blocks,
                    total_blocks=p.client.info['num_blocks']) for name,p in getattr(engine,'peers',{}).items()}))
            assert now-start<cfg['timeout'],'Serving watchdog'
            if not active and index<n:time.sleep(min(.001,max(0.,start+offsets[index]-now)))
    finally:
        if cfg.get('nsys') and not warmup:
            import torch
            torch.cuda.nvtx.range_pop()
    if not warmup:write(folder/'window.json',dict(start=start,end=time.monotonic(),last_arrival=max(arrivals.values()),
        arrival_pattern='SERIAL_CALIBRATION' if serial else 'OPEN_LOOP_POISSON'))
    return out


def main():
    from .runtime import install
    install();cfg=config();folder=ROOT/'raw'/cfg['name']
    source=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    assert len(source)>=cfg['count'] and len(cfg['request_caps'])==cfg['count']
    assert all(len(source[i]['input_ids'][cfg['model']])+cfg['request_caps'][i]<=cfg['context_limit'] for i in range(cfg['count']))
    from ..adaptive_loop_token_pd_2gpu import worker as inherited
    inherited.ROOT=ROOT;inherited.serve=serve
    sys.argv=['worker',os.environ['ARC_CONFIG']]
    inherited.main()


if __name__=='__main__':main()
