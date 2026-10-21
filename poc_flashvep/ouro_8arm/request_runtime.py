"""Whole-request asynchronous PP. Dynamic exits include charged last-exited KV copies."""
import argparse
import collections
import os
import subprocess
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_flashvep.ouro_request_pp.model_adapter import tokenizer
from .stage import Stage,decision
from .wave import NoRefillWave
from .contract import *

def path_for(arm):
    p=ARMS[arm]['parallelism']
    return [(rank,loop,rank*6,(rank+1)*6) for loop in range(4) for rank in range(4)] if p=='layer_pp4' else [(loop if p=='loop_pp4' else 0,loop,0,24) for loop in range(4)]

@torch.inference_mode()
def worker(rank,queues,init,acks,args,groups):
    n=args.n;torch.cuda.set_device(rank);torch.set_num_threads(1)
    if n>1:dist.init_process_group('gloo',init_method='file://'+str(ROOT/'control'/f'{args.label}.gloo'),rank=rank,world_size=n)
    arm=args.arm;placement=ARMS[arm]['parallelism'];dynamic=ARMS[arm]['mode']=='dynamic'
    part=Stage(rank,'layer_pp4' if placement=='layer_pp4' else 'loop_pp4');tok=tokenizer();route=path_for(arm)
    owned=[(loop,first,last) for target,loop,first,last in route if target==rank]
    for batch_index,requests in enumerate(groups):
        batch_index=requests[0].get('original_cohort',batch_index)
        for row in requests:
            if 'input_ids' not in row:row['input_ids']=tok.encode(row['prompt'],add_special_tokens=False)
        prep=time.perf_counter();torch.cuda.reset_peak_memory_stats(rank);part.local_graphs.clear();part.prepare(requests,args.cap,owned)
        bank={};receive={}
        for row in requests:
            rid=row['id']
            for kind,shape,dtype in [('prefill',(1,len(row['input_ids']),2048),torch.bfloat16),('decode',(1,1,2048),torch.bfloat16),
                    ('token',(1,1),torch.long),('exit',(len(part.indices),2,1,16,128),torch.bfloat16)]:
                bank[rid,kind]=torch.empty(shape,device=rank,dtype=dtype);receive[rid,kind]=torch.empty_like(bank[rid,kind])
        torch.cuda.synchronize(rank)
        for target in range(n):
            if target!=rank:init[target].put((rank,bank))
        # Local storage does not need CUDA IPC reconstruction in its own process.
        peers={rank:bank}
        peers.update(dict(init[rank].get(timeout=300) for _ in range(n-1)))
        inputs={r['id']:torch.tensor([r['input_ids']],device=rank) for r in requests}
        byid={r['id']:r for r in requests};outputs={r['id']:dict(ids=[],predictions=[],depths=[],times=[],gates=[]) for r in requests}
        prepared=time.perf_counter()-prep
        if n>1:dist.barrier()
        if rank==0:
            write(ROOT/'raw'/args.label/f'GPU_setup_{batch_index:03d}.json',dict(
                GPUs=ARMS[arm]['GPUs'],measurement='NVML snapshot after setup, before serving. Includes CUDA contexts and reserved pools.',
                GPUs_csv=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,ARMS[arm]['GPUs'])),
                    '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True)))
        if n>1:dist.barrier()
        torch.cuda.synchronize(rank);preparation_peak=torch.cuda.max_memory_allocated(rank);torch.cuda.reset_peak_memory_stats(rank)
        origin=time.perf_counter();done=0;events=[];sample_events=[];samples=[];counts=collections.Counter();bytes_=collections.Counter()
        no_refill=dynamic or args.replay_case is not None or args.lockstep
        wave=NoRefillWave([r['id'] for r in requests])
        if args.nsys:torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push('EIGHT_COHORT')
        if rank==0:
            for r in requests:queues[0].put(('work',r['id'],0,0,-1,'input',0.,1.,origin))
        while True:
            m=queues[rank].get(timeout=1200)
            if m[0]=='stop':break
            if m[0]=='done':
                done+=1
                if done==len(requests):
                    for other in range(1,n):queues[other].put(('stop',))
                    break
                continue
            if m[0]=='sample':
                _,rid,step,loop,source,actual,predicted,gpu_finish=m
                assert rank==0
                sample_begin=time.perf_counter()
                if args.trace or args.nsys:torch.cuda.nvtx.range_push(f'EIGHT_CONTROLLER|{rid}|step{step}')
                finish=time.perf_counter();row=byid[rid];out=outputs[rid]
                out['ids'].append(actual);out['predictions'].append(predicted);out['depths'].append(loop+1);out['times'].append(finish-origin)
                replay=args.replay_case is not None
                text='' if replay else tok.decode(out['ids'],skip_special_tokens=False)
                ended=step+1>=args.cap or (not replay and (actual==tok.eos_token_id or any(text.endswith(s) for s in row['stop_strings'])))
                if ended:
                    successor=None
                    record=dict(id=rid,dataset=row['dataset'],arm=arm,engine='Custom MP request FIFO / full-capacity causal-mask graphs',
                        GPUs=ARMS[arm]['GPUs'],prompt_sha256=row['prompt_sha256'],prompt_ids=row['input_ids'],output_ids=out['ids'],predicted_ids=out['predictions'],text=text,
                        finish_reason='eos' if actual==tok.eos_token_id else 'length' if step+1>=args.cap else 'stop',
                        output_tokens=len(out['ids']),depths=out['depths'],mean_depth=sum(out['depths'])/len(out['depths']),
                        token_publications=out['times'],E2E_s=finish-origin,TTFT_s=out['times'][0],
                        max_new_tokens=args.cap,evaluation_eligible=args.cap==4096 and not args.probe and not args.forced_depths and not replay,
                        token_successor_policy='NO_REFILL_COHORT_WAVES' if no_refill else 'ASYNCHRONOUS_REQUEST_FIFO',
                        TPOT_s=(out['times'][-1]-out['times'][0])/(len(out['ids'])-1) if len(out['ids'])>1 else None,
                        lane='FROZEN_TOKEN_HARDWARE_REPLAY' if replay else 'FORCED_EXIT_DIAGNOSTIC' if args.forced_depths else 'SYNTHETIC_PROBE' if args.probe else 'ACTUAL_GREEDY_TASK_GENERATION',
                        status='EXPLORATORY_BACKEND_QUALIFICATION_PENDING',cohort_index=batch_index)
                    filename=rid.replace('/','_')+(f'_repeat{batch_index}' if replay else '')+'.json'
                    if replay:samples.append((filename,record))
                    else:write(ROOT/'raw'/args.label/filename,record)
                    if not replay:print('QUALITY',arm,rid,len(out['ids']),record['finish_reason'],round(record['mean_depth'],3),flush=True)
                    queues[0].put(('done',rid))
                else:
                    successor=('work',rid,step+1,0,source,'token',0.,1.,finish)
                    if not no_refill:queues[0].put(successor)
                if no_refill:
                    for ready_work in wave.complete(rid,successor):queues[0].put(ready_work)
                if args.trace or args.nsys:
                    torch.cuda.nvtx.range_pop()
                    sample_events.append(dict(id=rid,step=step,start_s=sample_begin-origin,end_s=time.perf_counter()-origin,
                        scope='Host controller: detokenization, stop checks, bookkeeping, metadata, and task-record I/O.'))
                continue
            if m[0]=='fill_local':
                _,rid,loop,position,source=m
                part.fill_local(rid,loop,position);torch.cuda.synchronize(rank);acks[source].put((rid,rank))
                counts['local_exit_fills']+=1
                continue
            if m[0]=='fill':
                _,rid,source,loop,position,targets=m
                incoming=receive[rid,'exit'];incoming.copy_(peers[source][rid,'exit'],non_blocking=True)
                part.import_exit(rid,loop,targets,position,incoming);torch.cuda.synchronize(rank)
                counts['exit_KV_copies']+=1;bytes_['exit_KV']+=incoming.numel()*incoming.element_size();acks[source].put((rid,rank));continue
            _,rid,step,visit,source,kind,cdf,survival,ready=m
            _,loop,first,last=route[visit];row=byid[rid];position=0 if step==0 else len(row['input_ids'])+step-1
            torch.cuda.nvtx.range_push(f'EIGHT|{arm}|{rid}|step{step}|R{loop+1}|L{first}-{last}')
            begin=time.perf_counter()
            if args.trace:
                a,b,c,d=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                a.record()
            if kind=='input':payload=inputs[rid]
            else:
                payload=receive[rid,kind];payload.copy_(peers[source][rid,kind],non_blocking=True)
                transfer=('peer_' if source!=rank else 'local_')+kind
                counts[transfer]+=1;bytes_[transfer]+=payload.numel()*payload.element_size()
            if args.trace:b.record()
            hidden=part.embedding(payload) if visit==0 else payload
            h,gate=part.execute(hidden,rid,step,position,loop,first,last)
            counts['physical_layer_invocations']+=last-first
            counts['token_layer_work']+=(last-first)*hidden.shape[1]
            counts[f'R{loop+1}_'+('prefill_visits' if step==0 else 'decode_visits')]+=1
            if args.trace:c.record()
            terminal=visit==len(route)-1
            if dynamic and step>0 and last==24:
                exiting,cdf,survival,hazard=decision(gate,loop,cdf,survival);terminal=exiting
                if args.forced_depths:terminal=loop+1==args.forced_depths[(step-1+row.get('forced_phase',0))%len(args.forced_depths)]
                if 'teacher_depths' in row:terminal=loop+1==row['teacher_depths'][step]
                outputs[rid]['gates'].append(dict(step=step,loop=loop+1,hazard=hazard,cdf=cdf))
            if terminal:
                if dynamic and step>0 and loop<3:
                    if placement=='layer_pp4':
                        part.fill_local(rid,loop,position);targets=[k for k in range(n) if k!=rank]
                        for target in targets:queues[target].put(('fill_local',rid,loop,position,rank))
                        # Layer-local filling is handled below through the same ordered worker protocol.
                    elif placement=='single':part.fill_local(rid,loop,position);targets=[]
                    else:
                        part.export_exit(rid,loop,position,bank[rid,'exit']);torch.cuda.synchronize(rank)
                        targets=list(range(loop+1,4))
                        for target in targets:queues[target].put(('fill',rid,rank,loop,position,[target]))
                    for _ in targets:
                        acknowledged=acks[rank].get(timeout=1200);assert acknowledged[0]==rid,acknowledged
                z=part.head(h[:,-1:,:]);token=z[:,-1].argmax(-1).view(1,1)
                if args.replay_case is None:assert torch.isfinite(z).all()
                actual=int(token);predicted=actual
                if args.replay_case is not None:
                    actual=row['teacher_ids'][step];token.fill_(actual)
                bank[rid,'token'].copy_(token);torch.cuda.synchronize(rank);finish=time.perf_counter()
                queues[0].put(('sample',rid,step,loop,rank,actual,predicted,finish))
            else:
                nextkind='prefill' if step==0 else 'decode';bank[rid,nextkind].copy_(h);torch.cuda.synchronize(rank)
                queues[route[visit+1][0]].put(('work',rid,step,visit+1,rank,nextkind,cdf,survival,time.perf_counter()))
            torch.cuda.nvtx.range_pop()
            if args.trace:
                d.record();d.synchronize()
                events.append(dict(rank=rank,id=rid,step=step,loop=loop+1,layers=[first,last],ready_s=ready-origin,
                    dispatch_s=begin-origin,published_s=time.perf_counter()-origin,copy_ms=a.elapsed_time(b),compute_ms=b.elapsed_time(c),
                    head_export_ms=c.elapsed_time(d),scope='Rank-local CUDA service intervals, not additive E2E shares.'))
        end=time.perf_counter()
        torch.cuda.nvtx.range_pop()
        if args.nsys:torch.cuda.cudart().cudaProfilerStop()
        if n>1:dist.barrier()
        for filename,record in samples:
            record['text']=tok.decode(record['output_ids'],skip_special_tokens=False)
            write(ROOT/'raw'/args.label/filename,record)
        if samples:print('REPLAY',arm,args.replay_case,batch_index,round(end-origin,5),flush=True)
        mem=dict(weights=part.weight_bytes,KV_bytes=sum(part.cache_summary(r['id'])['bytes'] for r in requests),
            allocated=torch.cuda.memory_allocated(rank),reserved=torch.cuda.memory_reserved(rank),peak=torch.cuda.max_memory_allocated(rank),
            preparation_peak_allocated=preparation_peak)
        write(ROOT/'raw'/args.label/f'cohort_{batch_index:03d}_rank{rank}.json',dict(rank=rank,origin=origin,end=end,preparation_s=prepared,
            preparation_begin=prep,admission_to_service_s=origin-prep,resident_model_cohort_wall_s=end-prep,
            counts=counts,bytes=bytes_,memory=mem,events=events,controller_events=sample_events,gates={rid:v['gates'] for rid,v in outputs.items()},
            ownership=dict(physical_layers=part.indices,loop_stage_tuples=owned,parameter_devices=sorted({str(p.device) for p in part.layers.parameters()}),
                UUID=str(torch.cuda.get_device_properties(rank).uuid),mask=os.environ['CUDA_VISIBLE_DEVICES']),
            graph_count=len(part.local_graphs),replay_case=args.replay_case,repeat=batch_index,
            lane='TRACE' if args.trace or args.nsys else 'CLEAN_REPLAY' if args.replay_case is not None else 'TASK_OBSERVER',
            scope='Graph construction precedes admission. Warm-up repeat zero is excluded from replay headline.'))
        if args.probe:
            torch.save({r['id']:part.snapshot(r['id']) for r in requests},ROOT/'.tensor_cache'/f'{args.label}_rank{rank}.pt')
        del peers
        if n>1:dist.barrier()
    if n>1:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['S4','SD','L4','LD','O4','OD'],required=True)
    p.add_argument('--label',required=True);p.add_argument('--cap',type=int,default=4096);p.add_argument('--limit',type=int)
    p.add_argument('--trace',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--cohort',type=int,default=4)
    p.add_argument('--probe',action='store_true');p.add_argument('--forced-depths',type=lambda s:list(map(int,s.split(','))))
    p.add_argument('--replay-case',type=int);p.add_argument('--repeats',type=int,default=5);p.add_argument('--nsys',action='store_true')
    p.add_argument('--resume-from')
    p.add_argument('--depth-replay')
    p.add_argument('--lockstep',action='store_true')
    p.add_argument('--ids',type=lambda s:s.split(','))
    p.add_argument('--extension400',action='store_true')
    args=p.parse_args();args.n=len(ARMS[args.arm]['GPUs']);rows=dataset(not args.full)
    if args.extension400:rows=subset400(extension_only=True)
    if args.ids:
        selected=set(args.ids);rows=[r for r in rows if r['id'] in selected];assert {r['id'] for r in rows}==selected
    if args.probe:
        tok=tokenizer();rows=[]
        for i in range(4):
            text='Numerical calibration: '+('add integers 1 2 3. ' if i%2==0 else 'write a short Python function. ')
            ids=tok.encode(text,add_special_tokens=False);ids=(ids*((128+len(ids)-1)//len(ids)))[:128]
            rows.append(dict(id=f'probe-{i}',dataset='calibration',prompt=text,input_ids=ids,prompt_sha256=sha(ROOT/'spec_frozen.md'),stop_strings=[],forced_phase=i))
    if args.limit:rows=rows[:args.limit]
    if args.depth_replay:
        replay=read(args.depth_replay);rows=replay['requests'];args.cap=replay['O'];args.replay_case=replay['name']
        groups=[rows for _ in range(args.repeats+1)]
    elif args.replay_case is not None:
        from poc_flashvep.ouro_tp4.contract import cases
        case=next(c for c in cases() if c['index']==args.replay_case);args.cap=case['O']
        rows=[dict(r,dataset='frozen_hardware',prompt='',prompt_sha256=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest(),stop_strings=[]) for r in case['requests']]
        groups=[rows for _ in range(args.repeats+1)]
    else:groups=[rows[i:i+args.cohort] for i in range(0,len(rows),args.cohort)]
    if not args.probe and args.replay_case is None:
        assert (ROOT/'frozen_prompt_ids.json').exists(),'Freeze cross-version prompt IDs before task evaluation.'
        for i,group in enumerate(groups):
            for row in group:row['original_cohort']=row.get('cohort400',i)
    if args.resume_from:
        kept=[];reused=[]
        for group in groups:
            i=group[0]['original_cohort'];source=ROOT/'raw'/args.resume_from
            comparable=(source/f'cohort_{i:03d}_rank0.json').exists()
            for row in group:
                path=source/(row['id'].replace('/','_')+'.json')
                if not path.exists():comparable=False;continue
                old=read(path)
                comparable=comparable and old['prompt_ids']==row['input_ids'] and old['arm']==args.arm
            if comparable:reused.append(i)
            else:kept.append(group)
        groups=kept
        write(ROOT/'state'/(args.label+'_resume.json'),dict(source=args.resume_from,reused_complete_cohorts=reused,
            executed_cohorts=[g[0]['original_cohort'] for g in kept],rule='Reuse only complete C4 cohorts whose prompt IDs match the frozen canonical input.'))
    (ROOT/'raw'/args.label).mkdir(exist_ok=False);(ROOT/'control').mkdir(exist_ok=True)
    ctx=mp.get_context('spawn');queues=[ctx.Queue() for _ in range(args.n)];init=[ctx.Queue() for _ in range(args.n)];acks=[ctx.Queue() for _ in range(args.n)]
    mp.spawn(worker,args=(queues,init,acks,args,groups),nprocs=args.n,join=True)
    write(ROOT/'raw'/args.label/'complete.json',dict(success=True,arm=args.arm,requests=len(rows),
        requests_executed=sum(len(g) for g in groups),resume_from=args.resume_from))

if __name__=='__main__':main()
