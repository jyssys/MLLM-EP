"""Matched BF16 TP4 diagnostic runtime with actual token-level CDB exits."""
import argparse
import os
import time
import torch
import torch.distributed as dist
from poc_flashvep.ouro_tp4.tp_model import TPModel
from poc_flashvep.ouro_request_pp.model_adapter import tokenizer
from .stage import decision
from .contract import *

class View:
    def __init__(self,storage,pos,prefill):self.storage=storage;self.pos=pos;self.prefill=prefill
    def update(self,k,v,index,metadata):
        self.storage[index,0].index_copy_(2,self.pos,k);self.storage[index,1].index_copy_(2,self.pos,v)
        return (k,v) if self.prefill else (self.storage[index,0],self.storage[index,1])

class Engine(TPModel):
    def prepare(self,requests,cap):
        self.pool={r['id']:torch.zeros((96,2,1,4,len(r['input_ids'])+cap-1,128),device=self.device,dtype=torch.bfloat16) for r in requests}
        self.tapes={}
        for r in requests:
            rid=r['id'];P=len(r['input_ids']);h=torch.zeros((1,P,2048),device=self.device,dtype=torch.bfloat16)
            for loop in range(4):
                self.block(h,rid,loop,0,0)
                self.block(h[:,:1],rid,loop,1,P)
        torch.cuda.synchronize(self.device)

    def block(self,h,rid,loop,step,position):
        storage=self.pool[rid]
        def run(inp,pos,mask,prefill):
            rotary=self.rotary(inp,pos.unsqueeze(0));view=View(storage,pos,prefill)
            for layer in self.layers:
                inp=layer(inp,attention_mask=mask,position_ids=pos.unsqueeze(0),past_key_value=view,
                    cache_position=pos,position_embeddings=rotary,current_ut=loop,use_cache=True)
            inp=self.norm(inp);return inp,self.gate(inp[:,-1:,:])
        if step==0:return run(h,torch.arange(h.shape[1],device=self.device),None,True)
        key=(rid,loop)
        if key not in self.tapes:
            inp=torch.empty_like(h);inp.copy_(h);pos=torch.tensor([position],device=self.device)
            full=torch.arange(storage.shape[-2],device=self.device)
            def call():
                mask=torch.where(full<=pos[0],0.,float('-inf')).to(torch.bfloat16).view(1,1,1,-1)
                return run(inp,pos,mask,False)
            call();torch.cuda.synchronize(self.device);dist.barrier()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out,gate=call()
            self.tapes[key]=inp,pos,out,gate,graph,full
        inp,pos,out,gate,graph,full=self.tapes[key];inp.copy_(h);pos.fill_(position);graph.replay()
        return out,gate

    def forward_token(self,ids,rid,step,P,dynamic,check_finite=True,forced_depth=None,validate_KV=False):
        h=self.embedding(ids);cdf=0.;survival=1.;gates=[];position=0 if step==0 else P+step-1
        for loop in range(4):
            h,gate=self.block(h,rid,loop,step,position)
            if dynamic and step>0:
                end,cdf,survival,hazard=decision(gate,loop,cdf,survival)
                if forced_depth is not None:end=loop+1==forced_depth
                control=torch.tensor([int(end)],device=self.device);dist.broadcast(control,src=0)
                gates.append(dict(loop=loop+1,hazard=hazard,cdf=cdf))
                if bool(control):
                    trace_repair=getattr(self,'trace_ranges',False) and loop<3
                    if trace_repair:torch.cuda.nvtx.range_push(f'EIGHT_KV_REPAIR|TD|rank{self.rank}|{rid}|R{loop+1}|pos{position}')
                    for later in range(loop+1,4):
                        self.pool[rid][24*later:24*(later+1),:,:,: ,position:position+1].copy_(self.pool[rid][24*loop:24*(loop+1),:,:,: ,position:position+1])
                        if validate_KV:
                            assert torch.equal(self.pool[rid][24*later:24*(later+1),:,:,:,position:position+1],
                                self.pool[rid][24*loop:24*(loop+1),:,:,:,position:position+1])
                    if trace_repair:torch.cuda.nvtx.range_pop()
                    break
        z=self.head(h[:,-1:,:])
        if check_finite:assert torch.isfinite(z).all()
        token=z[:,-1].argmax(-1).view(1,1);dist.broadcast(token,src=0)
        return token,loop+1,gates

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--dynamic',action='store_true');p.add_argument('--label',required=True)
    p.add_argument('--cap',type=int,default=4096);p.add_argument('--limit',type=int);p.add_argument('--probe',action='store_true')
    p.add_argument('--replay-case',type=int);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--nsys',action='store_true');p.add_argument('--full',action='store_true')
    p.add_argument('--depth-replay')
    p.add_argument('--ids',type=lambda s:s.split(','))
    p.add_argument('--extension400',action='store_true')
    p.add_argument('--serving-KV-shape',action='store_true',help='Bounded O32 trace with the unchanged O4096 serving KV allocation.')
    p.add_argument('--forced-depths',type=lambda s:list(map(int,s.split(','))))
    args=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    if args.serving_KV_shape:
        assert args.cap==32 and args.limit==4 and args.replay_case is None and not args.depth_replay and not args.probe and not args.extension400
    assert not args.forced_depths or (args.dynamic and args.probe and all(v in [2,3,4] for v in args.forced_depths))
    assert os.environ['WORLD_SIZE']=='4' and os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank));model=Engine(rank);tok=tokenizer();arm='TD' if args.dynamic else 'T4'
    model.trace_ranges=args.nsys
    rows=dataset(not args.full)
    if args.extension400:rows=subset400(extension_only=True)
    if args.ids:
        selected=set(args.ids);rows=[r for r in rows if r['id'] in selected];assert {r['id'] for r in rows}==selected
    if args.probe:
        rows=[]
        for i in range(4):
            text='Numerical calibration: '+('add integers 1 2 3. ' if i%2==0 else 'write a short Python function. ')
            ids=tok.encode(text,add_special_tokens=False);ids=(ids*((128+len(ids)-1)//len(ids)))[:128]
            rows.append(dict(id=f'probe-{i}',dataset='calibration',prompt=text,input_ids=ids,prompt_sha256=sha(ROOT/'spec_frozen.md'),stop_strings=[],forced_phase=i))
    if args.limit:rows=rows[:args.limit]
    path=ROOT/'raw'/args.label
    if rank==0:path.mkdir(exist_ok=False)
    dist.barrier()
    if args.depth_replay:
        replay=read(args.depth_replay);args.cap=replay['O'];args.replay_case=replay['name'];group=replay['requests']
        groups=[group for _ in range(args.repeats+1)]
    elif args.replay_case is not None:
        from poc_flashvep.ouro_tp4.contract import cases
        case=next(c for c in cases() if c['index']==args.replay_case);args.cap=case['O']
        group=[dict(r,dataset='frozen_hardware',prompt='',prompt_sha256=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest(),stop_strings=[]) for r in case['requests']]
        groups=[group for _ in range(args.repeats+1)]
    else:groups=[[dict(r,input_ids=r.get('input_ids') or tok.encode(r['prompt'],add_special_tokens=False)) for r in rows[offset:offset+4]] for offset in range(0,len(rows),4)]
    for cohort_index,group in enumerate(groups):
        cohort_index=group[0].get('cohort400',cohort_index)
        preparation_begin=time.perf_counter()
        allocation=KV_allocation_cap(args.cap,args.serving_KV_shape)
        assert all(len(r['input_ids'])+allocation<=8192 for r in group)
        torch.cuda.reset_peak_memory_stats(rank);model.prepare(group,allocation);dist.barrier();torch.cuda.synchronize(rank)
        preparation_peak=torch.cuda.max_memory_allocated(rank);torch.cuda.reset_peak_memory_stats(rank)
        prompt_inputs={r['id']:torch.tensor([r['input_ids']],device=rank) for r in group}
        outputs={r['id']:dict(ids=[],predictions=[],depths=[],times=[],gates=[]) for r in group};live=list(group);samples=[];origin=time.perf_counter()
        if args.nsys:torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push('EIGHT_COHORT')
        for step in range(args.cap):
            next_live=[]
            for r in live:
                out=outputs[r['id']];ids=prompt_inputs[r['id']] if step==0 else out['GPU_token']
                forced=r['teacher_depths'][step] if 'teacher_depths' in r else (
                    args.forced_depths[(step-1+r.get('forced_phase',0))%len(args.forced_depths)] if args.forced_depths and step>0 else None)
                token,depth,gates=model.forward_token(ids,r['id'],step,len(r['input_ids']),args.dynamic,check_finite=args.replay_case is None,
                    forced_depth=forced,validate_KV=args.probe);value=int(token)
                out['predictions'].append(value)
                if args.replay_case is not None:value=r['teacher_ids'][step];token.fill_(value)
                out['GPU_token']=token
                out['ids'].append(value);out['depths'].append(depth);out['times'].append(time.perf_counter()-origin);out['gates'].append(gates)
                text='' if args.replay_case is not None else tok.decode(out['ids'],skip_special_tokens=False)
                ended=step+1==args.cap or (args.replay_case is None and (value==tok.eos_token_id or any(text.endswith(s) for s in r['stop_strings'])))
                if ended:
                    if rank==0:
                        finish='eos' if value==tok.eos_token_id else 'length' if step+1==args.cap else 'stop'
                        record=dict(id=r['id'],dataset=r['dataset'],arm=arm,engine='Custom TP4 / four-head full-capacity graph',GPUs=[4,5,6,7],
                            prompt_sha256=r['prompt_sha256'],prompt_ids=r['input_ids'],output_ids=out['ids'],predicted_ids=out['predictions'],text=text,
                            finish_reason=finish,output_tokens=len(out['ids']),depths=out['depths'],mean_depth=sum(out['depths'])/len(out['depths']),
                            max_new_tokens=args.cap,KV_allocation_output_cap=allocation,
                            evaluation_eligible=args.cap==4096 and not args.probe and not args.forced_depths and args.replay_case is None,
                            token_successor_policy='NO_REFILL_COHORT_WAVES',
                            E2E_s=out['times'][-1],TTFT_s=out['times'][0],TPOT_s=(out['times'][-1]-out['times'][0])/(len(out['ids'])-1) if len(out['ids'])>1 else None,
                            gates=out['gates'],lane='FROZEN_TOKEN_HARDWARE_REPLAY' if args.replay_case is not None else 'FORCED_EXIT_DIAGNOSTIC' if args.forced_depths else 'SYNTHETIC_PROBE' if args.probe else 'ACTUAL_GREEDY_TASK_GENERATION',status='NUMERICALLY_UNQUALIFIED_EXPLORATORY',cohort_index=cohort_index)
                        filename=r['id'].replace('/','_')+(f'_repeat{cohort_index}' if args.replay_case is not None else '')+'.json'
                        if args.replay_case is not None:samples.append((filename,record))
                        else:write(path/filename,record)
                        if args.replay_case is None:print('QUALITY',arm,r['id'],len(out['ids']),round(record['mean_depth'],3),flush=True)
                else:next_live.append(r)
            live=next_live
            if not live:break
        torch.cuda.nvtx.range_pop()
        if args.nsys:torch.cuda.cudart().cudaProfilerStop()
        service_end=time.perf_counter();elapsed=service_end-origin
        for filename,record in samples:
            record['text']=tok.decode(record['output_ids'],skip_special_tokens=False)
            write(path/filename,record)
        if samples:print('REPLAY',arm,args.replay_case,cohort_index,round(elapsed,5),flush=True)
        write(path/f'cohort_{cohort_index:03d}_rank{rank}.json',dict(rank=rank,makespan_s=elapsed,weights=model.weights,
            origin=origin,preparation_begin=preparation_begin,admission_to_service_s=origin-preparation_begin,
            resident_model_cohort_wall_s=service_end-preparation_begin,
            ownership=dict(UUID=str(torch.cuda.get_device_properties(rank).uuid),mask=os.environ['CUDA_VISIBLE_DEVICES'],
                local_Q_heads=4,local_KV_heads=4,MLP_intermediate_width=1408,physical_layers=24,logical_KV_slots=96),
            reserved_KV_bytes=sum(v.numel()*v.element_size() for v in model.pool.values()),peak=torch.cuda.max_memory_allocated(rank),
            preparation_peak_allocated=preparation_peak,
            forward_loop_invocations=sum(sum(v['depths']) for v in outputs.values()),
            token_loop_work=4*sum(len(r['input_ids']) for r in group)+sum(sum(v['depths'][1:]) for v in outputs.values()),graph_tapes=len(model.tapes),
            KV_allocation_output_cap=allocation,
            replay_case=args.replay_case,repeat=cohort_index,lane='TRACE' if args.nsys else 'CLEAN_REPLAY' if args.replay_case is not None else 'TASK_OBSERVER'))
        if args.probe:
            snapshot={r['id']:{slot:(value[slot,0,:,:,:len(r['input_ids'])+len(outputs[r['id']]['ids'])-1].cpu(),
                value[slot,1,:,:,:len(r['input_ids'])+len(outputs[r['id']]['ids'])-1].cpu()) for slot in range(96)}
                for r in group for value in [model.pool[r['id']]]}
            torch.save(snapshot,ROOT/'.tensor_cache'/f'{args.label}_rank{rank}.pt')
    # Destroy graph-owned NCCL references while the communicator still exists.
    torch.cuda.synchronize(rank);model.tapes.clear();dist.barrier()
    dist.destroy_process_group()
    if rank==0:write(path/'complete.json',dict(success=True,requests=len(rows),arm=arm))

if __name__=='__main__':main()
