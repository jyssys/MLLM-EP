"""Four GPU-affine processes. Metadata-only queues and persistent CUDA IPC banks."""
import argparse
import gc
import gzip
import time
import types
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_flashvep.ouro_request_pp.model_adapter import Part
from poc_flashvep.ouro_request_pp.contract import route
from poc_flashvep.ouro_request_pp.reference import errors
from .mp_graph import graph_stage
from .contract import *


def test_case():
    gold=torch.load(ROOT/'.tensor_cache/native_P128_O8.pt',weights_only=False)
    requests=[];keys=list(gold)
    for i in range(4):
        r=gold[keys[i%2]]
        requests.append(dict(id=f'request-{i}',input_ids=r['input_ids'],teacher_ids=r['ids'],native_id=keys[i%2]))
    return dict(index=100,R=4,C=4,P=128,O=8,requests=requests)


@torch.inference_mode()
def worker(rank,queues,initqueues,args,selected):
    mask();torch.cuda.set_device(rank);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rendezvous=ROOT/'control'/f'{args.label}.gloo'
    dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=rank,world_size=4)
    part=Part(rank,args.arm);part.precompute_rotary=True;part.graph_stage=types.MethodType(graph_stage,part)
    path=route(args.arm,4);owned=[v for v in path if v[0]==rank]
    stream=torch.cuda.Stream(device=rank);lastshape=None
    with torch.cuda.stream(stream):
        for c in selected:
            requests=c['requests'];O=c['O'];byid={r['id']:r for r in requests}
            shape=(tuple(sorted({len(r['input_ids']) for r in requests})),O)
            if shape!=lastshape:
                part.graphs.clear();part.caches.clear();part.rotary_constants.clear();gc.collect();torch.cuda.empty_cache();lastshape=shape
            warm=time.perf_counter()
            for P in sorted({len(r['input_ids']) for r in requests}):
                row=next(r for r in requests if len(r['input_ids'])==P)
                part.configure([row],O,True);part.reset()
                for step in range(O):
                    hidden=torch.zeros((1,P if step==0 else 1,2048),dtype=torch.bfloat16,device=rank)
                    position=0 if step==0 else P+step-1
                    for _,loop,first,stop in owned:part.stage(hidden,row['id'],step,position,loop,first,stop)
            part.configure(requests,O,True);part.reset()
            # One persistent bank per request and tensor kind. Only handles cross Python at setup.
            bank={};receive={}
            for r in requests:
                for kind,shape0,dtype in [('prefill',(1,len(r['input_ids']),2048),torch.bfloat16),
                                         ('decode',(1,1,2048),torch.bfloat16),('token',(1,1),torch.long)]:
                    bank[(r['id'],kind)]=torch.empty(shape0,device=rank,dtype=dtype)
                    receive[(r['id'],kind)]=torch.empty(shape0,device=rank,dtype=dtype)
            torch.cuda.synchronize(rank)
            initqueues[(rank+1)%4].put(bank)
            incoming=initqueues[rank].get(timeout=180)
            # This setup opens only IPC handles. Serving messages contain no tensor payload.
            inputs={r['id']:torch.tensor([r['input_ids']],dtype=torch.long) for r in requests}
            feedback={r['id']:torch.tensor(r['teacher_ids'],device=rank).view(-1,1,1) for r in requests}
            prep=time.perf_counter()-warm
            torch.cuda.reset_peak_memory_stats(rank);dist.barrier();torch.cuda.synchronize(rank)
            ngraphs=len(part.graphs);events=[];counters={};bytecounts={};queue_total=0.;host_total=0.
            outputs={r['id']:dict(id=r['id'],input_ids=r['input_ids'],ids=[],computed_ids=[],token_times_abs_s=[]) for r in requests}
            if args.nsys:torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('MEASURED_WORKLOAD')
            origin=time.perf_counter()
            if rank==0:
                for r in requests:queues[0].put(('work',r['id'],0,0,'input_H2D',origin))
            done=0
            while True:
                message=queues[rank].get(timeout=600)
                if message[0]=='stop':break
                if message[0]=='done':
                    done+=1
                    if done==len(requests):
                        for other in [1,2,3]:queues[other].put(('stop',))
                        break
                    continue
                _,rid,step,visit,kind,ready=message
                target,loop,first,stop=path[visit];assert target==rank
                position=0 if step==0 else len(byid[rid]['input_ids'])+step-1
                host_start=time.perf_counter();queue_total+=(host_start-ready)*1000
                torch.cuda.nvtx.range_push(f'MP_{args.arm}|{rid}|step{step}|R{loop+1}|L{first}-{stop}')
                if args.trace:
                    a,b,e,d=[torch.cuda.Event(enable_timing=True) for _ in range(4)];a.record(stream)
                if kind=='input_H2D':payload=inputs[rid].to(rank,non_blocking=True)
                else:
                    payload=receive[(rid,kind)];payload.copy_(incoming[(rid,kind)],non_blocking=True)
                counters[kind]=counters.get(kind,0)+1;bytecounts[kind]=bytecounts.get(kind,0)+payload.numel()*payload.element_size()
                if args.trace:b.record(stream)
                hidden=part.embedding(payload) if visit==0 else payload
                hidden=part.stage(hidden,rid,step,position,loop,first,stop)
                if args.trace:e.record(stream)
                terminal=visit==len(path)-1
                if terminal:
                    logits=part.head(hidden[:,-1:,:]);token=logits[:,-1].argmax(-1).view(1,1)
                    actual=int(token)
                    token_id=actual if args.correctness else byid[rid]['teacher_ids'][step]
                    bank[(rid,'token')].copy_(token if args.correctness else feedback[rid][step])
                    nextkind='token'
                else:
                    nextkind='prefill' if step==0 else 'decode'
                    bank[(rid,nextkind)].copy_(hidden)
                if args.trace:d.record(stream)
                else:d=torch.cuda.Event(enable_timing=False);d.record(stream)
                d.synchronize();published=time.perf_counter();host_total+=(published-host_start)*1000
                torch.cuda.nvtx.range_pop()
                if args.trace:
                    events.append(dict(rank=rank,id=rid,step=step,loop=loop+1,layers=[first,stop],q_tokens=hidden.shape[1],
                        ready_abs_s=ready,host_start_abs_s=host_start,publication_abs_s=published,
                        queue_ms=(host_start-ready)*1000,copy_ms=a.elapsed_time(b),compute_ms=b.elapsed_time(e),
                        head_export_ms=e.elapsed_time(d),transfer_kind=kind,bytes=payload.numel()*payload.element_size()))
                if terminal:
                    outputs[rid]['ids'].append(token_id);outputs[rid]['computed_ids'].append(actual);outputs[rid]['token_times_abs_s'].append(published)
                    if args.correctness and step==O-1:
                        part.snapshots[rid]=dict(logits=logits.float().cpu(),hidden=hidden[:,-1:,:].float().cpu())
                    if step+1==O:queues[0].put(('done',rid))
                    else:queues[0].put(('work',rid,step+1,0,'token',published))
                else:queues[path[visit+1][0]].put(('work',rid,step,visit+1,nextkind,published))
            end=time.perf_counter();torch.cuda.nvtx.range_pop()
            if args.nsys:torch.cuda.cudart().cudaProfilerStop()
            dist.barrier()
            cachemeta={rid:part.cache_summary(rid) for rid in byid}
            result=dict(arm='LPP4_mp' if args.arm=='layer_pp4' else 'LoopPP4_mp',rank=rank,physical_GPU=rank+4,
                case={k:v for k,v in c.items() if k!='requests'},repeat=args.repeat,start_abs_s=origin,end_abs_s=end,
                preparation_wall_s=prep,outputs=outputs if rank==3 else {},events=events,
                graph_captures_during_run=len(part.graphs)-ngraphs,transfer_counts=counters,transfer_bytes=bytecounts,
                ready_queue_total_ms=queue_total,host_worker_total_ms=host_total,
                memory=dict(weights=part.weight_bytes,KV_bytes=sum(v['bytes'] for v in cachemeta.values()),caches=cachemeta,
                    graph_scratch_bytes=sum(v[1].numel()*v[1].element_size()+v[0].numel()*v[0].element_size() for v in part.graphs.values()),
                    peer_export_bytes=sum(v.numel()*v.element_size() for v in bank.values()),
                    allocated_bytes=torch.cuda.memory_allocated(rank),reserved_bytes=torch.cuda.memory_reserved(rank),peak_bytes=torch.cuda.max_memory_allocated(rank)),
                transport='Persistent CUDA IPC buffers + CUDA peer copy. CPU queues carry metadata only during serving.',
                scheduler='Independent process FIFO per GPU. No central host scheduling at intermediate boundaries.',
                source_math='Pinned HF BF16 decoder, exact SDPA length, owner-local KV, immutable RoPE constants')
            if args.correctness:
                gold=torch.load(ROOT/'.tensor_cache/native_P128_O8.pt',weights_only=False);checks=[]
                for r in requests:
                    ref=gold[r['native_id']];cache=part.snapshot(r['id'])
                    exact=all(torch.equal(x,y) for s,kv in cache.items() for x,y in zip(kv,ref['cache'][s]))
                    check=dict(rank=rank,id=r['id'],native_id=r['native_id'],KV_slots=len(cache),KV_bitwise=exact)
                    if rank==3:
                        check.update(token_exact=outputs[r['id']]['ids']==ref['ids'],
                            logits=errors(part.snapshots[r['id']]['logits'],ref['logits'][-1].view(1,1,-1)),
                            hidden=errors(part.snapshots[r['id']]['hidden'],ref['hidden'][-1]))
                    check['PASS']=exact and (rank!=3 or (check['token_exact'] and check['logits']['bitwise'] and check['hidden']['bitwise']))
                    checks.append(check)
                write(ROOT/'correctness'/f'{args.label}_rank{rank}.json',checks);assert all(v['PASS'] for v in checks),checks
            target=ROOT/'raw'/args.label;target.mkdir(parents=True,exist_ok=True)
            with gzip.open(target/f'case_{c["index"]:02d}_rank{rank}.json.gz','wt') as f:json.dump(result,f,separators=(',',':'))
            if rank==0:print('MP_MEASURED',args.arm,c['index'],c['C'],c['P'],O,round(end-origin,4),flush=True)
            # Release imported views before any producer releases its backing storage.
            del incoming;gc.collect();dist.barrier();del bank,receive;gc.collect();dist.barrier()
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['layer_pp4','loop_pp4'],required=True)
    parser.add_argument('--label',required=True);parser.add_argument('--indices',default='0,1');parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--trace',action='store_true');parser.add_argument('--nsys',action='store_true');parser.add_argument('--correctness',action='store_true')
    args=parser.parse_args();mask();(ROOT/'control').mkdir(exist_ok=True)
    indices=set(map(int,args.indices.split(',')))
    selected=[test_case()] if args.correctness else [c for c in cases() if c['index'] in indices]
    ctx=mp.get_context('spawn');queues=[ctx.Queue() for _ in range(4)];initqueues=[ctx.Queue() for _ in range(4)]
    mp.spawn(worker,args=(queues,initqueues,args,selected),nprocs=4,join=True)


if __name__=='__main__':main()
