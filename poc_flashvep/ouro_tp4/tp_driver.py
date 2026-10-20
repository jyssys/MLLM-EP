"""Four-process NCCL TP4 correctness and batch-one round-robin serving."""
import argparse
import datetime
import gc
import gzip
import time
import torch
import torch.distributed as dist
from poc_flashvep.ouro_request_pp.reference import errors
from .contract import *
from .tp_model import TPModel


def require_qualification():
    """Refuse serving measurements when this implementation lacks full C2 evidence."""
    for rank in range(4):
        path=ROOT/'correctness'/f'tp_fp32columns_full_rank{rank}.json'
        records=json.loads(path.read_text())
        aligned=[r for r in records if r.get('teacher_forced')]
        assert {r['P'] for r in aligned}=={128,512,2048},'C2 is incomplete. Do not run the TP serving matrix.'
        assert all(r['PASS'] for r in aligned),'C2 failed. Do not run the TP serving matrix.'
        natural=[r for r in records if r.get('lane')=='free_running_greedy']
        assert {r['P'] for r in natural}=={128,512,2048} and all(r['token_exact'] for r in natural)


def initialize():
    mask();rank=int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==4
    torch.cuda.set_device(rank);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(minutes=10))
    x=torch.ones(1,device=f'cuda:{rank}');dist.all_reduce(x);assert int(x)==4
    return rank


@torch.inference_mode()
def operators(model,label):
    records=torch.load(ROOT/'.tensor_cache/operators.pt',weights_only=False)
    result=[];a_cache=model.remote.UniversalTransformerCache(96);b_cache=model.remote.UniversalTransformerCache(96)
    layer=model.layers[0]
    for r in records:
        step,loop=r['step'],r['loop'];device=model.device
        pos=torch.arange(16,device=device) if step==0 else torch.tensor([16],device=device)
        rotary=tuple(v.to(device) for v in r['rotary'])
        h=r['hidden_in'].to(device);attn_in=r['attention_in'].to(device)
        attn,_=layer.self_attn(attn_in,position_embeddings=rotary,attention_mask=None,past_key_value=a_cache,
                             cache_position=pos,current_ut=loop)
        mlp=layer.mlp(r['mlp_in'].to(device))
        block=layer(h,attention_mask=None,position_ids=pos.unsqueeze(0),past_key_value=b_cache,
                    cache_position=pos,position_embeddings=rotary,current_ut=loop,use_cache=True)
        for name,out,ref in [('attention',attn,r['attention_out']),('MLP',mlp,r['mlp_out']),
                             ('norm',layer.input_layernorm(h),r['attention_in']),('full_layer',block,r['hidden_out'])]:
            e=errors(out.cpu(),ref)
            result.append(dict(step=step,loop=loop,component=name,error=e,PASS=e['relative_L2']<=TOL['operator_relative_L2']))
        for observed,ref in zip([a_cache.key_cache[24*loop],a_cache.value_cache[24*loop]],r['KV']):
            ref=ref[:,4*model.rank:4*(model.rank+1)]
            e=errors(observed.cpu(),ref)
            result.append(dict(step=step,loop=loop,component='KV_head_slice',error=e,PASS=e['relative_L2']<=TOL['KV_relative_L2']))
    write(ROOT/'correctness'/f'{label}_rank{model.rank}.json',result)
    write(ROOT/'correctness'/f'{label}_operations_rank{model.rank}.json',model.meter.operations)
    assert all(r['PASS'] for r in result),result
    print('TP_OPERATOR_PASS',model.rank,len(result),flush=True)


@torch.inference_mode()
def correctness(model,args):
    summaries=[]
    for P,O in [(128,8),(512,32),(2048,32)]:
        if args.smoke and P!=128:continue
        gold=torch.load(ROOT/f'.tensor_cache/native_P{P}_O{O}.pt',weights_only=False)
        requests=[dict(id=k,input_ids=r['input_ids']) for k,r in gold.items()]
        model.configure(requests,O)
        for rid,r in gold.items():
            actual=[];logit=[];hidden=[]
            for step in range(O):
                ids=torch.tensor([r['input_ids']] if step==0 else [[r['ids'][step-1]]],device=model.device)
                z,token,h=model.forward(ids,rid,step,P,graphs=args.graphs)
                observed=int(z[0,-1].argmax());actual.append(observed)
                logit.append(errors(z[0,-1].float().cpu(),r['logits'][step]))
                hidden.append(errors(h.float().cpu(),r['hidden'][step]))
            num=den=0.;max_abs=0.
            local=model.caches[rid][:,:,:,:,:P+O-1].cpu()
            for slot,(k,v) in r['cache'].items():
                for j,ref in enumerate([k,v]):
                    a=local[slot,j].float();b=ref[:,4*model.rank:4*(model.rank+1)].float()
                    delta=a-b;num+=float(delta.square().sum());den+=float(b.square().sum());max_abs=max(max_abs,float(delta.abs().max()))
            kv_rel=(num/max(den,1e-30))**.5
            stable=[i for i,gap in enumerate(r['margins']) if gap>=TOL['stable_native_margin']]
            mismatches=[i for i in stable if actual[i]!=r['ids'][i]]
            summary=dict(rank=model.rank,P=P,O=O,id=rid,graphs=args.graphs,teacher_forced=True,
                observed_argmax_ids=actual,native_ids=r['ids'],stable_positions=stable,stable_mismatches=mismatches,
                all_token_agreement=actual==r['ids'],logit_errors=logit,hidden_errors=hidden,
                KV_relative_L2=kv_rel,KV_max_abs=max_abs,KV_local_shape=list(local.shape),
                PASS=(max(x['max_abs'] for x in logit)<=TOL['logit_max_abs'] and
                    max(x['mean_abs'] for x in logit)<=TOL['logit_mean_abs'] and
                    max(x['relative_L2'] for x in hidden)<=TOL['hidden_relative_L2'] and
                    kv_rel<=TOL['KV_relative_L2'] and not mismatches))
            summaries.append(summary)
            write(ROOT/'correctness'/f'{args.label}_rank{model.rank}.json',summaries)
            assert summary['PASS'],summary
        # Real free-running greedy feedback, separate from aligned numeric comparisons.
        model.configure(requests,O)
        for rid,r in gold.items():
            observed=[];inp=torch.tensor([r['input_ids']],device=model.device)
            for step in range(O):
                z,token,h=model.forward(inp,rid,step,P,graphs=args.graphs)
                observed.append(int(token));inp=token
                if observed[-1]==0:break
            summaries.append(dict(rank=model.rank,P=P,O=O,id=rid,lane='free_running_greedy',
                ids=observed,native_ids=r['ids'],token_exact=observed==r['ids'],
                first_divergence=next((i for i,(a,b) in enumerate(zip(observed,r['ids'])) if a!=b),None),
                finish_reason='eos' if observed[-1]==0 else 'length'))
        write(ROOT/'correctness'/f'{args.label}_rank{model.rank}.json',summaries)
        # A shape has its own graph storage. Do not retain unrelated shape pools.
        model.graphs.clear();model.caches.clear();del gold,local;gc.collect();torch.cuda.empty_cache()
    print('TP_FULL_CORRECTNESS',model.rank,'graphs',args.graphs,flush=True)


@torch.inference_mode()
def one_case(model,c,args):
    requests=c['requests'];O=c['O'];device=model.device
    before=time.perf_counter();initial_graphs=len(model.graphs)
    model.configure(requests,O)
    # Every exact key length and full-prompt shape is captured before admission.
    for P in sorted({len(r['input_ids']) for r in requests}):
        row=next(r for r in requests if len(r['input_ids'])==P)
        model.configure([row],O)
        for step in range(O):
            inp=torch.tensor([row['input_ids']] if step==0 else [[row['teacher_ids'][step-1]]],device=device)
            z,token,h=model.forward(inp,row['id'],step,P,graphs=args.graphs)
            token.item()
    model.configure(requests,O)
    inputs={r['id']:torch.tensor([r['input_ids']],device=device) for r in requests}
    feedback={r['id']:torch.tensor(r['teacher_ids'],device=device).view(-1,1,1) for r in requests}
    prepared=time.perf_counter()-before
    for r in requests:model.caches[r['id']].zero_()
    dist.barrier();torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    ngraphs=len(model.graphs);meter_start=len(model.meter.operations)
    output={r['id']:dict(id=r['id'],input_ids=r['input_ids'],ids=[],computed_ids=[],token_times_abs_s=[]) for r in requests}
    events=[]
    if args.nsys:torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push('MEASURED_WORKLOAD')
    start=time.perf_counter()
    for step in range(O):
        for r in requests:
            rid=r['id'];P=len(r['input_ids']);inp=inputs[rid] if step==0 else feedback[rid][step-1]
            torch.cuda.nvtx.range_push(f'TP4|{rid}|step{step}|rank{model.rank}')
            if args.trace:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            z,token,h=model.forward(inp,rid,step,P,graphs=args.graphs)
            if args.trace:b.record()
            actual=int(token);finish=time.perf_counter()
            out=output[rid];out['ids'].append(r['teacher_ids'][step]);out['computed_ids'].append(actual);out['token_times_abs_s'].append(finish)
            if args.trace:events.append(dict(rank=model.rank,id=rid,step=step,host_end_abs_s=finish,full_forward_bracket_ms=a.elapsed_time(b)))
            torch.cuda.nvtx.range_pop()
    end=time.perf_counter();torch.cuda.nvtx.range_pop()
    if args.nsys:torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()
    result=dict(arm='TP4_matched',case={k:v for k,v in c.items() if k!='requests'},requests=output,
        rank=model.rank,physical_GPU=model.rank+4,start_abs_s=start,end_abs_s=end,root_wall_s=end-start,
        graph_preparation_wall_s=prepared,graph_captures_during_run=len(model.graphs)-ngraphs,
        direct_python_collectives=model.meter.operations[meter_start:],
        captured_collectives_by_shape={str(k):v[-1] for k,v in model.graphs.items()},
        memory=model.memory(),events=events,graphs=args.graphs,trace=args.trace,
        scope='Actual batch-one round-robin TP4. Frozen feedback, full LM head and argmax, one token broadcast per forward.')
    target=ROOT/'raw'/args.label;target.mkdir(parents=True,exist_ok=True)
    with gzip.open(target/f'case_{c["index"]:02d}_rank{model.rank}.json.gz','wt') as f:json.dump(result,f,separators=(',',':'))
    if model.rank==0:print('TP_MEASURED',c['index'],c['C'],c['P'],O,round(end-start,4),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['operators','correctness','benchmark'],required=True)
    parser.add_argument('--graphs',action='store_true');parser.add_argument('--trace',action='store_true');parser.add_argument('--nsys',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--indices',default='');parser.add_argument('--label',default='tp_pilot')
    parser.add_argument('--attention-geometry-control',action='store_true')
    args=parser.parse_args()
    if args.mode=='benchmark':require_qualification()
    rank=initialize();model=TPModel(rank,attention_geometry_control=args.attention_geometry_control)
    write(ROOT/'ownership'/f'{args.label}_rank{rank}.json',dict(rank=rank,physical_GPU=rank+4,
        UUID=str(torch.cuda.get_device_properties(rank).uuid),weights=model.weights,
        QKV_weight_shape=list(model.layers[0].self_attn.q_proj.weight.shape),
        O_weight_shape=list(model.layers[0].self_attn.o_proj.weight.shape),
        MLP_gate_weight_shape=list(model.layers[0].mlp.gate_proj.weight.shape),
        down_weight_shape=list(model.layers[0].mlp.down_proj.weight.shape),torch=torch.__version__,CUDA=torch.version.cuda,NCCL=torch.cuda.nccl.version()))
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream(device=rank)):
        if args.mode=='operators':
            if args.nsys:
                torch.cuda.cudart().cudaProfilerStart();torch.cuda.nvtx.range_push('TP_OPERATOR_DIAGNOSTIC')
            operators(model,args.label)
            if args.nsys:
                torch.cuda.nvtx.range_pop();torch.cuda.cudart().cudaProfilerStop()
        elif args.mode=='correctness':correctness(model,args)
        else:
            selected=cases();indices=set(map(int,args.indices.split(','))) if args.indices else None
            old_shape=None
            for c in selected:
                if indices is not None and c['index'] not in indices:continue
                shape=(tuple(sorted({len(r['input_ids']) for r in c['requests']})),c['O'])
                if old_shape!=shape:
                    model.graphs.clear();model.caches.clear();gc.collect();torch.cuda.empty_cache();old_shape=shape
                one_case(model,c,args)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
