"""Physical service-only screen: native custom AR pair versus stock AG/RS."""
import argparse
from contextlib import nullcontext
import datetime as dt
import os
from pathlib import Path
import inspect
import sys
import statistics
import time
import traceback

from poc_ouro_parallelism_deep_discovery.contract import ROOT,GPUS,UUIDS,validate_mask,write,sha,MODELS,read


def layout(C,P,hidden=2048):
    if C<=0 or P not in (2,4):
        raise ValueError('Require positive rows and P2/P4')
    local=(C+P-1)//P
    return dict(useful_rows=C,padded_rows=P*local,local_rows=local,hidden=hidden,
        BF16_bytes=2,AR_input_bytes_per_rank_per_call=C*hidden*2,
        AG_input_bytes_per_rank=local*hidden*2,AG_output_bytes_per_rank=P*local*hidden*2,
        RS_input_bytes_per_rank=P*local*hidden*2,RS_output_bytes_per_rank=local*hidden*2,
        note='Tensor payload capacities,not measured link traffic. Padding remains charged.')


def check_mask(mask):
    if mask not in ([4,5],[4,5,6,7]):
        raise ValueError('Frozen masks are ordered physical[4,5] or[4,5,6,7]')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,nargs='+',default=[1,2,4,8,32,128])
    parser.add_argument('--repeats',type=int,default=31)
    parser.add_argument('--inner',type=int,default=16)
    args=parser.parse_args()
    if args.repeats!=31 or args.inner!=16 or args.rows!=[1,2,4,8,32,128]:
        raise ValueError('Use the frozen rows,31trials and16inner pairs')
    mask=validate_mask()
    check_mask(mask)
    rank=int(os.environ['LOCAL_RANK'])
    if 'torch' in sys.modules:
        raise RuntimeError('Pin allocator aliases before the first Torch import')
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:False'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
    from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
    configure_native_imports()
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    output=Path(os.environ['LOOPED_KV_OUTPUT']).resolve()
    if not output.is_relative_to(ROOT/'outputs') or not (output/'run_contract.json').exists():
        raise RuntimeError('Use the campaign GPU ownership guard')
    ar=None
    phase='GROUP_SETUP'
    success=False
    local_partial={}
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group('gloo',timeout=dt.timedelta(seconds=120))
        P=dist.get_world_size()
        if P!=len(mask):
            raise ValueError('Logical world and physical mask differ')
        nccl=dist.new_group(backend='nccl',timeout=dt.timedelta(seconds=120))
        uuid=str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')
        if uuid!=UUIDS[GPUS.index(mask[rank])]:
            raise ValueError('Physical UUID mapping differs')
        ar=CustomAllreduce(dist.group.WORLD,torch.device('cuda',rank))
        if ar.disabled:
            raise RuntimeError('Disabled native custom AR cannot qualify this comparison')
        ownership=[None]*P
        dist.all_gather_object(ownership,dict(rank=rank,physical_gpu=mask[rank],UUID=uuid,
            model=torch.cuda.get_device_name(rank),custom_AR_enabled=True))
        if rank==0:
            write(output/'ownership.json',ownership)
            write(output/'source_contract.json',dict(script_sha256=sha(__file__),
                custom_AR_sha256=sha(inspect.getfile(CustomAllreduce)),
                model_config_sha256={s:sha(v['path']/'config.json') for s,v in MODELS.items()},
                torch=torch.__version__,CUDA=torch.version.cuda,NCCL=torch.cuda.nccl.version(),
                allocator_aliases={key:os.environ[key] for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
                precision='BF16',model_weights_loaded=False,
                evidence_class='GPU_MICROBENCH',candidate_E2E='NOT_RUN',candidate_quality='NOT_RUN'))
        records=[]
        for C in args.rows:
            phase='CAPTURE_AND_FUNCTIONAL_CHECK'
            local_partial=dict(C=C,samples={})
            spec=layout(C,P)
            if any((read(info['path']/'config.json')['hidden_size']!=2048) for info in MODELS.values()):
                raise ValueError('Actual model hidden geometry differs')
            x=torch.full((C,2048),rank+1,dtype=torch.bfloat16,device='cuda')
            owned=torch.zeros((spec['local_rows'],2048),dtype=torch.bfloat16,device='cuda')
            for local in range(spec['local_rows']):
                if rank*spec['local_rows']+local<C:
                    owned[local].fill_(rank+1)
            gathered=torch.empty((spec['padded_rows'],2048),dtype=torch.bfloat16,device='cuda')
            reduced=torch.empty_like(owned)
            def native_pair():
                first=ar.custom_all_reduce(x)
                if first is None:
                    raise RuntimeError('Native fast path fell back')
                second=ar.custom_all_reduce(first)
                if second is None:
                    raise RuntimeError('Native second fast path fell back')
                return first,second
            def hybrid_pair():
                dist.all_gather_into_tensor(gathered,owned,group=nccl)
                dist.reduce_scatter_tensor(reduced,gathered,group=nccl)
                return gathered,reduced
            functions={'CUSTOM_AR_PAIR':native_pair,'STOCK_AG_RS_PAIR':hybrid_pair}
            graphs={}
            answers={}
            for name,fn in functions.items():
                for _ in range(7):
                    fn()
                torch.cuda.synchronize()
                dist.barrier()
                graph=torch.cuda.CUDAGraph()
                context=ar.capture() if name=='CUSTOM_AR_PAIR' else nullcontext()
                with context:
                    with torch.cuda.graph(graph):
                        for _ in range(args.inner):
                            result=fn()
                graph.replay()
                torch.cuda.synchronize()
                answers[name]=result
                graphs[name]=graph
            native_first,native_second=answers['CUSTOM_AR_PAIR']
            ag,rs=answers['STOCK_AG_RS_PAIR']
            total=P*(P+1)//2
            expected_ag=torch.cat([torch.full_like(owned,r+1) if (r+1)*spec['local_rows']<=C else
                torch.stack([torch.full((2048,),r+1 if r*spec['local_rows']+j<C else 0,
                    dtype=torch.bfloat16,device='cuda') for j in range(spec['local_rows'])]) for r in range(P)])
            correct=bool(torch.all(native_first==total)) and bool(torch.all(native_second==P*total)) \
                    and torch.equal(ag,expected_ag) and torch.equal(rs,owned*P)
            checks=[None]*P
            dist.all_gather_object(checks,dict(rank=rank,functional_collective_fixture_PASS=correct))
            if rank==0:
                write(output/f'C{C}_correctness.json',dict(layout=spec,ranks=checks,
                    criterion='Exact integer-valued BF16 collective outputs. Not model or substitution parity.'))
            if not all(r['functional_collective_fixture_PASS'] for r in checks):
                raise RuntimeError('Collective fixture correctness failed before timing')
            samples={name:[] for name in functions}
            local_partial['samples']=samples
            phase='TIMING'
            for repeat in range(args.repeats):
                order=list(functions) if repeat%2==0 else list(reversed(functions))
                for name in order:
                    torch.cuda.synchronize()
                    dist.barrier()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    before=time.perf_counter_ns()
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(dict(repeat=repeat,event_ms_per_pair=start.elapsed_time(end)/args.inner,
                        host_ms_per_pair=(time.perf_counter_ns()-before)/1e6/args.inner))
            phase='TRIAL_GATHER'
            all_samples=[None]*P
            dist.all_gather_object(all_samples,dict(rank=rank,samples=samples))
            if rank==0:
                timing={name:dict(critical_event_ms=[max(r['samples'][name][i]['event_ms_per_pair'] for r in all_samples)
                    for i in range(args.repeats)]) for name in functions}
                for values in timing.values():
                    values['median_critical_event_ms']=statistics.median(values['critical_event_ms'])
                records.append(dict(C=C,P=P,physical_gpus=mask,layout=spec,timing=timing,per_rank=all_samples,
                    collective_fixture_correctness='PASS',model_output_correctness='NOT_RUN',
                    evidence_class='GPU_MICROBENCH',
                    boundary='Maximum rank-local duration of each complete serialized pair. No absolute cross-GPU timestamp alignment.',
                    traffic_scope='Logical tensor capacities. No physical link counter or pure bandwidth interval.',
                    limitations=['No attention,MLP,norm,router,gate,KV,packing or row scheduling executes.',
                        'These pairs have different layouts/functions without the omitted model operators.',
                        'Pair timing is not full-layer parity,critical-path share or serving speedup.',
                        'Stock NCCL AG/RS does not cover every fused or optimized hybrid implementation.',
                        'Shape compatibility does not validate both model sizes or depth modes.']))
                write(output/'collective_pairs.json',dict(records=records,ownership=ownership,
                    created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),repeats=31,inner_pairs=16,
                    model_weights_loaded=False,clean_E2E='NOT_RUN',quality='NOT_RUN',
                    purpose='Test the named stock communication substitution,not a new architecture.'))
                print('PAIR',P,C,{k:v['median_critical_event_ms'] for k,v in timing.items()},flush=True)
            dist.barrier()
        if rank==0:
            write(output/'status.json',dict(success=True,P=P,records=len(records),evidence_class='GPU_MICROBENCH'))
        success=True
    except BaseException as error:
        write(output/f'failure_rank{rank}.json',dict(error_type=type(error).__name__,phase=phase,
            traceback=traceback.format_exc(),local_partial=local_partial,
            rule='Available CPU records only. No new CUDA query after the failure. Outer guard remains authoritative.'))
        raise
    finally:
        errors=[]
        try:
            if ar is not None:
                ar.close()
        except BaseException as error:
            errors.append(dict(operation='custom_AR_close',error_type=type(error).__name__))
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except BaseException as error:
            errors.append(dict(operation='destroy_process_group',error_type=type(error).__name__))
        write(output/f'local_status_rank{rank}.json',dict(success=success,cleanup_errors=errors,
            local_partial=local_partial,phase=phase))
        if errors and success:
            raise RuntimeError('Collective cleanup did not complete normally')


if __name__=='__main__':
    main()
