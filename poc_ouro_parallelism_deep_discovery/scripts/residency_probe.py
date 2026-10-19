"""Weight-alias and intervening-access controls. Not a recurrent model executor."""
import argparse
import datetime as dt
import statistics
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--capture',type=Path,required=True);args=p.parse_args()
    mask=validate_mask();P=len(mask);assert P in [2,4]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==P
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[GPUS.index(mask[rank])]
    info=MODELS[args.size];out=Path(os.environ['LOOPED_KV_OUTPUT']);records=[]
    captured=torch.load(args.capture,map_location='cpu',weights_only=True)
    selected=next(v for k,v in captured.items() if '_layer0_call' in k and v.shape[0]>=4)
    with safe_open(info['path']/'model.safetensors',framework='pt',device='cpu') as ck:
        weights=[ck.get_tensor('model.layers.0.mlp.'+name+'.weight').to(rank)
            for name in ['gate_proj','up_proj','down_proj']]
    shard=[weights[0].chunk(P,0)[rank].contiguous(),weights[1].chunk(P,0)[rank].contiguous(),
           weights[2].chunk(P,1)[rank].contiguous()]
    copies=[[v.clone() for v in shard] for _ in range(4)]
    assert all(torch.equal(v,copies[i][j]) for i in range(4) for j,v in enumerate(shard))
    assert all(len({shard[j].data_ptr(),*(copies[i][j].data_ptr() for i in range(4))})==5 for j in range(3))
    def mlp(x,w):
        value=F.linear(F.silu(F.linear(x,w[0]))*F.linear(x,w[1]),w[2])
        dist.all_reduce(value)
        return value
    # Actual event overhead is measured, not assumed zero.
    a0,b0=torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)
    empty=torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty):a0.record();b0.record()
    empty_values=[]
    for _ in range(31):empty.replay();b0.synchronize();empty_values.append(a0.elapsed_time(b0))
    for rows in [1,16]:
        x=selected.repeat((rows+len(selected)-1)//len(selected),1)[:rows].to(rank)
        reference=mlp(x,shard)
        for intervening_MiB in [0,128,512]:
            buffer=torch.ones(max(1,intervening_MiB*1024**2//4),device='cuda',dtype=torch.float32)
            for arrangement in ['aliased','physical_copies']:
                current=[shard]*4 if arrangement=='aliased' else copies
                for _ in range(7):
                    for w in current:mlp(x,w)
                torch.cuda.synchronize();dist.barrier()
                starts=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(4)]
                ends=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(4)]
                outer_a,outer_b=torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)
                graph=torch.cuda.CUDAGraph();values=[]
                with torch.cuda.graph(graph):
                    outer_a.record()
                    for i,w in enumerate(current):
                        if intervening_MiB:buffer.add_(1.)
                        starts[i].record();values.append(mlp(x,w));ends[i].record()
                    outer_b.record()
                graph.replay();torch.cuda.synchronize()
                assert all(torch.equal(v,reference) for v in values)
                timing=[]
                for _ in range(31):
                    dist.barrier();torch.cuda.synchronize();graph.replay();outer_b.synchronize()
                    timing.append({'operator_ms':[a.elapsed_time(b) for a,b in zip(starts,ends)],
                        'whole_sequence_ms':outer_a.elapsed_time(outer_b)})
                gathered=[None]*P;dist.all_gather_object(gathered,timing)
                # Keep one critical rank per whole sequence. Do not add different component maxima.
                critical=[]
                for i in range(31):
                    winner=max(range(P),key=lambda r:gathered[r][i]['whole_sequence_ms'])
                    critical.append(dict(gathered[winner][i],critical_rank=winner))
                if rank==0:
                    records.append(dict(rows=rows,arrangement=arrangement,intervening_MiB=intervening_MiB,
                        per_rank=gathered,critical_sequence_trials=critical,
                        median_sequence_ms=statistics.median(t['whole_sequence_ms'] for t in critical),
                        median_sum_operator_ms=statistics.median(sum(t['operator_ms']) for t in critical),
                        operator_weight_bytes_per_rank=sum(w.numel()*w.element_size() for w in shard),
                        exact_output=True,evidence_class='GPU_MICROBENCH',
                        interpretation='Repeated identical pretrained MLP. This is not the full model reuse distance or a legal future-depth frontier.',
                        intervening_access='Read-modify-write of a separate buffer. L2 eviction and HBM traffic are not measured.',
                        extra_memory='Four physically distinct value-identical shard copies exist in both arms. Only pointers used by the schedule change.'))
                    print('RESIDENCY',args.size,P,rows,intervening_MiB,arrangement,records[-1]['median_sum_operator_ms'],flush=True)
                del graph,values
            del buffer
    if rank==0:write(out/'residency.json',dict(records=records,physical_gpus=mask,P=P,
        model=info['id'],revision=info['revision'],UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        event_pair_only_ms=empty_values,source_capture=str(args.capture),source_sha256=sha(args.capture),
        precision='BF16 pretrained weights. No altered routing, gate, or precision.',
        scope='A controlled alias/residency falsification. No candidate E2E or full-model quality result.',
        observer='External CUDA event nodes occur inside captured graphs. Their empty-pair duration is retained.'))
    dist.destroy_process_group()

if __name__=='__main__':main()
