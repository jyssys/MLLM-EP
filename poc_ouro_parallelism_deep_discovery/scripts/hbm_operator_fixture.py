"""Single-GPU physical-shard counter fixture. Not native TP or E2E timing."""
import argparse
import datetime as dt
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch
import torch.nn.functional as F
from safetensors import safe_open

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',choices=MODELS,required=True)
    parser.add_argument('--shards',type=int,choices=[1,2,4],default=2)
    parser.add_argument('--arrangement',choices=['alias','copies'],required=True)
    parser.add_argument('--intervening-mib',type=int,choices=[0,128,512],default=0)
    args=parser.parse_args();mask=validate_mask();assert len(mask)==1
    device=torch.cuda.get_device_properties(0)
    assert str(device.uuid).removeprefix('GPU-')==UUIDS[GPUS.index(mask[0])]
    info=MODELS[args.size]
    capture=ROOT/('outputs/capture_base14_v2/1.4B/actual_mlp_inputs.pt' if args.size=='1.4B'
                  else 'outputs/capture_base26_v1/2.6B/actual_mlp_inputs.pt')
    values=torch.load(capture,map_location='cpu',weights_only=True)
    key='fixed_0_layer0_call5';x=values[key].to('cuda');assert x.shape==(1,2048)
    with safe_open(info['path']/'model.safetensors',framework='pt',device='cpu') as source:
        wg=source.get_tensor('model.layers.0.mlp.gate_proj.weight').chunk(args.shards,0)[0].contiguous().to('cuda')
        wu=source.get_tensor('model.layers.0.mlp.up_proj.weight').chunk(args.shards,0)[0].contiguous().to('cuda')
        wd=source.get_tensor('model.layers.0.mlp.down_proj.weight').chunk(args.shards,1)[0].contiguous().to('cuda')
    original=[wg,wu,wd];copies=[[v.clone() for v in original] for _ in range(3)]
    all_sets=[original]+copies
    assert all(torch.equal(v.view(torch.int16),w.view(torch.int16)) for group in copies for v,w in zip(original,group))
    current=[original]*4 if args.arrangement=='alias' else all_sets
    disturbance=torch.ones(max(1,args.intervening_mib*1024**2//4),device='cuda')
    def mlp(weights):return F.linear(F.silu(F.linear(x,weights[0]))*F.linear(x,weights[1]),weights[2])
    reference=mlp(original)
    for _ in range(7):
        for weights in current:mlp(weights)
    torch.cuda.synchronize()
    outputs=[]
    for weights in current:
        if args.intervening_mib:disturbance.add_(1.)
        torch.cuda.nvtx.range_push('HBM_OPERATOR')
        outputs.append(mlp(weights))
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    assert all(torch.equal(v.view(torch.int16),reference.view(torch.int16)) for v in outputs)
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    write(output/'fixture.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),physical_gpu=mask[0],
        logical_cuda=0,UUID=str(device.uuid),model=info['id'],revision=info['revision'],
        arrangement=args.arrangement,shards=args.shards,selected_shard=0,rows=1,
        parameter_bytes_per_shard=sum(v.numel()*v.element_size() for v in original),
        four_value_identical_shards_allocated_in_both_arms=True,
        source_capture=str(capture.relative_to(ROOT)),source_sha256=sha(capture),source_key=key,
        intervening_MiB=args.intervening_mib,output_bitwise_equal=True,
        allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        evidence_class='GPU_MICROBENCH',
        scope='Actual pretrained single-GPU MLP-shard fixture. No collective, native multi-GPU serving, or task accuracy result.',
        timing='Profiler-perturbed execution. No latency headline is derived from this fixture.',
        traffic='Only successfully collected hardware counters can establish device traffic for this fixture. Allocated bytes are not traffic.'))

if __name__=='__main__':main()
