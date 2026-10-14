"""Native model loader and one native graph set, without prior event graphs."""
import torch
from poc_dllm_multigpu_e2e_atlas.contract import *


def setup(rank,port,args):
    from poc_dllm_prefix_canvas_cp.runtime import setup as previous
    args.backend='tp4';args.discovery=False;args.graphs=False
    model,initial,initial_dllm,tokenizer=previous(rank,port,args)
    server=initial.server_args
    del initial_dllm,initial
    runner,dllm=build(model,server,args.prefix,args.eager_prefill)
    write_layout(model,runner,rank,args)
    return model,runner,dllm,tokenizer


def build(model,server,prefix,eager_prefill=False):
    from dinfer.decoding.diffusion_runner import ModelRunner
    from dinfer import BlockDiffusionLLM,ThresholdParallelDecoder,BlockIteratorFactory,KVCacheFactory
    runner=ModelRunner(model,device=str(model.device),server_args=server,max_length=prefix+OUTPUT,
        enable_cuda_graph=True,enable_compile=False,supported_batch_sizes=[1],block_length=BLOCK,
        cache_lengths=[2*prefix],decoding_lengths=[32,64],
        prefill_lengths=[32] if eager_prefill else [prefix],use_cross_block=True)
    dllm=BlockDiffusionLLM(runner,ThresholdParallelDecoder(temperature=0,threshold=.9,mask_id=156895,eos_id=156892),
        BlockIteratorFactory(start_block_align=True,use_block_diffusion=True),
        KVCacheFactory('prefix',is_bd_model=True,backend='sglang',max_length=prefix+OUTPUT),
        early_stop=True,maximum_unroll=1,expected_tpf=15,backend='sglang',mini_batch_size=1,
        prefilling_limit=32768,use_naive_batching=True)
    return runner,dllm


def write_layout(model,runner,rank,args):
    write(Path(args.output)/f'native_layout.rank{rank}.json',dict(rank=rank,physical_gpu=GPUS[rank],
        cuda_uuid=str(torch.cuda.get_device_properties(rank).uuid),TP=4,EP=1,DP=1,PP=1,CP=1,
        graphs=[list(k) for k in runner.graph_runner.graphs],eager_prefill=args.eager_prefill,
        prefix=args.prefix,physical_keys=2*args.prefix,
        parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters()),
        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        weights=[dict(name=n,shape=list(v.shape),bytes=v.numel()*v.element_size()) for n,v in model.named_parameters()],
        cache_shape=list(runner.graph_runner.past_key_values.shape),
        EP_dispatch_all_to_all='NOT_PRESENT',EP_combine_all_to_all='NOT_PRESENT',
        torch=torch.__version__,CUDA=torch.version.cuda,NCCL=list(torch.cuda.nccl.version())))
