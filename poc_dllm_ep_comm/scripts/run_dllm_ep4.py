"""Native denoising on real sharded experts; no trace-only fake EP mapping."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import signal
import types
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_dllm_ep_comm.contract import ROOT,MODEL,MASK,GPUS,UUIDS,write
from poc_dllm_ep_comm.instrumentation import Recorder

def worker(rank,port,args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    torch.cuda.set_device(rank)
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
    if rank==0:
        write(Path(args.output)/'clean_start.json',dict(mask=MASK,
            gpus=subprocess.check_output(['nvidia-smi','--id='+MASK,'--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True),
            start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),LOCAL_RANK=str(rank),RANK=str(rank),WORLD_SIZE='4')
    from transformers import AutoConfig,AutoTokenizer
    from sglang.srt import distributed
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from dinfer import BlockIteratorFactory,KVCacheFactory,BlockDiffusionLLM,ThresholdParallelDecoder
    distributed.init_distributed_environment(4,rank,'env://',rank,'nccl')
    distributed.initialize_model_parallel(4,4,1,backend='nccl')
    cfg=AutoConfig.from_pretrained(MODEL,trust_remote_code=True)
    cfg.torch_dtype=torch.bfloat16
    server=ServerArgs(model_path=str(MODEL),trust_remote_code=True,tp_size=4,ep_size=4,dp_size=1,pp_size=1,
        enable_dp_attention=False,moe_a2a_backend='deepep',deepep_mode='normal')
    # Match the validated manual source-row partition bridge. ServerArgs may
    # auto-enable DP attention for DeepEP, which this DP1 runner does not use.
    server.enable_dp_attention=False
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        pass
    else:
        set_global_server_args_for_scheduler(server)
    initialize_dp_attention(server_args=server,model_config=cfg);initialize_moe_config(server)
    torch.set_default_dtype(torch.bfloat16)
    runtime=Path(args.output)/'runtime';runtime.mkdir(exist_ok=True)
    model=LLaDA2SGLangLM(config=cfg,expert_map_path=str(runtime)).eval()
    model.load_weights(str(MODEL),device=torch.device('cuda',rank));model=model.to(rank);model.after_processing()
    model.device=torch.device('cuda',rank)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    ownership=[]
    for layer in model.model.layers:
        block=layer.mlp
        if not hasattr(block,'deepep_dispatcher'):continue
        ex=block.experts;owned=torch.where(ex.expert_map_cpu>=0)[0].tolist()
        assert owned==list(range(rank*64,(rank+1)*64)) and ex.num_local_experts==64
        weights={n:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size()) for n,v in ex.named_parameters()}
        ownership.append(dict(layer=block.layer_id,owned=owned,weights=weights,
            dispatcher=type(block.deepep_dispatcher).__name__,kernel=type(ex.quant_method).__name__))
    assert len(ownership)==19
    write(Path(args.output)/f'ownership.rank{rank}.json',dict(rank=rank,physical_gpu=GPUS[rank],
        actual_cuda_uuid=str(torch.cuda.get_device_properties(rank).uuid),mask=MASK,
        topology='dense TP4 / source-SP4 / physical EP4 / DP1 / PP1',ownership=ownership,
        versions={k:importlib.metadata.version(k) for k in ('sglang','deep-ep','transformers','triton')},
        torch=torch.__version__,cuda=torch.version.cuda,nccl=list(torch.cuda.nccl.version()),
        torch_compile_disabled=os.environ.get('TORCHDYNAMO_DISABLE','0')))
    frozen=json.loads((ROOT/'results/frozen_manifest.json').read_text())
    rows=frozen['rows']
    # The balanced interleave below never looks at model outputs.
    rows=sorted(rows,key=lambda r:(int(r['request_id'].rsplit('_',1)[1]),r['domain']))[:args.limit]
    # Native SGLang runner unwraps KVCache and pads the cross-block attention
    # mask. Reuse it; passing KVCache directly to the model is not valid.
    from dinfer.decoding.diffusion_runner import ModelRunner
    runner=ModelRunner(model,device=str(model.device),server_args=server,max_length=4096,
        enable_cuda_graph=False,enable_compile=False,supported_batch_sizes=[1],
        block_length=32,decoding_lengths=[32,64],prefill_lengths=[32,64,96,128],use_cross_block=True)
    decoder=ThresholdParallelDecoder(temperature=0,threshold=.9,mask_id=156895,eos_id=156892)
    dllm=BlockDiffusionLLM(runner,decoder,BlockIteratorFactory(start_block_align=True,use_block_diffusion=True),
        KVCacheFactory('prefix',is_bd_model=True,backend='sglang',max_length=4096),early_stop=True,
        maximum_unroll=1,expected_tpf=15,backend='sglang',mini_batch_size=1,prefilling_limit=128,use_naive_batching=True)
    rec=Recorder(rank);rec.install(model,dllm)
    def watchdog(signum,frame):raise TimeoutError('per-request120s watchdog')
    signal.signal(signal.SIGALRM,watchdog)
    def generate(row,block,mode,name):
        torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
        prompt='<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+row['prompt']+'<|role_end|><role>ASSISTANT</role>'
        ids=tokenizer(prompt,return_tensors='pt')['input_ids'].to(rank)
        assert ids.shape[1]<3900
        rec.begin(mode,row['request_id'],ids.shape[1]);dist.barrier();torch.cuda.synchronize(rank)
        before=dllm.num_forwards;signal.alarm(120)
        a=Recorder.event();wall=time.perf_counter()
        try:out=dllm.generate(ids,gen_length=64,block_length=block)
        finally:signal.alarm(0)
        b=Recorder.event();torch.cuda.synchronize(rank)
        wall_ms=(time.perf_counter()-wall)*1000;event_ms=a.elapsed_time(b)
        tokens=out[0].detach().cpu().tolist();new=tokens[ids.shape[1]:]
        result=dict(request_id=row['request_id'],domain=row['domain'],mode=mode,block_length=block,
            prompt_tokens=ids.shape[1],output_ids=new,output_text=tokenizer.decode(new,skip_special_tokens=False),
            nfe=dllm.num_forwards-before,wall_ms=wall_ms,event_ms=event_ms)
        dest=Path(args.output)/name;dest.mkdir(exist_ok=True)
        write(dest/f'{mode}.result.rank{rank}.json',result)
        if mode!='bypass':rec.finish(dest,mode,new)
        if rank==0:print(name,mode,'NFE',result['nfe'],'TOKENS',len(new),'WALL_MS',round(wall_ms,2),flush=True)
        return result
    # Three actual denoising warmups, including both supported block shapes.
    warm=dict(request_id='warmup',domain='diagnostic',prompt='What is two plus three?')
    for n,block in enumerate((32,32,64)):generate(warm,block,'bypass',f'warmup{n}')
    for index,row in enumerate(rows):
        name=row['request_id']+'_b32'
        # Counterbalanced clean timing order; trace follows independently.
        modes=('bypass','timing') if index%2==0 else ('timing','bypass')
        for mode in modes:generate(row,32,mode,name)
        generate(row,32,'trace',name)
        if index<2:generate(row,32,'acceptance',name)
    if args.secondary:
        for row in rows:
            if int(row['request_id'].rsplit('_',1)[1])>=4:continue
            name=row['request_id']+'_b64'
            for mode in ('bypass','timing','trace'):generate(row,64,mode,name)
    dist.barrier();dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=32)
    p.add_argument('--secondary',action='store_true');args=p.parse_args()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    try:mp.spawn(worker,args=(port,args),nprocs=4,join=True)
    except Exception as error:
        write(Path(args.output)/'status.json',dict(ok=False,status='RUNTIME_UNAVAILABLE',error=str(error)));raise
    write(Path(args.output)/'status.json',dict(ok=True,status='COMPLETE',physical_gpus=MASK))

if __name__=='__main__':main()
