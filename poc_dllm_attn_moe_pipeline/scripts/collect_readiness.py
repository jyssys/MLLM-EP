"""Reuse validated model setup; collect clean/event/trace, bounded split costs."""
import argparse
import ast
import inspect
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_dllm_attn_moe_pipeline.contract import *
from poc_dllm_attn_moe_pipeline.readiness import Observer


def setup(rank, port, args):
    import poc_dllm_ep_comm.scripts.run_dllm_ep4 as prior
    tree = ast.parse(inspect.getsource(prior.worker))
    function = tree.body[0]
    index = next(i for i,n in enumerate(function.body) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'frozen' for t in n.targets))
    function.name = 'validated_setup'
    function.body = function.body[:index] + [ast.Return(ast.Call(ast.Name('locals', ast.Load()), [], []))]
    ast.fix_missing_locations(tree)
    exec(compile(tree, inspect.getfile(prior)+'[setup-only]', 'exec'), prior.__dict__)
    return prior.validated_setup(rank, port, args)


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def worker(rank, port, args):
    state = setup(rank, port, args)
    model, server, tokenizer = (state[k] for k in ('model','server','tokenizer'))
    from dinfer import BlockIteratorFactory, KVCacheFactory, BlockDiffusionLLM, ThresholdParallelDecoder
    from dinfer.decoding.diffusion_runner import ModelRunner
    from poc_dllm_ep_comm.instrumentation import Recorder
    runner = ModelRunner(model, device=str(model.device), server_args=server, max_length=4096,
        enable_cuda_graph=False, enable_compile=False, supported_batch_sizes=[1], block_length=32,
        decoding_lengths=[32,64], prefill_lengths=[32,64,96,128], use_cross_block=True)
    dllm = BlockDiffusionLLM(runner, ThresholdParallelDecoder(temperature=0, threshold=.9, mask_id=156895, eos_id=156892),
        BlockIteratorFactory(start_block_align=True,use_block_diffusion=True),
        KVCacheFactory('prefix',is_bd_model=True,backend='sglang',max_length=4096),early_stop=True,
        maximum_unroll=1,expected_tpf=15,backend='sglang',mini_batch_size=1,prefilling_limit=128,use_naive_batching=True)
    rec = Recorder(rank)
    rec.install(model, dllm)
    observer = Observer(rank)
    observer.decoder_observer = rec
    observer.install(model)
    observer.install_consumer_capture(model)
    def watchdog(signum, frame):
        raise TimeoutError('120s per request watchdog')
    signal.signal(signal.SIGALRM, watchdog)
    def generate(row, mode, rep, capture=False):
        torch.manual_seed(3407)
        torch.cuda.manual_seed_all(3407)
        prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+row['prompt']+'<|role_end|><role>ASSISTANT</role>'
        ids = tokenizer(prompt,return_tensors='pt')['input_ids'].to(rank)
        rec.begin('acceptance' if mode != 'clean' else 'bypass',row['request_id'],ids.shape[1])
        dist.barrier()
        torch.cuda.synchronize(rank)
        observer.begin(mode,capture=capture)
        nfe = dllm.num_forwards
        a = event()
        started = time.perf_counter()
        signal.alarm(120)
        try:
            out = dllm.generate(ids,gen_length=64,block_length=32)
        finally:
            signal.alarm(0)
        b = event()
        torch.cuda.synchronize(rank)
        wall_ms = (time.perf_counter()-started)*1000
        result = dict(request_id=row['request_id'],domain=row.get('domain','probe'),split=row.get('split','probe'),
                      mode=mode,repetition=rep,prompt_tokens=ids.shape[1],output_ids=out[0,ids.shape[1]:].cpu().tolist(),
                      nfe=dllm.num_forwards-nfe,wall_ms=wall_ms,device_span_ms=a.elapsed_time(b))
        if mode != 'clean':
            result.update(observer.finish())
            result['acceptances'] = [dict(index=x['index'],block=x['block'],native_iter=x['native_iter'],
                newly_accepted=torch.where((x['before'].flatten()==156895)&(x['after'].flatten()!=156895))[0].cpu().tolist())
                for x in rec.acceptances]
        write(Path(args.output)/f"{row['request_id']}.{mode}.{rep}.rank{rank}.json",result)
        if rank == 0:
            print(row['request_id'],mode,rep,'NFE',result['nfe'],'WALL',round(wall_ms,1),flush=True)
    warm = dict(request_id='warmup',prompt='Return a short Python function that adds two integers.')
    for n in range(3):
        generate(warm,'clean',n)
    smoke = [dict(request_id='smoke_short',prompt='2+3='),
             dict(request_id='smoke_prefix',prompt='Write a Python function with a docstring that removes repeated items from a list while preserving order.')]
    for row in smoke:
        for rep, mode in enumerate(('clean','clean','events')):
            generate(row,mode,rep)
    frozen = json.loads((ROOT/'results/main/design_freeze.json').read_text())
    rows = [r for r in frozen['rows'] if r['early_kill']]
    for index,row in enumerate(rows[:args.limit]):
        modes = ('clean','events') if index%2==0 else ('events','clean')
        for mode in modes:
            generate(row,mode,0)
        generate(row,'clean',1)
        generate(row,'trace',0,capture=index == 0)
    observer.mode = 'clean'
    from poc_dllm_attn_moe_pipeline.scripts.benchmark_two_layer_seam import benchmark_components
    benchmark_components(model,observer.snapshots,rank,Path(args.output))
    write(Path(args.output)/f'collective_contract.rank{rank}.json', dict(rank=rank,
        actual_scatter_modes={str(l.layer_id):str(l.layer_communicator.layer_scatter_modes) for l in model.model.layers},
        mask_contract='Native mask passed through unchanged; actual selected hot attention snapshot is unmasked with padded cache (see cache_mask_audit), no assumed causal mask',
        shared_expert_replicated=True, snapshots=list(observer.snapshots),
        release='combine event wait -> shared/add -> whole-buffer source all_gather -> next residual/norm'))
    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--physical',action='store_true')
    p.add_argument('--output')
    p.add_argument('--limit',type=int,default=12)
    args = p.parse_args()
    if args.physical:
        with socket.socket() as s:
            s.bind(('127.0.0.1',0))
            port = s.getsockname()[1]
        mp.spawn(worker,args=(port,args),nprocs=4,join=True)
        write(Path(args.output)/'status.json',dict(ok=True,mask=MASK,physical_EP4=True))
        return
    out = ROOT/('raw/'+(args.output or 'main_v1'))
    out.mkdir(parents=True,exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=MASK,PYTHONPATH=str(PROJECT)+':'+str(DINFER/'python'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        SGL_ENABLE_JIT_DEEPGEMM='false',SGLANG_ENABLE_JIT_DEEPGEMM='false',
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='1024',TORCHDYNAMO_DISABLE='1')
    for k in list(env):
        if k.startswith(('LLADA_EP_TRACE','LLADA_BLOCK_TRACE','LLADA_DENOISE_TRACE','FLASHVEP_','SD_EP_','AGENT_EP_')):
            env.pop(k)
    # Contract sec6: restore only pre-existing authorized burner state.
    from poc_dllm_attn_moe_pipeline.scripts.gpu_session import guarded_run
    with (out/'full.log').open('x') as log:
        code = guarded_run([PYTHON,'-m','poc_dllm_attn_moe_pipeline.scripts.collect_readiness','--physical',
                            '--output',str(out),'--limit',str(args.limit)],env,log,out,timeout=1800)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
