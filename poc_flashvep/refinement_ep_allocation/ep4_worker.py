"""Native BF16 DeepEP execution with explicit expert ownership and traffic traces."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from .common import ROOT, REPO, MODEL, REVISION, UUIDS, LEGACY_EP, append, write, rows, reference_helpers,snapshot_driver,digest

def worker(rank, port, args):
    gpus = list(map(int, args.gpus.split(',')))
    assert len(gpus) == 4 and len(set(gpus)) == 4
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpus
    torch.cuda.set_device(rank)
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-') == UUIDS[gpus[rank]].removeprefix('GPU-')
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE='4')
    from transformers import AutoConfig, AutoTokenizer
    from sglang.srt import distributed
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from dinfer import BlockIteratorFactory, KVCacheFactory, BlockDiffusionLLM, ThresholdParallelDecoder
    distributed.init_distributed_environment(4, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(4, 4, 1, backend='nccl')
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.torch_dtype = torch.bfloat16
    server = ServerArgs(model_path=str(MODEL), trust_remote_code=True, tp_size=4, ep_size=4, dp_size=1,
        pp_size=1, enable_dp_attention=False, moe_a2a_backend='deepep', deepep_mode='normal')
    server.enable_dp_attention = False
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        pass
    else:
        set_global_server_args_for_scheduler(server)
    initialize_dp_attention(server_args=server, model_config=cfg)
    initialize_moe_config(server)
    torch.set_default_dtype(torch.bfloat16)
    dest = ROOT/'ep4'/args.run_id
    runtime = dest/'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    model = LLaDA2SGLangLM(config=cfg, expert_map_path=str(runtime)).eval()
    model.load_weights(str(MODEL), device=torch.device('cuda', rank))
    model = model.to(rank)
    model.after_processing()
    model.device = torch.device('cuda', rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    inventory = []
    dispatch_counts = [0]
    hook = {'enabled': False}
    for layer in model.model.layers:
        block = layer.mlp
        if not hasattr(block, 'deepep_dispatcher'):
            continue
        ex = block.experts
        owned = torch.where(ex.expert_map_cpu >= 0)[0].tolist()
        assert owned == list(range(rank*64, (rank+1)*64)) and ex.num_local_experts == 64
        inventory.append(dict(layer=block.layer_id, experts=owned,
            weights={n:dict(shape=list(v.shape), dtype=str(v.dtype), bytes=v.numel()*v.element_size())
                     for n, v in ex.named_parameters()}, dispatcher=type(block.deepep_dispatcher).__name__))
        original = block.deepep_dispatcher.dispatch
        def identity(*a, _original=original, **kw):
            if hook['enabled']:
                dispatch_counts[0] += 1
                assert kw['topk_idx'].shape[-1] == 8
            return _original(*a, **kw)
        block.deepep_dispatcher.dispatch = identity
    assert len(inventory) == 19
    write(dest/f'ownership.rank{rank}.json', dict(rank=rank, physical_gpu=gpus[rank],
        uuid=str(torch.cuda.get_device_properties(rank).uuid), visible_gpus=args.gpus,
        topology='dense TP4 / source SP4 / physical TRUE EP4 / DP1 / PP1',
        model_revision=REVISION, inventory=inventory, torch=str(torch.__version__), cuda=torch.version.cuda,
        nccl=list(torch.cuda.nccl.version()),
        packages={n:importlib.metadata.version(n) for n in ('sglang','deep-ep','transformers','triton')},
        driver=snapshot_driver(__file__),runtime_source_sha256=digest(__import__('inspect').getfile(type(model)))))
    from dinfer.decoding.diffusion_runner import ModelRunner
    runner = ModelRunner(model, device=str(model.device), server_args=server, max_length=8192,
        enable_cuda_graph=False, enable_compile=False, supported_batch_sizes=[1], block_length=32,
        decoding_lengths=[32,64], prefill_lengths=[32,64,96,128], use_cross_block=True)
    decoder = ThresholdParallelDecoder(temperature=0, threshold=.95, mask_id=156895, eos_id=156892)
    dllm = BlockDiffusionLLM(runner, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
        KVCacheFactory('prefix', is_bd_model=True, backend='sglang', max_length=8192), early_stop=True,
        maximum_unroll=1, expected_tpf=15, backend='sglang', mini_batch_size=1,
        prefilling_limit=128, use_naive_batching=True)
    forward_counter = [0]
    original_forward = model.forward
    def counted_forward(*a, **kw):
        forward_counter[0] += 1
        return original_forward(*a, **kw)
    model.forward = counted_forward
    # Read-only reuse of the validated, bounded observer. Source hashes are frozen.
    sys.path.insert(0, str(LEGACY_EP))
    from poc_dllm_ep_comm.instrumentation import Recorder
    rec = Recorder(rank)
    rec.install(model, dllm)
    from dinfer.decoding.utils import TokenArray
    original_return = TokenArray.get_generated_tokens
    terminal = {}
    def capture_terminal(x):
        terminal['state'] = x.data.detach().clone()
        return original_return(x)
    TokenArray.get_generated_tokens = capture_terminal
    ref = reference_helpers()
    def alarm(_signum, _frame):
        raise TimeoutError('Request exceeded the 900-second watchdog')
    signal.signal(signal.SIGALRM, alarm)

    def generate(row, mode, name):
        output = dest/name/f'{mode}.result.rank{rank}.json'
        siblings=[dest/name/f'{mode}.result.rank{r}.json' for r in range(4)]
        if all(p.exists() for p in siblings):
            return json.loads(output.read_text())
        if any(p.exists() for p in siblings):
            raise RuntimeError('Incomplete rank result set. Preserve it and use a fresh run ID.')
        seed = 20260915+row['id']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if args.timing_kind == 'ISOLATED_EP4':
            import subprocess
            active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader'],text=True)
            other = {UUIDS[g] for g in range(8) if g not in gpus}
            assert not any(line.split(',')[0].strip() in other for line in active.splitlines()), 'Other EP4 group is not quiet'
        request_start = time.perf_counter()
        prompt = tokenizer.apply_chat_template([{'role':'user','content':row['raw_prompt']}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template([{'role':'user','content':row['raw_prompt']}],
            tokenize=True, add_generation_prompt=True, return_tensors='pt').to(rank)
        assert ids.shape[1] + row['gen_length_cap'] <= 8192
        rec.begin('bypass' if mode == 'hooked_noop' else mode, row['sample_id'], ids.shape[1])
        hook['enabled'] = mode == 'hooked_noop'
        dispatch_counts[0] = 0
        dist.barrier()
        torch.cuda.synchronize(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        before = dllm.num_forwards
        before_actual = forward_counter[0]
        terminal.clear()
        start = time.perf_counter()
        signal.alarm(900)
        try:
            with torch.inference_mode():
                generated = dllm.generate(ids, gen_length=row['gen_length_cap'], block_length=32)
            torch.cuda.synchronize(rank)
        finally:
            signal.alarm(0)
        elapsed = 1000*(time.perf_counter()-start)
        result_ids = generated[0].detach().cpu().tolist()
        assert result_ids[:ids.shape[1]] == ids[0].tolist()
        new = result_ids[ids.shape[1]:]
        terminal_ids = terminal['state'][0,ids.shape[1]:].detach().cpu().tolist()
        eos = 156892 in terminal_ids
        text = tokenizer.decode(new, skip_special_tokens=True)
        service_wall_ms = 1000*(time.perf_counter()-request_start)
        parsed, parser_rule = ref.parse_gsm8k(text)
        gold = row['answer'].split('####')[-1].strip().replace(',','') if row.get('answer') else None
        result = dict(row, method='P0_HOOKED' if mode == 'hooked_noop' else 'P0_FULL', mode=mode,
            run_id=args.run_id, physical_gpu=gpus[rank], ep_rank=rank, seed=seed, revision=REVISION,
            formatted_prompt=prompt, input_ids=ids[0].tolist(), prompt_tokens=ids.shape[1],
            output_ids=new, generation_tokens=len(new), raw_generation=tokenizer.decode(new, skip_special_tokens=False),
            postprocessed_generation=text, original_prompt=row['question'], parsed_answer=parsed,
            parser_rule=parser_rule, ground_truth=gold, correct=(parsed==gold) if gold is not None else None,
            nfe=forward_counter[0]-before_actual, native_num_forwards_counter=dllm.num_forwards-before,
            nfe_evidence='MEASURED_MODEL_FORWARD_COUNTER',
            wall_ms=service_wall_ms if args.timing_kind=='ISOLATED_EP4' else elapsed,
            generate_wall_ms=elapsed, request_preparation_to_text_ms=service_wall_ms,
            threshold=.95, temperature=0, block_length=32,
            nfe_definition='Every native model.forward call, including individual chunked prefill calls and KV refresh.',
            num_generated_blocks=((ids.shape[1]+terminal_ids.index(156892))//32-ids.shape[1]//32+1) if eos else (ids.shape[1]+len(terminal_ids)-1)//32-ids.shape[1]//32+1,
            remaining_mask_tokens=new.count(156895), termination_reason='REMAINING_MASK' if 156895 in new else 'EOS' if eos else 'GEN_LENGTH_CAP',
            terminal_buffer_ids=terminal_ids, native_eos_filtered=True,
            termination_evidence='MEASURED_TERMINAL_BUFFER',
            no_op_dispatch_calls=dispatch_counts[0], peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
            timing_kind='MEASURED_WALL_ISOLATED_EP4_REQUEST' if args.timing_kind=='ISOLATED_EP4' and mode in ('bypass','hooked_noop') else 'MEASURED_WALL_PARALLEL_QUALITY_OR_OBSERVER_NOT_PRIMARY_LATENCY', time=time.time())
        write(output, result)
        if mode in ('trace','timing','acceptance'):
            rec.finish(output.parent, mode, new)
        hook['enabled'] = False
        if rank == 0:
            append(dest/'quality_samples.jsonl', result)
            print(name, mode, 'NFE', result['nfe'], 'TOKENS', len(new), 'MS', round(elapsed,1),
                  'CORRECT', result['correct'], flush=True)
        return result

    warm = dict(sample_id='warmup:0', id=0, task='sanity', raw_prompt='What is two plus three?',
        question='What is two plus three?', answer=None, gen_length_cap=32)
    for i in range(2):
        generate(warm, 'bypass', f'warmup{i}')
    chosen = rows(args.manifest)
    if args.limit is not None:
        chosen = chosen[:args.limit]
    chosen = [r for i,r in enumerate(chosen) if i % args.shards == args.shard]
    for row in chosen:
        for repeat in range(args.repeats):
            name = row['sample_id'].replace(':','_') + (f'_repeat{repeat}' if args.repeats>1 else '')
            modes=['bypass','hooked_noop'] if args.parity else ['bypass']
            if (row['id']+repeat)%2:
                modes.reverse()
            results={mode:generate(row,mode,name) for mode in modes}
            baseline=results['bypass']
            if args.parity:
                noop=results['hooked_noop']
                assert baseline['output_ids']==noop['output_ids'] and baseline['nfe']==noop['nfe'], 'NOOP_PARITY_FAILED'
            if args.trace:
                traced=generate(row,'trace',name)
                assert traced['output_ids']==baseline['output_ids'], 'TRACE_PARITY_FAILED'
    dist.barrier()
    dist.destroy_process_group()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--gpus', required=True, choices=('0,1,2,3','4,5,6,7'))
    p.add_argument('--run-id', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--parity', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--timing-kind', choices=('PARALLEL_QUALITY','ISOLATED_EP4'), default='PARALLEL_QUALITY')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    assert 0 <= args.shard < args.shards
    assert args.repeats>=1
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    status = ROOT/'ep4'/args.run_id/'status.json'
    write(status.parent/'driver_metadata.json',dict(driver=snapshot_driver(__file__),
        command=sys.argv,model_revision=REVISION,physical_gpus=args.gpus,time=time.time()))
    write(status, dict(status='RUNNING', pid=os.getpid(), GPUs=args.gpus, time=time.time()))
    try:
        mp.spawn(worker, args=(port,args), nprocs=4, join=True)
    except BaseException as error:
        write(status, dict(status='FAILED', error=repr(error), GPUs=args.gpus, time=time.time()))
        raise
    write(status, dict(status='COMPLETE', GPUs=args.gpus, time=time.time()))

if __name__ == '__main__':
    main()
