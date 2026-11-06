"""Benchmark legal ready recurrent states. Never count these as online speedup."""
import hashlib
import statistics
import time
from dataclasses import replace
from .common import *


def main():
    configure()
    cfg = json.loads(Path(os.environ['ARC_CONFIG']).read_text())
    folder = ROOT / 'raw' / cfg['name']
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    from ..looped_pd_b_numerical_diagnosis import common as numerical
    numerical.ROOT = ROOT
    os.environ.update(B_NUMERICAL_VARIANT='down_norm' if cfg['model']=='ouro' else 'down_norm_adapter',
        B_NUMERICAL_RUN=cfg['name'], ABC_ALLOW_STAGED_BOUNDARY='1')
    from ..looped_pd_b_numerical_diagnosis.patches import install
    install()
    from vllm_rlt import CacheConfig, ExecutionConfig, SchedulerConfig, ExitConfig, SamplingParams
    from vllm_rlt.models import AutoModelForCausalLM
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import Operation, ExecutionBatch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.worker.cuda_graph import RecurrentGraphs
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[cfg['model']], device='cuda:0', dtype=torch.bfloat16)
    staging_chunk = max((item['chunk'] for item in cfg.get('micro_grid',[])), default=cfg.get('micro_chunk',32))
    engine = StageEngine(model, attention_backend='flash_attn_4',
        cache_config=CacheConfig(num_blocks=cfg['D_kv_blocks'], block_size=16, layout='last_exited', enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(mode='mixed', max_num_seqs=8, max_num_batched_tokens=max(136,4*staging_chunk),
            prefill_chunk_size=staging_chunk, wavefront_chunked_prefill=False),
        execution_config=ExecutionConfig(async_scheduling=False, prefill_uva=False,
            cuda_graphs=True, cuda_graph_max_batch_size=8, cuda_graph_max_graphs=16),
        exit_config=ExitConfig('ouro'))
    data = json.loads((ROOT / 'data' / cfg['dataset']).read_text())
    cache = engine.cache_manager
    if cfg.get('micro_align_depth') or cfg.get('micro_grid'):
        engine.scheduler._micro_align_depth = cfg.get('micro_align_depth', False)
        base_schedule = engine.scheduler.policy.schedule
        def aligned(scheduler, *, prefer_recurrent=False):
            from vllm_rlt.request import RequestPhase as P
            recurrent = scheduler.queues[P.RECURRENT]
            prompt = scheduler.queues[P.PREFILLING_RECURRENT]
            if scheduler._micro_align_depth and recurrent and prompt:
                d = scheduler.requests[recurrent[0]].loops_done
                p = scheduler.requests[prompt[0]].prefill_depth(scheduler.requests[prompt[0]].num_prefilled_tokens)
                if p < d:
                    return scheduler._take(P.PREFILLING_RECURRENT)
                if p > d:
                    return scheduler._take(P.RECURRENT)
            if getattr(scheduler,'_micro_force_cross',False) and recurrent and prompt:
                d=scheduler.requests[recurrent[0]].loops_done
                p=scheduler.requests[prompt[0]].prefill_depth(scheduler.requests[prompt[0]].num_prefilled_tokens)
                if d==p and d+1<cfg['depth']:
                    scheduler._micro_force_cross=False
                    return scheduler._take(P.RECURRENT)
            return base_schedule(scheduler, prefer_recurrent=prefer_recurrent)
        engine.scheduler.policy.schedule = aligned
    rows = []
    seen = set()
    dense = {}
    capture_dense = [False]
    dense_split = [None]
    current_spec = [None]
    original = ModelRunner._execute_batch

    def timing(fn, repetitions=15):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repetitions):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); fn(); end.record(); end.synchronize()
            samples.append(begin.elapsed_time(end))
        return samples

    def state(metadata):
        # Only the query rows written by this legal operation. Other KV remains
        # the same causal history; no request allocation is freed during replay.
        return tuple(tensor[metadata.write_blocks, :, metadata.write_offsets].clone()
                     for tensor in (cache.key_cache, cache.value_cache))

    def execute(runner, batch, prepared):
        phases = {str(s.request.phase) for s in batch.spans}
        from vllm_rlt.request import RequestPhase as P
        prompt = [s for s in batch.spans if s.request.phase==P.PREFILLING_RECURRENT]
        decode = [s for s in batch.spans if s.request.phase==P.RECURRENT]
        if runner is not engine.model_runner or batch.operation != Operation.RECURRENT or not prompt or not decode:
            return original(runner, batch, prepared)
        depth_label = 'same' if len({s.depth for s in batch.spans}) == 1 else 'cross'
        key = (sum(s.token_count for s in prompt), len(decode), depth_label)
        spec=current_spec[0]
        if cfg.get('micro_strict_geometry') and (len(prompt)!=1 or key!=(spec['chunk'],spec['batch'],
                'same' if spec['aligned'] else 'cross')):
            return original(runner,batch,prepared)
        if key in seen or len(seen) >= cfg.get('micro_cases', 8):
            return original(runner, batch, prepared)
        seen.add(key)
        f = prepared.forwards[0].forward
        assert f.hidden is not None
        hidden = f.hidden.clone()
        assert len(hidden) >= batch.num_rows
        hidden = hidden[:batch.num_rows]
        metadata = cache.prepare_recurrent(batch, packed=True)
        offsets, start = [], 0
        for span in batch.spans:
            offsets.append((start, start + span.token_count, span))
            start += span.token_count
        # Keep the existing decode batch intact. The separate control splits
        # phases, not individual decode requests.
        groups=[]
        for left,right,span in offsets:
            phase='prefill' if span.request.phase==P.PREFILLING_RECURRENT else 'decode'
            if groups and groups[-1][0]==phase:
                groups[-1][2]=right;groups[-1][3].append(span)
            else:
                groups.append([phase,left,right,[span]])
        parts=[(left,right,cache.prepare_recurrent(ExecutionBatch(Operation.RECURRENT,tuple(spans)),packed=True))
            for phase,left,right,spans in groups]
        execution = ExecutionConfig(cuda_graphs=True, cuda_graph_max_batch_size=max(256, len(hidden)), cuda_graph_max_graphs=8)
        fused_graph = RecurrentGraphs(model, cache, execution, True)
        separate_graphs = [RecurrentGraphs(model, cache, execution, phase=='decode')
            for phase,left,right,spans in groups]
        def call(graph, x, kv):
            selected = graph.select(x, kv)
            assert selected.entry is not None, 'Unexpected eager graph fallback'
            graph.prepare_inputs(selected, x)
            return graph.execute(selected, x)[0]
        def fused():
            return call(fused_graph, hidden, metadata)
        def separate():
            return tuple(call(graph, hidden[left:right], kv)
                for graph, (left, right, kv) in zip(separate_graphs, parts))
        warm_begin = time.perf_counter()
        capture_dense[0] = True
        dense_split[0] = [right-left for _,left,right,_ in groups]
        actual = fused().clone(); actual_kv = state(metadata)
        capture_dense[0] = False
        expected = torch.cat(separate()).clone(); expected_kv = state(metadata)
        torch.cuda.synchronize()
        exact = torch.equal(actual, expected)
        kv_exact = all(torch.equal(a, b) for a, b in zip(actual_kv, expected_kv))
        warm_s = time.perf_counter() - warm_begin
        item = dict(chunk=key[0], decode_batch=key[1], depth_class=depth_label,
            depths=[s.depth for s in batch.spans], positions=[s.token_start for s in batch.spans],
            query_rows=batch.num_rows, hidden_exact=exact, KV_exact=kv_exact,
            control_version=5,separate_phase_groups=[dict(phase=phase,rows=right-left) for phase,left,right,spans in groups],
            separate_control='Decode batch remains intact. No result concatenation inside the timed separate function.',
            phases=[str(s.request.phase) for s in batch.spans],
            gate_policy='Native mixed batch computes gates; separate decode computes gates and prefill does not.',
            max_hidden_abs_error=(actual.float()-expected.float()).abs().max().item(), warmup_host_s=warm_s,
            evidence='MEASURED_READY_STATE_MICRO_NOT_ONLINE_SERVING', graph_eager_fallbacks=0,
            scope='Full shared recurrent core with causal attention, injection, valid per-depth KV, graph preparation and packing. CUDA-event enclosure, not exclusive kernel union.')
        if exact and kv_exact:
            item['paired_repetitions'] = []
            for repeat in range(3):
                funcs = ((fused, 'batched'), (separate, 'separate')) if repeat % 2 == 0 else ((separate, 'separate'), (fused, 'batched'))
                result = {}
                for fn, label in funcs:
                    result[label+'_ms'] = timing(fn)
                item['paired_repetitions'].append(result)
            if cfg.get('micro_profile') and not rows:
                profiles = {}
                for function,label in ((fused,'batched'),(separate,'separate')):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        function(); torch.cuda.synchronize()
                    target = folder / f'core_profile_{label}.json'
                    profiler.export_chrome_trace(str(target))
                    events = json.loads(target.read_text()).get('traceEvents',[])
                    kernels = [e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
                    profiles[label] = dict(kernels=len(kernels),
                        profiled_kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
                        attention_kernels=[dict(name=e['name'],duration_us=e['dur']) for e in kernels
                            if any(word in e['name'].lower() for word in ('flash','attention'))],
                        dense_kernels=[dict(name=e['name'],duration_us=e['dur']) for e in kernels
                            if any(word in e['name'].lower() for word in ('gemm','matmul'))],
                        scope='Separate profiler diagnostic. Kernel counts/names only support attribution. Profiled times are not clean serving measurements.')
                write(folder/'core_kernel_profiles.json',profiles)
        else:
            item['status'] = 'BLOCKED_EXACTNESS_NO_PERFORMANCE_EXPANSION'
        rows.append(item)
        write(folder / 'full_core_micro.json', rows)
        return original(runner, batch, prepared)

    ModelRunner._execute_batch = execute
    # Capture a real dense input from a valid ready operation. Depth labels are
    # deliberately not an arithmetic input to Linear. Same/cross controls below
    # use the exact same tensor, weights, shape and graph policy.
    for name, module in model.named_modules():
        core = name.startswith('model.layers.') if cfg['model']=='ouro' else (
            name.startswith('transformer.core_block.') or name.startswith('transformer.adapter'))
        if core and isinstance(module, torch.nn.Linear) and any(x in name for x in ('q_proj', 'down_proj', 'mlp.proj', 'adapter')):
            original_forward = module.forward
            def forward(x, _original=original_forward, _name=name):
                if capture_dense[0] and _name not in dense and len(x) > 1 and len(x) <= 256:
                    dense[_name] = (list(dense_split[0]), x.detach().clone(), _original)
                return _original(x)
            module.forward = forward

    sampling = lambda seed, cap: SamplingParams(max_tokens=cap, min_loops=cfg['depth'], max_loops=cfg['depth'],
        exit_threshold=1., temperature=0., ignore_eos=True, seed=seed)
    completed = {}
    grid = cfg.get('micro_grid', [dict(chunk=cfg.get('micro_chunk',32), batch=cfg.get('micro_decode_batch',2),
        aligned=cfg.get('micro_align_depth',False))])
    for case, spec in enumerate(grid):
        current_spec[0]=spec
        added = False
        begin = time.monotonic()
        engine.scheduler.config = replace(engine.scheduler.config, prefill_chunk_size=spec['chunk'])
        engine.scheduler.policy.config = engine.scheduler.config
        engine.scheduler._micro_align_depth = spec['aligned']
        engine.scheduler._micro_force_cross = bool(cfg.get('micro_strict_geometry') and not spec['aligned'])
        decode_count = spec['batch']
        for i in range(decode_count):
            engine.add_request(f'case{case}_decode_{i}', data[i]['input_ids'][cfg['model']], sampling(cfg['seed']+i, 8))
        decoded=set()
        while engine.has_unfinished_requests():
            assert time.monotonic()-begin < 420, 'Bounded micro workload'
            outputs = engine.step()
            decoded.update(o.request_id for o in outputs if '_decode_' in o.request_id and len(o.token_ids)>0)
            ready_to_add=len(decoded)==decode_count if cfg.get('micro_strict_geometry') else any(len(o.token_ids) for o in outputs)
            if not added and ready_to_add:
                engine.add_request(f'case{case}_prompt', data[decode_count]['input_ids'][cfg['model']], sampling(cfg['seed']+decode_count, 2))
                added = True
            for out in outputs:
                if out.finished:
                    completed[out.request_id] = dict(tokens=list(out.token_ids), depths=list(out.exit_depths))
        if cfg.get('micro_strict_geometry'):
            assert (spec['chunk'],spec['batch'],'same' if spec['aligned'] else 'cross') in seen,'Requested legal ready state was not captured'
    ModelRunner._execute_batch = original
    dense_rows = []
    for name, (splits, x, fn) in dense.items():
        if len(dense_rows) >= 3:
            break
        # Identical dense workload. Separate execution uses the original fixed
        # projection/norm fixes. The same/cross labels do not alter arithmetic.
        cut = splits[0]
        if cut == len(x):
            continue
        expected = torch.cat((fn(x[:cut]), fn(x[cut:])))
        actual = fn(x)
        if not torch.equal(expected,actual):
            dense_rows.append(dict(name=name,rows=len(x),columns=x.shape[1],exact=False,
                control_version=3,status='BLOCKED_DENSE_OUTPUT_PARITY_NO_TIMING',
                max_abs_error=(expected.float()-actual.float()).abs().max().item(),
                evidence='MEASURED_DENSE_CORRECTNESS_FAILURE'))
            continue
        def captured(function):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                saved = function()
            torch.cuda.current_stream().wait_stream(stream)
            return graph, saved
        separate_graph, separate_saved = captured(lambda: (fn(x[:cut]), fn(x[cut:])))
        batch_graph, batch_saved = captured(lambda: fn(x))
        repetitions = []
        for repeat in range(3):
            result = {}
            order = ('separate','same_depth','cross_depth') if repeat%2 == 0 else ('cross_depth','same_depth','separate')
            for label in order:
                graph = separate_graph if label=='separate' else batch_graph
                result[label+'_ms'] = timing(graph.replay, 30)
            repetitions.append(result)
        row = dict(name=name, rows=len(x), columns=x.shape[1], exact=torch.equal(expected, actual),
            control_version=3,separate_phase_rows=splits,source='Legal mixed recurrent-core operation; no Prelude/Coda linear is used.',
            paired_repetitions=repetitions, identical_same_cross_graph=True,
            label_control='Identical input tensor and weight pointer. Depth is not a Linear argument.',
            evidence='MEASURED_DENSE_MICRO_NOT_SERVING')
        dense_rows.append(row)
    write(folder / 'dense_micro.json', dense_rows)
    write(folder / 'micro_summary.json', dict(cases=len(rows), completed=completed,
        all_micro_states_exact=all(x['hidden_exact'] and x['KV_exact'] for x in rows),
        kernel_union='NOT_MEASURED', SM_cycles='NOT_MEASURED', weight_read_bytes='NOT_MEASURED',
        allocated_GiB=torch.cuda.memory_allocated()/2**30, peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    engine.close()
    write(folder / 'status.json', dict(status='COMPLETE'))


if __name__ == '__main__':
    main()
