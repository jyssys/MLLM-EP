"""Run one standard continuous TP2 instance, not two TP2 replicas."""
import collections
import json
import os
import sys
import time

from .common import ROOT, MODEL, MODEL_REVISION, write


def main():
    config = json.loads(open(sys.argv[1]).read())
    folder = ROOT / 'raw' / config['name']
    write(folder / 'status.json', dict(status='RUNNING', pid=os.getpid(), time=time.time()))
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    if config.get('kv_layout'):
        os.environ['VLLM_KV_CACHE_LAYOUT'] = config['kv_layout']
    import torch
    from vllm import LLM, SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from ..dense_ar_slo_layer_pd.watch import GPUWatch
    torch.set_num_threads(1)
    data = json.loads((ROOT / 'data' / config['dataset']).read_text())[:config['count']]
    spans, samples, steps, refusals = [], [], [], []
    schedule = Scheduler.schedule
    def observed_schedule(self, *args, **kwargs):
        before = {rid: r.num_computed_tokens for rid, r in self.requests.items()}
        result = schedule(self, *args, **kwargs)
        now = time.monotonic()
        for rid, count in result.num_scheduled_tokens.items():
            request = self.requests.get(rid)
            if request is None:
                continue
            start = before.get(rid, 0)
            if start < len(request.prompt_token_ids):
                spans.append(dict(time=now, request_id=rid, start=start,
                    count=min(count, len(request.prompt_token_ids) - start),
                    prompt_tokens=len(request.prompt_token_ids),
                    batch_tokens=sum(result.num_scheduled_tokens.values()),
                    scheduled_requests=len(result.num_scheduled_tokens)))
        if not samples or now - samples[-1]['time'] >= .25:
            samples.append(dict(time=now, waiting=len(self.waiting), running=len(self.running),
                KV_usage=self.kv_cache_manager.usage, KV_free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                waiting_statuses=dict(collections.Counter(str(r.status) for r in self.waiting))))
        return result
    Scheduler.schedule = observed_schedule
    allocate = KVCacheManager.allocate_slots
    def observed_allocate(self, request, *args, **kwargs):
        value = allocate(self, request, *args, **kwargs)
        if value is None:
            refusals.append(dict(time=time.monotonic(), request_id=request.request_id,
                free_blocks=self.block_pool.get_num_free_blocks(), reason='KV_ALLOCATION_RETURNED_NONE'))
        return value
    KVCacheManager.allocate_slots = observed_allocate
    options = dict(model=str(MODEL), dtype='bfloat16', tensor_parallel_size=2,
        distributed_executor_backend='mp', worker_cls='poc_flashvep.duetserve_agg_decode.tp_worker.ExactTPWorker',
        max_model_len=config['length'] + config['cap'], max_num_seqs=config['D_slots'],
        max_num_batched_tokens=config['token_budget'], enable_chunked_prefill=True,
        long_prefill_token_threshold=config['chunk'], enable_prefix_caching=False,
        async_scheduling=config['ar_async_scheduling'], kv_cache_memory_bytes=config['fixed_KV_bytes'],
        seed=config['seed'], compilation_config=dict(mode=config['ar_compile_mode'], custom_ops=['all'],
            cudagraph_mode=config['ar_cudagraph_mode'], cudagraph_capture_sizes=config['explicit_cudagraph_capture_sizes']))
    monitor = GPUWatch(folder, config['gpus'], config['name'])
    monitor.start()
    llm = None
    scheduler = None
    initial_free = None
    complete = False
    rows = []
    def add(rid, row, cap, seed):
        llm.llm_engine.add_request(rid, {'prompt_token_ids': row['input_ids']['qwen8']},
            SamplingParams(temperature=0, max_tokens=cap, ignore_eos=True, seed=seed))
    try:
        llm = LLM(**options)
        write(folder / 'TP2_ready.json', dict(options=options, ranks=llm.collective_rpc('duetserve_report', timeout=30)))
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        initial_free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        for i in range(config.get('warmup_count', 2)):
            add('warm_' + str(i), data[i], 16, config['seed'] - 10000 + i)
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()
        first, events, arrivals = {}, {}, {}
        i = 0
        start = time.monotonic()
        write(folder / 'measurement_begin.json', dict(time=start))
        with (folder / 'offered_requests.jsonl').open('w') as offered:
            while i < len(data) or llm.llm_engine.has_unfinished_requests():
                now = time.monotonic()
                while i < len(data) and now >= start + config['arrival_offsets'][i]:
                    rid = 'q' + str(i)
                    arrivals[rid] = start + config['arrival_offsets'][i]
                    before = time.monotonic()
                    add(rid, data[i], config['request_caps'][i], config['seed'] + i)
                    offered.write(json.dumps(dict(sample_id=i, prompt_id=data[i]['prompt_id'], arrival=arrivals[rid],
                        coordinator_observed_arrival_time=now, actual_send_start=before,
                        actual_send_end=time.monotonic(), cap=config['request_caps'][i], seed=config['seed'] + i)) + '\n')
                    i += 1
                begin = time.monotonic()
                outputs = llm.llm_engine.step()
                end = time.monotonic()
                if outputs:
                    steps.append(dict(start=begin, end=end, outputs=len(outputs)))
                for result in outputs:
                    rid = result.request_id
                    index = int(rid[1:])
                    delivered = time.monotonic()
                    value = result.outputs[0]
                    first.setdefault(rid, delivered)
                    events.setdefault(rid, []).append(dict(time=delivered, count=len(value.token_ids), worker_time=delivered))
                    if not result.finished:
                        continue
                    cap = config['request_caps'][index]
                    assert len(value.token_ids) == cap
                    rows.append(dict(model='qwen8', arm='TP2', sample_id=index, prompt_id=data[index]['prompt_id'],
                        seed=config['seed'] + index, input_ids=data[index]['input_ids']['qwen8'], output_ids=list(value.token_ids),
                        input_tokens=len(data[index]['input_ids']['qwen8']), output_tokens=cap,
                        arrival=arrivals[rid], admission=arrivals[rid], first_token=first[rid], finish=delivered,
                        ttft_ms=1000 * (first[rid] - arrivals[rid]),
                        tpot_ms=1000 * (delivered - first[rid]) / (cap - 1), e2e_ms=1000 * (delivered - arrivals[rid]),
                        token_events=events[rid], model_revision=MODEL_REVISION, warmup=False,
                        evidence='MEASURED_STANDARD_VLLM_TP2',
                        timing_scope='Scheduled arrival to direct TP engine client delivery. No HTTP or interprocess output-forwarding coordinator.'))
                assert time.monotonic() - start < config['timeout']
                if not llm.llm_engine.has_unfinished_requests():
                    time.sleep(.0005)
        assert len(rows) == config['count']
        write(folder / 'window.json', dict(start=start, end=time.monotonic(), last_arrival=max(arrivals.values()), arrival_pattern='OPEN_LOOP_POISSON'))
        complete = True
    except BaseException as error:
        write(folder / 'status.json', dict(status='FAILED', error=repr(error), time=time.time()))
        raise
    finally:
        with (folder / 'requests.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        write(folder / 'native_TP2.json', dict(role='TP2', prompt_spans=spans, queue_samples=samples,
            host_step_spans=steps, KV_allocation_refusals=refusals,
            scope='Host engine/scheduler observations only. No GPU kernel or NCCL duration inferred.'))
        normal = False
        if llm is not None and scheduler is not None:
            pool = scheduler.kv_cache_manager.block_pool
            write(folder / 'KV_lifetime_TP2.json', dict(initial_free_blocks=initial_free,
                final_free_blocks=pool.get_num_free_blocks(), pending_requests=len(scheduler.requests),
                passed=pool.get_num_free_blocks() == initial_free and not scheduler.requests,
                scope='Normal no-prefix-cache scheduler retirement before the standard TP executor shutdown.'))
            llm.llm_engine.engine_core.shutdown()
            normal = True
        monitor.stop()
        write(folder / 'worker_exits.json', dict(passed=normal,
            scope='Normal standard engine/executor shutdown. Rank process outcomes remain in the runtime log.'))
        if complete:
            write(folder / 'status.json', dict(status='COMPLETE', time=time.time()))


if __name__ == '__main__':
    main()
