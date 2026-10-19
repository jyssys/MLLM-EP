"""Fixed-R4 native cohorts. This is a generic baseline, not a candidate method."""
import argparse
import dataclasses
import os
import random
import time

from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--tp', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    mask = validate_mask()
    assert len(mask) == args.tp
    protocol = read(args.protocol)
    assert protocol['status'] == 'FROZEN_BEFORE_RESULTS'
    output = Path(os.environ['LOOPED_KV_OUTPUT'])
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    for variable, directory in [('VLLM_CACHE_ROOT', 'vllm_cache'),
                                ('TORCHINDUCTOR_CACHE_DIR', 'inductor_cache'),
                                ('TRITON_CACHE_DIR', 'triton_cache')]:
        os.environ[variable] = str(output / directory)
    configure_native_imports()
    from vllm import LLM, SamplingParams

    short = args.size.replace('.', '').replace('B', '')
    source = ROOT / f'outputs/control_async_base{short}_P{args.tp}_v1/{args.size}_R4_TP{args.tp}'
    inputs = read(source / 'workload.json')['inputs']
    token = inputs[0]['teacher_ids'][0]
    assert token != 0 and 0 <= token < 49152
    cases = []
    for count in protocol['cohort_sizes']:
        for context in protocol['contexts']:
            rows = [inputs[index % len(inputs)] for index in range(count)]
            cases.append(dict(C=count, context=context,
                requests=[dict(source_request_id=row['id'], cohort_slot=index,
                               prompt_token_ids=row['prompt_ids'][str(context)])
                          for index, row in enumerate(rows)]))
    write(output / 'protocol.json', dict(protocol=protocol, protocol_sha256=sha(args.protocol),
          model={key: str(value) if isinstance(value, Path) else value
                 for key, value in MODELS[args.size].items()}, physical_gpus=mask, tp=args.tp,
          fixed_token_id=token, cases=cases,
          quality='NOT_RUN. Constant allowed-token output is a controlled hardware workload.'))
    config = read(source / 'engine_config.json')
    assert config['async_scheduling'] and not config['enable_chunked_prefill']
    assert not config['enable_expert_parallel'] and not config['enable_prefix_caching']
    assert config['hf_overrides']['total_ut_steps'] == 4
    config.update(max_num_seqs=protocol['max_num_seqs'],
                  max_num_batched_tokens=protocol['max_num_batched_tokens'],
                  kv_cache_memory_bytes=protocol['kv_cache_memory_bytes_per_rank'],
                  worker_extension_cls='poc_ouro_parallelism_deep_discovery.cohort_probe.CohortProbe')
    config['compilation_config']['cache_dir'] = str(output / 'compile_cache')
    write(output / 'engine_config.json', config)
    llm = LLM(**config)
    ownership = llm.collective_rpc('inspect_ownership', timeout=120)
    assert len(ownership) == args.tp
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership} == {UUIDS[GPUS.index(g)] for g in mask}
    assert all(r['cache_slots'] == 4 * MODELS[args.size]['layers'] for r in ownership)
    write(output / 'ownership.json', ownership)

    def generate(case, length):
        prompts = [dict(prompt_token_ids=r['prompt_token_ids']) for r in case['requests']]
        params = SamplingParams(temperature=0, max_tokens=length,
                                ignore_eos=True, allowed_token_ids=[token])
        start = time.perf_counter()
        results = llm.generate(prompts, params, use_tqdm=False)
        wall_ms = (time.perf_counter() - start) * 1000
        assert len(results) == case['C']
        records = []
        for request in results:
            assert list(request.outputs[0].token_ids) == [token] * length
            metrics = None if request.metrics is None else dataclasses.asdict(request.metrics)
            records.append(dict(request_id=request.request_id, metrics=metrics,
                                output_tokens=length, finish_reason=request.outputs[0].finish_reason))
        return dict(wall_ms=wall_ms, generated_tokens=case['C'] * length,
                    cohort_output_TPS=case['C'] * length * 1000 / wall_ms,
                    request_records=records, output_identity='ALL_FROZEN_CONSTANT_TOKEN_IDS')

    for case in cases:
        for _ in range(protocol['warmup_repeats']):
            generate(case, 16)
    order = []
    rng = random.Random(protocol['order_seed'])
    for repeat in range(protocol['repeats']):
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        for index in indices:
            case = cases[index]
            value = generate(case, protocol['generated_tokens'])
            record = dict(size=args.size, tp=args.tp, R=4, physical_gpus=mask,
                          C=case['C'], context=case['context'], repeat=repeat,
                          evidence_class='LIVE_E2E', quality='NOT_RUN',
                          metric='Submitted closed-cohort API wall, not B1 latency or steady-state throughput.',
                          **value)
            write(output / f"C{case['C']}_N{case['context']}_repeat{repeat}.json", record)
            order.append(dict(C=case['C'], context=case['context'], repeat=repeat))
            print('NATIVE_COHORT', args.size, args.tp, case['C'], case['context'], repeat, flush=True)
    write(output / 'trial_order.json', order)
    traces = []
    for case in cases:
        llm.collective_rpc('cohort_trace_start', timeout=120)
        value = generate(case, protocol['generated_tokens'])
        observed = llm.collective_rpc('cohort_trace_finish', timeout=120)
        traces.append(dict(C=case['C'], context=case['context'], output=value, ranks=observed,
                           timing='EXCLUDED_FROM_HEADLINE. Schedule-observation lane.'))
    write(output / 'schedule_trace.json', traces)
    write(output / 'status.json', dict(success=True, quality='NOT_RUN', candidate_speedup='NOT_RUN',
          controlled_work='Native fixed R4 with constant allowed-token output.',
          admission='C is the submitted cohort size. Native asynchronous admission remains unchanged.'))


if __name__ == '__main__':
    main()
