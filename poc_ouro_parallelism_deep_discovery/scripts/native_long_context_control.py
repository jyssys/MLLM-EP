"""Frozen C1 context scaling. This is a native baseline, not a new mechanism."""
import argparse
import dataclasses
import math
import os
import random
import time

from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--tp', type=int, choices=[1, 2, 4], required=True)
    args = parser.parse_args()
    mask = validate_mask()
    assert len(mask) == args.tp
    output = Path(os.environ['LOOPED_KV_OUTPUT'])
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    for variable, suffix in [('VLLM_CACHE_ROOT', 'vllm_cache'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor_cache'),
                             ('TRITON_CACHE_DIR', 'triton_cache')]:
        os.environ[variable] = str(output / suffix)
    configure_native_imports()
    from vllm import LLM, SamplingParams
    short = args.size.replace('.', '').replace('B', '')
    source = ROOT / f'outputs/control_async_base{short}_P{args.tp}_v1/{args.size}_R4_TP{args.tp}'
    inputs = read(source / 'workload.json')['inputs']
    protocol = read(ROOT / 'artifacts/native_long_context_protocol_v1.json')
    assert protocol['status'] == 'FROZEN_BEFORE_RESULTS'
    assert protocol['max_model_len'] == protocol['max_num_batched_tokens'] == 32896
    config = read(source / 'engine_config.json')
    assert config['hf_overrides']['total_ut_steps'] == 4 and config['async_scheduling']
    assert not config['enable_chunked_prefill'] and not config['enable_prefix_caching']
    assert not config['enable_expert_parallel']
    config.update(max_model_len=protocol['max_model_len'], max_num_seqs=1,
                  max_num_batched_tokens=protocol['max_num_batched_tokens'],
                  kv_cache_memory_bytes=protocol['kv_pool_aggregate_bytes'] // args.tp,
                  worker_extension_cls='poc_ouro_parallelism_deep_discovery.long_context_probe.LongContextDiagnosticProbe')
    config['compilation_config'].update(cache_dir=str(output / 'compile_cache'), cudagraph_capture_sizes=[1])
    token = inputs[0]['teacher_ids'][0]
    model_config = read(MODELS[args.size]['path'] / 'config.json')
    assert token != model_config['eos_token_id'] and model_config['max_position_embeddings'] == 65536
    cases = []
    for row in inputs:
        base = row['prompt_ids']['8192']
        for context in protocol['contexts']:
            ids = (base * math.ceil(context / len(base)))[:context]
            assert len(ids) == context
            cases.append(dict(source_request_id=row['id'], context=context, prompt_token_ids=ids))
    write(output / 'frozen_plan.json', dict(protocol=protocol,
        protocol_sha256=sha(ROOT / 'artifacts/native_long_context_protocol_v1.json'),
        model_revision=MODELS[args.size]['revision'], size=args.size, TP=args.tp, R=4,
        physical_gpus=mask, fixed_token_id=token, engine_config=config, cases=cases,
        source_workload_sha256=sha(source / 'workload.json'), source_config_sha256=sha(source / 'engine_config.json')))
    write(output / 'engine_config.json', config)
    llm = LLM(**config)
    ownership = llm.collective_rpc('inspect_ownership', timeout=120)
    assert len(ownership) == args.tp
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership} == {UUIDS[GPUS.index(g)] for g in mask}
    assert all(r['cache_slots'] == 4 * MODELS[args.size]['layers'] for r in ownership)
    write(output / 'ownership.json', ownership)

    def generate(case, length):
        start = time.perf_counter_ns()
        result = llm.generate([{'prompt_token_ids': case['prompt_token_ids']}],
            SamplingParams(temperature=0, max_tokens=length, ignore_eos=True, allowed_token_ids=[token]),
            use_tqdm=False)[0]
        elapsed = time.perf_counter_ns() - start
        assert list(result.outputs[0].token_ids) == [token] * length
        metrics = None if result.metrics is None else dataclasses.asdict(result.metrics)
        return dict(wall_ms=elapsed/1e6, generated_tokens=length,
                    output_TPS=length*1e9/elapsed, metrics=metrics)

    for case in cases:
        for _ in range(protocol['warmup_repeats']):
            generate(case, 4)
    rng = random.Random(protocol['order_seed'])
    for repeat in range(protocol['repeats']):
        order = list(range(len(cases)))
        rng.shuffle(order)
        for index in order:
            case = cases[index]
            value = generate(case, protocol['generated_tokens'])
            write(output / f'case{index}_repeat{repeat}.json', dict(value,
                size=args.size, TP=args.tp, R=4, C=1, context=case['context'],
                source_request_id=case['source_request_id'], case=index, repeat=repeat,
                evidence_class='LIVE_E2E', work='Constant-token controlled native work. Not natural quality.'))
            print('LONG_CONTEXT', args.size, args.tp, case['context'], repeat, index, flush=True)
    write(output / 'memory_after.json', llm.collective_rpc('inspect_memory', timeout=120))
    # Numerical capture is separate from every clean timing interval.
    case = next(c for c in cases if c['context'] == 32768)
    llm.collective_rpc('start_parity_capture', timeout=120)
    generate(case, 4)
    capture = llm.collective_rpc('finish_parity_capture', timeout=120,
                                  args=('long_context_numerical', 32768))
    write(output / 'numerical_capture.json', dict(capture=capture,
        scope='Post-timing logits, hidden, and sampled KV. Finite-value analysis pending. Not natural quality.'))
    write(output / 'status.json', dict(success=True, clean_trials=len(cases)*protocol['repeats'],
        numerical_finiteness='PENDING_INDEPENDENT_ANALYSIS', candidate_speedup='NOT_RUN', natural_quality='NOT_RUN'))


if __name__ == '__main__':
    main()
