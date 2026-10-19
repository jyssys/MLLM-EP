"""Bounded P1 graph diagnostic. Never extrapolate the last replay to E2E."""
import argparse
import os
import time

from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, required=True)
    args = parser.parse_args()
    if any(os.environ.get(key) != '1' for key in
           ['OURO_CUDA_GRAPH_EVENT_DIAGNOSTIC', 'OURO_CUDA_GRAPH_TYPED_EVENT_DIAGNOSTIC']):
        raise RuntimeError('Both explicit event-diagnostic opt-ins are required')
    mask = validate_mask()
    assert len(mask) == 1
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    configure_native_imports()
    output = Path(os.environ['LOOPED_KV_OUTPUT'])
    for name, suffix in [('VLLM_CACHE_ROOT', 'vllm_cache'),
                         ('TORCHINDUCTOR_CACHE_DIR', 'inductor_cache'),
                         ('TRITON_CACHE_DIR', 'triton_cache')]:
        os.environ[name] = str(output / suffix)
    from vllm import LLM, SamplingParams
    short = args.size.replace('.', '').replace('B', '')
    source = ROOT / f'outputs/control_async_base{short}_P1_v1/{args.size}_R4_TP1'
    config = read(source / 'engine_config.json')
    assert config['async_scheduling']
    config['worker_extension_cls'] = 'poc_ouro_parallelism_deep_discovery.graph_event_probe.GraphEventProbe'
    config['compilation_config']['cache_dir'] = str(output / 'compile_cache')
    write(output / 'engine_config.json', config)
    row = read(source / 'workload.json')['inputs'][0]
    teacher = row['teacher_ids'][:4]
    llm = LLM(**config)
    ownership = llm.collective_rpc('inspect_ownership', timeout=120)
    assert {r['GPU_uuid'].removeprefix('GPU-') for r in ownership} == {UUIDS[GPUS.index(g)] for g in mask}
    assert all(r['cache_slots'] == 4 * MODELS[args.size]['layers'] for r in ownership)
    write(output / 'ownership.json', ownership)

    def generate():
        llm.collective_rpc('teacher', timeout=120, args=(teacher,))
        start = time.perf_counter_ns()
        value = llm.generate([{'prompt_token_ids': row['prompt_ids']['1024']}],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False)[0]
        elapsed = time.perf_counter_ns() - start
        assert list(value.outputs[0].token_ids) == teacher
        return dict(ids=teacher, request_wall_ns=elapsed)

    def diagnostic_request(label):
        llm.collective_rpc('external_intervals_start', timeout=120, args=(64,))
        llm.collective_rpc('start_parity_capture', timeout=120)
        request = generate()
        captures = llm.collective_rpc('finish_parity_capture', timeout=120, args=(label, 1024))
        intervals = llm.collective_rpc('external_intervals_finish', timeout=120)
        return dict(request=request, captures=captures, intervals=intervals,
                    scope='Observer lane only. Request wall includes diagnostic overhead.')

    for _ in range(3):
        generate()
    clean_before = [generate() for _ in range(3)]
    before = diagnostic_request('typed_event_before')
    expected = 48 * MODELS[args.size]['layers'] + 2
    armed = llm.collective_rpc('graph_events_arm', timeout=120,
                               args=(expected, MODELS[args.size]['layers']))
    observed = diagnostic_request('typed_event_observed')
    records = llm.collective_rpc('graph_events_finish', timeout=120)
    assert all(r['total_observed_replays'] == 3 for r in records)
    after = diagnostic_request('typed_event_after')
    clean_after = [generate() for _ in range(3)]
    write(output / 'typed_event_observation.json', dict(size=args.size, P=1, R=4,
        model=MODELS[args.size]['id'], revision=MODELS[args.size]['revision'], physical_gpus=mask,
        context=1024, generated_tokens=4, clean_before=clean_before, clean_after=clean_after,
        before=before, observed=observed, after=after, armed=armed, records=records,
        evidence_class='GPU_MICROBENCH', numerical_comparison='PENDING_INDEPENDENT_COMPARISON',
        request_extrapolation='FORBIDDEN', candidate_speedup='NOT_RUN', natural_quality='NOT_RUN',
        limitation='First observed replay includes graph mutation and reinstantiation. Exclude it from event-overhead estimates. '
                   'Cuts delimit kernel groups. They are not exact semantic recurrent-pass boundaries. '
                   'Preserved programmatic edges do not establish overlap duration or full serialization.'))
    write(output / 'status.json', dict(success=True, component_diagnostic=True,
          natural_quality='NOT_RUN', candidate_speedup='NOT_RUN'))


if __name__ == '__main__':
    main()
