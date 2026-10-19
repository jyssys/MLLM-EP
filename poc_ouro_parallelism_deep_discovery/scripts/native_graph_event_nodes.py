"""Opt-in P1 event-node observation. No request latency extrapolation."""
import argparse
import os
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--expected-nodes', type=int, required=True)
    args = parser.parse_args()
    if os.environ.get('OURO_CUDA_GRAPH_EVENT_DIAGNOSTIC') != '1':
        raise RuntimeError('This diagnostic requires OURO_CUDA_GRAPH_EVENT_DIAGNOSTIC=1')
    mask = validate_mask()
    assert len(mask) == 1
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    configure_native_imports()
    output = Path(os.environ['LOOPED_KV_OUTPUT'])
    os.environ['VLLM_CACHE_ROOT'] = str(output / 'vllm_cache')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(output / 'inductor_cache')
    os.environ['TRITON_CACHE_DIR'] = str(output / 'triton_cache')
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
        result = llm.generate([{'prompt_token_ids': row['prompt_ids']['1024']}],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False)[0]
        ids = list(result.outputs[0].token_ids)
        assert ids == teacher
        return ids
    for _ in range(3):
        generate()
    armed = llm.collective_rpc('graph_events_arm', timeout=120,
                               args=(args.expected_nodes, MODELS[args.size]['layers']))
    llm.collective_rpc('start_parity_capture', timeout=120)
    ids = generate()
    capture = llm.collective_rpc('finish_parity_capture', timeout=120,
                                  args=('graph_event_capture', 1024))
    records = llm.collective_rpc('graph_events_finish', timeout=120)
    for rank in records:
        assert rank['total_observed_replays'] == 3, rank
    write(output / 'event_nodes.json', dict(size=args.size, P=1, R=4, physical_gpus=mask,
          context=1024, generated_tokens=4, output_ids=ids, capture=capture, armed=armed, ranks=records,
          evidence_class='GPU_MICROBENCH', measurement='LAST_REPLAY_COMPONENT_DIAGNOSTIC',
          quality='Controlled teacher output and capture only. Independent numerical comparison remains required.',
          request_total_ms=None, request_extrapolation='FORBIDDEN', candidate_speedup='NOT_RUN'))
    write(output / 'status.json', dict(success=True, component_diagnostic=True,
          natural_quality='NOT_RUN', candidate_speedup='NOT_RUN'))


if __name__ == '__main__':
    main()
