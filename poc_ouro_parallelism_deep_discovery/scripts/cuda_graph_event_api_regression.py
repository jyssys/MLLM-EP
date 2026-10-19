"""Constant-tensor CUDA API regression. The coordinator must authorize GPU execution."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-cuda', action='store_true')
    args = parser.parse_args()
    if not args.run_cuda:
        raise SystemExit('GPU regression NOT_RUN. Use --run-cuda only inside the campaign GPU guard.')
    from poc_ouro_parallelism_deep_discovery.contract import GPUS, UUIDS, validate_mask, write
    from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
    mask = validate_mask()
    if len(mask) != 1:
        raise ValueError('The constant-tensor regression requires exactly one permitted GPU')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    configure_native_imports()
    import torch
    from poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes import Cudart, EventGraph, chain_order
    output = Path(os.environ['LOOPED_KV_OUTPUT'])
    uuid = str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert uuid == UUIDS[GPUS.index(mask[0])], 'The physical GPU UUID differs from the permitted mask'
    x = torch.full((256,), 2, device='cuda', dtype=torch.bfloat16)
    y, z, out = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    def work():
        torch.add(x, 2, out=y)
        torch.mul(y, 3, out=z)
        out.copy_(z)
    for _ in range(3):
        work()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        work()
    graph.instantiate()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(out, 12))
    runtime = Cudart()
    nodes, edges, kinds = runtime.topology(graph.raw_cuda_graph())
    order = chain_order(nodes, edges, kinds)
    if len(order) < 2:
        raise RuntimeError('The fixture needs at least two graph nodes')
    runtime.dot(graph.raw_cuda_graph(), output / 'constant_source.dot')
    timer = EventGraph(runtime, graph, [0, 1, len(order)])
    try:
        for value in (5, 7, 11):
            x.fill_(value)
            timer.launch(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            assert torch.equal(out, torch.full_like(out, (value + 2) * 3))
        record = timer.last_replay()
        assert record['observed_replays'] == record['timed_replay_ordinal'] == 3
        assert record['request_total_ms'] is None
    finally:
        timer.close()
    restored = runtime.topology(graph.raw_cuda_graph())
    assert (set(restored[0]), set(restored[1]), restored[2]) == (set(nodes), set(edges), kinds)
    x.fill_(13)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(out, 45))
    write(output / 'api_regression.json', dict(success=True, numerical_equal=True,
          topology_restored=True, original_torch_replay_used=True, runtime_version=runtime.version,
          runtime_library=runtime.path, physical_gpus=mask, GPU_uuid=uuid, record=record, model_experiment=False,
          evidence_class='GPU_MICROBENCH', scope='Constant tensors only. Not model or request timing.'))


if __name__ == '__main__':
    main()
