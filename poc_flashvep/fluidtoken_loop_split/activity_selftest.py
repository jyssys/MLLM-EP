"""Validate kernel-event coverage. This is not a model performance test."""
import csv
import ctypes
import time
from .common import *


def library():
    build = json.loads((ROOT / 'tools/activity_build.json').read_text())
    assert digest(Path(build['library'])) == build['library_SHA256']
    handle = ctypes.CDLL(build['library'])
    from .collector_mapping import record
    record(json.loads(Path(os.environ['ARC_CONFIG']).read_text())['name'], os.getpid())
    handle.activity_timestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    handle.activity_end.argtypes = [ctypes.c_char_p]
    for name in ('activity_dropped', 'activity_invalid', 'activity_count'):
        getattr(handle, name).restype = ctypes.c_uint64
    return handle


def clock(handle):
    timestamp = ctypes.c_uint64()
    before = time.monotonic()
    assert handle.activity_timestamp(ctypes.byref(timestamp)) == 0
    after = time.monotonic()
    return dict(device_clock_ns=timestamp.value, monotonic_s=(before + after) / 2,
                host_uncertainty_s=after - before)


def main():
    cfg = json.loads(Path(os.environ['ARC_CONFIG']).read_text())
    folder = ROOT / 'raw' / cfg['name']
    import torch
    torch.cuda.set_device(0)
    tensor = torch.ones((256, 256), device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            tensor.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tensor.add_(1)
    torch.cuda.synchronize()
    before = tensor[0, 0].item()
    handle = library()
    assert handle.activity_begin() == 0
    start = clock(handle)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    iterations = cfg.get('collector_test_iterations', 100)
    for _ in range(iterations):
        tensor.add_(1)
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    stop = clock(handle)
    target = folder / 'kernel_activity.csv'
    assert handle.activity_end(str(target).encode()) == 0
    rows = list(csv.DictReader(target.open()))
    intervals = [(int(row['start_ns']), int(row['end_ns'])) for row in rows]
    from ..looped_pd_headroom_recovery.profile_analysis import union
    duration_ms = union(intervals) / 1e6
    event_ms = begin.elapsed_time(end)
    exact = bool(torch.all(tensor == before + 2 * iterations).item())
    result = dict(evidence='GPU_INSTRUMENTATION_SELF_TEST_NOT_MODEL_RESULT',
        collector_build=json.loads((ROOT / 'tools/activity_build.json').read_text()),
        expected_kernel_invocations=2 * iterations, recorded_kernel_invocations=len(rows),
        graph_kernel_invocations=sum(int(row['graph_node']) != 0 for row in rows),
        dropped=handle.activity_dropped(), invalid=handle.activity_invalid(),
        kernel_union_ms=duration_ms, CUDA_event_enclosure_ms=event_ms,
        output_exact=exact, start_anchor=start, stop_anchor=stop,
        passed=len(rows) == 2 * iterations and sum(int(row['graph_node']) != 0 for row in rows) == iterations
               and handle.activity_dropped() == 0 and handle.activity_invalid() == 0
               and exact and duration_ms <= event_ms + .1)
    write(folder / 'activity_selftest.json', result)
    assert result['passed'], result
    write(folder / 'status.json', dict(status='COMPLETE'))


if __name__ == '__main__':
    main()
