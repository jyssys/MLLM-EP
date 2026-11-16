"""Measure the inherited exactness correction apart from serving latency."""
import fcntl
import os
import subprocess
import sys
import time

from .common import PYTHON, QUIET_LOCK, REPO, ROOT, initialize, resources, write, bind_previous


def worker_main():
    initialize()
    assert int(os.environ['DUETSERVE_DIAGNOSTIC_PARENT']) == os.getppid()
    result_name = os.environ.get('DUETSERVE_NORMALIZATION_RESULT', 'normalization_primitive_cost_v2')
    assert not (ROOT / 'analysis' / (result_name + '.json')).exists()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_BATCH_INVARIANT'] = '1'
    # The parent keeps the same open-file description locked until this
    # process exits. CUDA teardown can outlive this function's return.
    with os.fdopen(int(os.environ['DUETSERVE_DIAGNOSTIC_LEASE_FD']), 'a+', closefd=False) as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        owner.stop(list(range(8)))
        try:
            import torch
            from vllm import _custom_ops as ops
            bind_previous()
            from ..dense_ar_slo_layer_pd.phase_mask_norm import load
            module = load()
            results = []
            torch.manual_seed(2026111626)
            for rows, phase in ((1, 'DECODE'), (8, 'DECODE'), (16, 'DECODE'), (32, 'DECODE'), (512, 'PREFILL'), (8192, 'PREFILL')):
                x = torch.randn(rows, 4096, dtype=torch.bfloat16, device='cuda')
                residual = torch.randn_like(x)
                weight = torch.randn(4096, dtype=torch.bfloat16, device='cuda')
                mask = torch.full((rows,), int(phase == 'DECODE'), dtype=torch.int32, device='cuda')
                a, ar = x.clone(), residual.clone()
                b, br = x.clone(), residual.clone()
                ops.fused_add_rms_norm(a, ar, weight, 1e-6)
                module.masked_norm(b, br, weight, mask, 1e-6)
                exact = torch.equal(a, b) and torch.equal(ar, br)
                assert exact
                values = {}
                for label, operation in (
                    ('original', lambda: ops.fused_add_rms_norm(a, ar, weight, 1e-6)),
                    ('phase_correct', lambda: module.masked_norm(b, br, weight, mask, 1e-6)),
                ):
                    for _ in range(30):
                        operation()
                    pairs = []
                    for _ in range(100):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        operation()
                        end.record()
                        pairs.append((begin, end))
                    torch.cuda.synchronize()
                    samples = sorted(begin.elapsed_time(end) for begin, end in pairs)
                    values[label] = dict(mean_ms=sum(samples) / len(samples), median_ms=samples[50],
                                         p95_ms=samples[94], samples_ms=samples)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(100):
                            operation()
                    graph.replay()
                    pairs = []
                    for _ in range(30):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        graph.replay()
                        end.record()
                        pairs.append((begin, end))
                    torch.cuda.synchronize()
                    amortized = sorted(begin.elapsed_time(end) / 100 for begin, end in pairs)
                    values[label]['graph100_amortized_mean_ms'] = sum(amortized) / len(amortized)
                    values[label]['graph100_amortized_samples_ms'] = amortized
                results.append(dict(rows=rows, phase=phase, byte_equal=exact, timings=values))
            write(ROOT / 'analysis' / (result_name + '.json'), dict(time=time.time(), pid=os.getpid(), session=os.getsid(0),
                results=results, gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                evidence='MEASURED',
                scope='CUDA-event primitive enclosures in an isolated process. Graph100 amortizes a captured chain of100 normalization operations. Excludes phase-mask host preparation and H2D copy. This does not measure serving slowdown or kernel-union time.'))
        finally:
            # Do not start burners until this process releases its CUDA context.
            write(ROOT / 'state' / (result_name + '_finished.json'), dict(time=time.time(), pid=os.getpid()))


def main():
    if '--worker' in sys.argv:
        return worker_main()
    initialize()
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        env = dict(os.environ, DUETSERVE_DIAGNOSTIC_PARENT=str(os.getpid()),
                   DUETSERVE_DIAGNOSTIC_LEASE_FD=str(lease.fileno()))
        child = subprocess.Popen([str(PYTHON), '-m',
            'poc_flashvep.duetserve_agg_decode.normalization_cost', '--worker'],
            cwd=REPO, env=env, pass_fds=(lease.fileno(),), start_new_session=True)
        write(ROOT / 'state/normalization_primitive_v2_command.json', dict(pid=child.pid,
            parent_pid=os.getpid(), time=time.time(), GPUs=[0],
            scope='The CPU parent owns the lease until the CUDA child exits and disappears from NVML.'))
        try:
            code = child.wait(180)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            raise
        until = time.monotonic() + 15
        while any(row['pid'] == child.pid for row in resources().processes()) and time.monotonic() < until:
            time.sleep(.1)
        assert not any(row['pid'] == child.pid for row in resources().processes())
        # The child's CUDA context is gone before another experiment can hold
        # this lease. A measured result cannot authorize another process kill.
        if not resources().processes():
            resources().restore(list(range(8)))
        assert code == 0


if __name__ == '__main__':
    main()
