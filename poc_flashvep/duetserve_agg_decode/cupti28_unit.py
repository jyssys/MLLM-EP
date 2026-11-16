"""Qualify the existing kernel-only collector with the CUDA12.9 CUPTI library."""
import csv
import ctypes
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from .common import PYTHON, QUIET_LOCK, REPO, ROOT, SOURCE, digest, resources, write

CUPTI = Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/nvidia/cuda_cupti')
CUDA_HEADERS = CUPTI.parent / 'cuda_runtime/include'
CUDA_CRT_FALLBACK = Path('/usr/local/cuda-12.8/include')


def stem():
    variant = os.environ.get('DUETSERVE_CUPTI28_VARIANT', 'v1')
    assert variant in ('v1', 'v2', 'v3')
    return 'cupti28' + ('' if variant == 'v1' else '_' + variant)


def stamp(lib):
    result = ctypes.c_uint64()
    before = time.monotonic_ns()
    assert lib.activity_timestamp(ctypes.byref(result)) == 0
    after = time.monotonic_ns()
    return dict(cupti_ns=result.value, host_lower_ns=before, host_upper_ns=after)


def child():
    build = json.loads((ROOT / 'tools' / (stem() + '_build.json')).read_text())
    lib = ctypes.CDLL(build['library'])
    lib.activity_begin.argtypes = [ctypes.c_size_t, ctypes.c_int]
    lib.activity_timestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.activity_end.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64]
    for key in ('dropped', 'invalid', 'count', 'all_kernels'):
        getattr(lib, 'activity_' + key).restype = ctypes.c_uint64
    version = ctypes.c_uint()
    assert lib.cuptiGetVersion(ctypes.byref(version)) == 0
    assert version.value == 28
    assert lib.activity_begin(512 * 1024**2, 0) == 0
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    a = torch.ones(1024, dtype=torch.bfloat16, device='cuda')
    b = torch.empty_like(a)
    for _ in range(10):
        torch.add(a, 1, out=b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.add(a, 1, out=b)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    begin = stamp(lib)
    for _ in range(100):
        torch.add(a, 1, out=b)
        graph.replay()
    torch.cuda.synchronize()
    end = stamp(lib)
    path = ROOT / 'tools' / (stem() + '_witness.csv')
    code = lib.activity_end(str(path).encode(), begin['cupti_ns'], end['cupti_ns'])
    records = list(csv.DictReader(path.open()))
    kernels = [r for r in records if r['kind'] == 'kernel']
    counters = {key: getattr(lib, 'activity_' + key)() for key in ('dropped', 'invalid', 'count', 'all_kernels')}
    graph_count = sum(int(r['graph_node']) != 0 for r in kernels)
    passed = code == 0 and len(kernels) == 200 and graph_count == 100 and not counters['dropped'] and not counters['invalid']
    write(ROOT / 'analysis' / (stem() + '_witness_gate.json'), dict(passed=passed, CUPTI_API_version=version.value,
        code=code, kernels=len(kernels), expected_kernels=200, graph_kernels=graph_count,
        expected_graph_kernels=100, counters=counters, begin=begin, end=end,
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid), pid=os.getpid(),
        CUPTI_mappings=[line.strip() for line in Path('/proc/self/maps').read_text().splitlines() if 'libcupti' in line],
        evidence='CORRECTNESS_ONLY',
        scope='Collector coverage witness with100 eager and100 graph-launched kernels. No serving workload, speedup, idle-capacity, or overhead claim follows from this witness.'))
    assert passed


def main():
    if '--child' in sys.argv:
        return child()
    if '--variant' in sys.argv:
        os.environ['DUETSERVE_CUPTI28_VARIANT'] = sys.argv[sys.argv.index('--variant') + 1]
    assert (ROOT / 'state/user_primary_scope_revision_v2.json').exists()
    gate = ROOT / 'analysis' / (stem() + '_witness_gate.json')
    assert not gate.exists()
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        owner.stop(list(range(8)))
        process = None
        try:
            source = SOURCE / 'cupti28_preallocated.cc'
            assert digest(source) == digest(REPO / 'poc_flashvep/prefill_heavy_decode_safe/preallocated_activity.cc')
            library = ROOT / 'tools' / (stem() + '_kernel_only.so')
            library.parent.mkdir(exist_ok=True)
            command = ['g++', '-O2', '-std=c++17', '-fPIC', '-shared', '-pthread', str(source),
                '-I' + str(CUPTI / 'include'), '-I' + str(CUDA_HEADERS),
                '-I' + str(CUDA_CRT_FALLBACK),
                '-L' + str(CUPTI / 'lib'), '-Wl,-rpath,' + str(CUPTI / 'lib'),
                '-l:libcupti.so.12', '-o', str(library)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            write(ROOT / 'tools' / (stem() + '_build_log.json'), dict(command=command, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr))
            assert result.returncode == 0
            write(ROOT / 'tools' / (stem() + '_build.json'), dict(source=str(source), source_SHA256=digest(source),
                reused_source='poc_flashvep/prefill_heavy_decode_safe/preallocated_activity.cc',
                library=str(library), library_SHA256=digest(library), command=command,
                CUPTI_library=str(CUPTI / 'lib/libcupti.so.12'),
                CUPTI_library_SHA256=digest(CUPTI / 'lib/libcupti.so.12'),
                CUDA_header_SHA256=digest(CUDA_HEADERS / 'cuda.h'),
                CRT_fallback_header=str(CUDA_CRT_FALLBACK / 'crt/host_defines.h'),
                CRT_fallback_header_SHA256=digest(CUDA_CRT_FALLBACK / 'crt/host_defines.h'),
                scope='Verbatim existing fixed-buffer concurrent-kernel collector. Link to the installed CUDA12.9 CUPTI API28 instead of the older API26 library. No installed runtime is changed.'))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1')
            with (ROOT / 'logs' / (stem() + '_witness.log')).open('w') as stream:
                process = subprocess.Popen([str(PYTHON), '-u', '-m',
                    'poc_flashvep.duetserve_agg_decode.cupti28_unit', '--child'], cwd=REPO, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                write(ROOT / 'state' / (stem() + '_witness_command.json'), dict(pid=process.pid, time=time.time(),
                    scope='The CPU parent retains the quiet-lane lease until this GPU child exits.'))
                code = process.wait(180)
            assert code == 0, 'Collector witness failed'
        except BaseException as error:
            if not gate.exists():
                write(gate, dict(passed=False, error=repr(error),
                    evidence='FAILED', scope='Do not launch a serving capture after a failed build or collector witness.'))
            print('CUPTI28_WITNESS_FAILED', repr(error), flush=True)
        finally:
            if process is not None and process.poll() is None:
                assert os.getsid(process.pid) == process.pid
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            until = time.monotonic() + 15
            while process is not None and any(r['pid'] == process.pid for r in owner.processes()) and time.monotonic() < until:
                time.sleep(.1)
            if process is not None:
                assert not any(r['pid'] == process.pid for r in owner.processes())
            if not owner.processes():
                owner.restore(list(range(8)))


if __name__ == '__main__':
    main()
