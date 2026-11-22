"""Lease an idle approved pair for one non-performance FP32 diagnostic."""
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
import zipfile
from .common import ROOT, REPO, PYTHON, resources, preserve, write, dispatch_allowed, digest
from .freeze_capsule import freeze


def main():
    preserve()
    capsule, sha = freeze('fp32_original_native_reference_v2_20261121')
    folder = ROOT / 'references/fp32_original_native_q0_v2'
    assert not folder.exists(), 'Retain every failed diagnostic. Use a new version for a retry.'
    pair = [6, 7]
    with (ROOT / 'state/pair_6_7.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        assert dispatch_allowed()
        guardian = json.loads((ROOT / 'state/guardian.json').read_text())
        assert guardian['status'] == 'ACTIVE'
        folder.mkdir()
        shutil.copyfile(ROOT / 'references/source_capsules' / capsule, folder / 'source.zip')
        source = folder / 'worker_source'
        with zipfile.ZipFile(folder / 'source.zip') as archive:
            archive.extractall(source)
        owner = resources()
        owner.stop(pair)
        owner.restore([7])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', PYTHONPATH=str(source),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [str(PYTHON), '-u', '-m', 'poc_flashvep.dense_ar_longbench400_pd.fp32_reference_worker', str(folder)]
        child = None
        state = ROOT / 'state/pair_lease_6_7.json'
        try:
            with (folder / 'worker.log').open('w') as log:
                child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, preexec_fn=lambda: os.sched_setaffinity(0, set(range(120, 160))))
                write(state, dict(name='DIAGNOSTIC_FP32_OriginalNative_q0_v2', pid=child.pid,
                    GPUs=pair, active_compute_GPU=6, status='RUNNING', time=time.time()))
                write(folder / 'command.json', dict(command=command, pid=child.pid, GPU=6,
                    source_SHA256=sha, evidence='DIAGNOSTIC', no_performance_claim=True))
                code = child.wait(timeout=900)
                write(folder / 'process_result.json', dict(returncode=code, passed=code == 0,
                    source_SHA256=sha, evidence='DIAGNOSTIC'))
                assert code == 0, 'Retain the high-precision diagnostic failure'
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write(state, dict(name='DIAGNOSTIC_FP32_OriginalNative_q0_v2', GPUs=pair,
                status='RELEASED', time=time.time(), complete=(folder / 'result.json').exists()))
            preserve()
            assert not [r for r in owner.processes() if r['gpu'] == 6]
            owner.restore([6])


if __name__ == '__main__':
    main()
