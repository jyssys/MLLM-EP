"""Run CPU contracts without allocating a GPU or changing frozen evidence."""
import subprocess
from .common import *


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_autonomous_headroom.common import PYTHON
    modules=['poc_flashvep.adaptive_idle_loop_continuation.test_evidence',
        'poc_flashvep.adaptive_idle_loop_continuation.test_monitor',
        'poc_flashvep.native_loop_2d_oracle.test_dependency',
        'poc_flashvep.native_loop_2d_oracle.test_native_path',
        'poc_flashvep.native_loop_2d_oracle.test_staircase',
        'poc_flashvep.native_split_deep_validation.test_scheduler_contract',
        'poc_flashvep.native_split_deep_validation.test_exact_fingerprint',
        'poc_flashvep.native_split_deep_validation.test_handoff_receipts',
        'poc_flashvep.dynamic_loop_split_serving.test_lifetime_gate',
        'poc_flashvep.dynamic_loop_split_serving.test_metrics',
        'poc_flashvep.dynamic_loop_split_serving.test_work_accounting']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[str(PYTHON),'-m','unittest',*modules]
    with quiet_analysis('CPU_contract_regression'):
        result=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True,timeout=120)
    write(ROOT/'state/CPU_regression.json',dict(time=__import__('time').time(),command=command,
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        scope='CPU contracts only. No serving latency, GPU state, or adaptive-policy qualification.'))
    print(result.stdout+result.stderr)
    assert result.returncode==0


if __name__=='__main__':main()
