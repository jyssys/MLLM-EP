"""Do not compete with the fixed native quality engine."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    marker=ROOT/'results/quality_v1/guard_status.json'
    deadline=time.monotonic()+28800
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Native collection did not release the assigned GPUs')
        time.sleep(10)
    assert read(marker)['success'], 'Review a failed native collection before further GPU work'
    code=subprocess.call([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
        '--gpus','4','--output','results/hf_prompt_corroboration_v1','--timeout','7200','--',
        CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_hf_prompt_corroboration'],cwd=PROJECT)
    write(ROOT/'results/hf_prompt_corroboration_execution.json',dict(exit_code=code))
    if code:
        raise RuntimeError('Preserve the reference failure. Do not substitute another prompt or question.')


if __name__=='__main__':
    main()
