"""Derive fixed quality tables after complete scoring. Never authorize GPU work."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    marker=ROOT/'results/quality_collection_and_scoring_complete.json'
    execution=ROOT/'results/full_quality_execution.json'
    deadline=time.monotonic()+36000
    while not marker.exists():
        if execution.exists() and read(execution)['exit_code']:
            write(ROOT/'results/quality_analysis_wait_status.json',dict(success=False,
                reason='The native collection did not finish successfully. Preserve raw outputs and review them before any continuation.'))
            return
        if time.monotonic()>deadline:
            raise TimeoutError('Complete quality scoring did not arrive')
        time.sleep(10)
    assert read(marker)['success']
    subprocess.run([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_quality'],
                   cwd=PROJECT,check=True)
    write(ROOT/'results/quality_analysis_wait_status.json',dict(success=True,
        broad_systems_authorized=False,method_authorized=False,
        reason='Derived tables do not replace manual operational-quality and fidelity review.'))


if __name__=='__main__':
    main()
