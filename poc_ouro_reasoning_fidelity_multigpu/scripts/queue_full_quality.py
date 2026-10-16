"""Run only the preregistered public quality collection after its diagnostic gates."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    marker=ROOT/'results/prompt_diagnostic_v1/guard_status.json'
    deadline=time.monotonic()+4200
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Prompt diagnostic did not finish')
        time.sleep(5)
    assert read(marker)['success'],'Do not bypass a failed diagnostic'
    fidelity=read(ROOT/'results/fidelity_summary.json')
    assert fidelity['reference_runtime_verdict']=='REFERENCE_RUNTIME_PARTIAL'
    assert fidelity['functional_confirmation_pass'] and fidelity['same_config_numerical_bitwise']
    assert fidelity['same_config_short_repeats_pass']
    assert read(ROOT/'results/prompt_diagnostic_v1/diagnostic_same_config_long_repro.json')['PASS']
    assert read(ROOT/'results/prompt_diagnostic_v1/effective_stop_contract.json')['renderer_eos_token_id']==2
    write(ROOT/'results/full_quality_admission.json',dict(admitted=True,
        protocol_sha256=sha(ROOT/'protocol_lock.json'),
        interpretation_sha256=sha(ROOT/'quality_characterization_lock.json'),
        reference_runtime_status='REFERENCE_RUNTIME_PARTIAL',original_state_gate_pass=False,
        scope='Public quality characterization only. No broad systems or method promotion.'))
    code=subprocess.call([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
        '--gpus','4,5,6,7','--output','results/quality_v1','--timeout','28800','--',
        PYTHON,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_quality'],cwd=PROJECT)
    write(ROOT/'results/full_quality_execution.json',dict(exit_code=code))
    if code:
        raise RuntimeError('Preserve partial quality outputs and the failed guard. Do not score them as a complete collection.')
    subprocess.run([CPU,'-m','poc_looped_exact_depth_chunk_pipeline.scripts.score_public_reasoning',
        '--source',str(ROOT/'results/quality_v1'),'--output',str(ROOT/'results/quality_score')],cwd=PROJECT,check=True)
    write(ROOT/'results/quality_collection_and_scoring_complete.json',dict(success=True,
        systems_authorized=False,reason='Operational quality and fidelity limits need explicit review.'))

if __name__=='__main__':
    main()
