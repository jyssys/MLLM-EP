"""Run the frozen small audit only after the primary quality gate fails."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def wait_for(path, deadline):
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'Missing prerequisite: {path.name}. No GPU job was started.')
        time.sleep(10)
    return read(path)


def guarded(name, gpus, module, arguments=(), timeout=1800):
    command = [CPU, '-m', 'poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
               '--gpus', gpus, '--output', 'results/' + name, '--timeout', str(timeout),
               '--', PYTHON, '-m', 'poc_ouro_reasoning_fidelity_multigpu.scripts.' + module,
               *arguments]
    code = subprocess.call(command, cwd=PROJECT)
    if code:
        raise RuntimeError(f'{name} failed. Preserve its evidence before further GPU work.')


def main():
    deadline = time.monotonic() + 43200
    analysis = wait_for(ROOT/'results/quality_analysis_wait_status.json', deadline)
    assert analysis['success'], 'The full quality collection requires review.'
    wait_for(ROOT/'results/restart_history_gate.json', deadline)
    quality = read(ROOT/'results/quality_score/native_quality_r4.json')
    primary = quality['final_lanes']['aime24_sampling']
    fidelity = read(ROOT/'results/fidelity_summary.json')
    assert primary['cap'] == 24576 and primary['collection_complete']
    fraction = primary['natural_complete'] / primary['total']
    assert fraction < .9, 'Do not use this fallback when the primary completion gate passes.'
    assert fidelity['calibration_pass'] is False
    assert fidelity['reference_runtime_verdict'] == 'REFERENCE_RUNTIME_PARTIAL'
    decision = ROOT/'results/quality_decision.json'
    assert not decision.exists(), 'Preserve the first operational-quality decision.'
    write(decision, dict(quality='QUALITY_BASELINE_WEAK',
        correctness='REFERENCE_RUNTIME_PARTIAL', systems='PARTIAL_PROFILE_ONLY',
        primary_natural_termination_fraction=fraction,
        predeclared_natural_termination_target=.9,
        primary_cap=primary['cap'], primary_total=primary['total'],
        primary_correct=primary['correct'], original_state_gate_pass=False,
        broad_systems_authorized=False, small_exploratory_audit_authorized=True,
        method_authorized=False, quality_score_sha256=sha(ROOT/'results/quality_score/native_quality_r4.json'),
        reasons=['The primary sampled cohort misses the frozen natural-termination target at the largest declared cap.',
                 'Strict reference-state limits remain unmet. Functional JSD agreement is narrower evidence.',
                 'Public exact-match evaluation does not reproduce the private author protocol.'],
        scope='This decision does not declare the checkpoint defective or prove that loop-specific headroom is absent.'))
    for suffix in ['v1', 'v2']:
        guarded('fixed_logits_sampler_' + suffix, '4', 'run_fixed_logits_sampler', timeout=300)
    a = read(ROOT/'results/fixed_logits_sampler_v1/fixed_logits_sampler.json')
    b = read(ROOT/'results/fixed_logits_sampler_v2/fixed_logits_sampler.json')
    write(ROOT/'results/fixed_logits_sampler_comparison.json', dict(
        fresh_process_records_equal=a['records'] == b['records'],
        same_process_repeats_equal=[a['same_process_repeats_equal'], b['same_process_repeats_equal']],
        B1_B4_first_row_equal=[a['B1_B4_first_row_equal'], b['B1_B4_first_row_equal']],
        model_forward='NOT_RUN', actual_model_request_RNG_offsets='NOT_CAPTURED',
        scope='Fixed logits and the installed sampler only. This is not model-output restart parity.'))
    for tp, mask in [(1, '4'), (2, '4,5'), (4, '4,5,6,7')]:
        guarded(f'exploratory_B1_TP{tp}_v1', mask, 'run_exploratory_b1', ['--tp', str(tp)])
    subprocess.run([CPU, '-m', 'poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_exploratory_b1'],
                   cwd=PROJECT, check=True)
    write(ROOT/'results/bounded_followup_execution.json', dict(success=True,
        broad_systems='NOT_RUN', new_method='NOT_IMPLEMENTED',
        scope='Completed only the frozen weak-quality fallback and the fixed-logit sampler diagnostic.'))


if __name__ == '__main__':
    main()
