"""Record the partial-reference interpretation before confirmation differences are read."""
import datetime
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    target=ROOT/'fidelity_interpretation_lock.json'
    assert not target.exists()
    assert not (ROOT/'results/fidelity_comparisons.json').exists()
    assert not (ROOT/'results/fidelity_summary.json').exists()
    limits=read(ROOT/'results/fidelity_tolerance_lock.json')
    diagnostic=read(ROOT/'results/calibration_diagnostic_summary_v2.json')
    assert diagnostic['calibration_only'] and diagnostic['tolerance_changed'] is False
    assert diagnostic['loaded_weight_identity']['all_equal']
    assert not all(r['calibration_pass'] for r in limits['limits'].values())
    write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_limit_sha256=sha(ROOT/'results/fidelity_tolerance_lock.json'),
        calibration_diagnostic_sha256=sha(ROOT/'results/calibration_diagnostic_summary_v2.json'),
        no_confirmation_numerical_differences_used=True,no_benchmark_score_used=True,
        state_gate='FAIL_UNDER_ORIGINAL_PLANNING_CEILINGS. Do not overwrite or relabel as PASS.',
        functional_distribution_limit=limits['limits']['next_token_JSD']['threshold'],
        limit_changed=False,
        reason='All loaded TP1 weights match. Same-config selected states repeat exactly. Norm-path changes alter the late state drift. First natural divergences occur at small or tied margins.',
        interpretation='At most REFERENCE_RUNTIME_PARTIAL, even if functional confirmation passes. No exact HF equivalence or model-bug resolution claim.',
        confirmation_action='Evaluate all frozen confirmation states with the original limits. Report failed state-space criteria separately from next-token JSD.',
        Q2_action='A small unmodified-native public-quality diagnostic is allowed only if functional confirmation and same-config reproducibility pass.',
        systems_action='No broad systems campaign until operational quality is credible. No new framework from a partial-reference result.',
        ablation_scope='The HF-operator norm arm is diagnostic only. It cannot replace the native benchmark or supply a speedup.'))
    print('Partial-reference interpretation frozen before confirmation analysis.')

if __name__=='__main__':
    main()
