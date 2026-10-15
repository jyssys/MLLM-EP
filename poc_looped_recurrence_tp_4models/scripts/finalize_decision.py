"""Apply the frozen numerical gate before interpreting timing contrasts."""
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read


def main():
    out = ROOT / 'results'
    primary = read(out / 'parity_and_kv_validation.json')
    natural = read(out / 'natural_native_tied_parity.json')
    audit = read(out / 'natural_discrepancy_audit.json')
    selected = audit['selected_same_prefix_fields']
    primary_pass = len(primary) == 126 and all(r['bitwise_equal'] for r in primary)
    natural_pass = audit['same_policy_natural_token_parity']
    selected_pass = all(r['bitwise_equal'] for r in selected)
    # A later successful probe does not erase a prior failed correctness check.
    retained_pass = all(r['same_token_IDs'] for r in natural)
    complete_control_pass = primary_pass and natural_pass and selected_pass and retained_pass
    verdict = 'NO_LOOP_SPECIFIC_GAP' if complete_control_pass else 'CONTROL_BLOCKED'
    write(out / 'verdict.json', dict(
        verdict=verdict,
        scientific_promotion='NO_LOOP_SPECIFIC_GO',
        primary_configurations_measured=24,
        primary_clean_timing_status='VALID_IN_THE_DECLARED_BOUNDED_SCOPE',
        blocked_scope='Unreconciled retained-graph natural parity. Primary selected probes and ordinary-graph follow-up pass.',
        all_four_families_fully_validated=complete_control_pass,
        primary_selected_bitwise_fields=dict(passed=sum(r['bitwise_equal'] for r in primary), total=len(primary)),
        retained_graph_natural_comparisons=dict(passed=sum(r['same_token_IDs'] for r in natural), total=len(natural),
            scope='Technical repeats across four fixed R/TP conditions and three prompts. Not independent requests.'),
        ordinary_graph_same_prefix_fields=dict(passed=sum(r['bitwise_equal'] for r in selected), total=len(selected)),
        ordinary_graph_natural_parity=natural_pass,
        tolerance=PARITY,
        removable_recurrence_specific_ms='NOT_IDENTIFIED',
        measured_new_method_E2E_gain='NOT_RUN',
        benchmark_quality='NOT_RUN',
        fresh_24_request_confirmation='NOT_TRIGGERED. No validated recurrence-specific physical mechanism.',
        next_action='Do not build a loop-aware framework. Resolve the numerical control before any positive causal claim.',
        limitation='The primary native scaling matrix remains measured. A blocked tied control cannot prove an absence of every recurrence effect.'))
    print(verdict)


if __name__ == '__main__':
    main()
