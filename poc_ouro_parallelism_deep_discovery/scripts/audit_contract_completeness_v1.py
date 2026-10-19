"""Export a read-only dossier audit. Field presence is not semantic completion."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_parallelism_deep_discovery_spec.md')
QUESTIONS = {
    'Q01': ([1, 2, 3], 'What mechanism uses which sourced Ouro property?'),
    'Q02': ([4, 17], 'Which models, modes, stages, workloads, and GPU counts apply?'),
    'Q03': ([5, 18], 'Which matched baseline, removed cost, and counterfactual apply?'),
    'Q04': ([6, 8], 'Which operation changes, and when do its actual inputs exist?'),
    'Q05': ([7, 10, 11], 'Who owns each state, message, version, and lifetime?'),
    'Q06': ([9], 'Which exactness guarantee and quality test apply?'),
    'Q07': ([12], 'Which complete resource-constrained cost must improve?'),
    'Q08': ([13, 14], 'Which nearest priors and previous failures constrain novelty?'),
    'Q09': ([15, 16], 'Which falsifier, observed result, and uncertainty apply?'),
    'Q10': ([19, 20], 'Which disposition, implementation limit, and next action apply?'),
}

# These are independently reviewed requirements. They are not test results.
DETAILS = {
 'C01': ('Published exact predecessor row plus resident operator weights.', 'A concrete pooled owner map and full six-state ledger remain missing.', 'A ready trace and benefit beyond generic aliased batching remain untested.'),
 'C02': ('Completed normalized post-attention MLP input.', 'The tested source/remote-worker component has explicit owners and both transfers.', 'Do not extend the component loss to every pooled architecture or claim full-generation parity.'),
 'C03': ('Actual row input, depth address, and completed adaptive terminal-depth decision.', 'Symbolic versions exist. Compact allocation, publication, and reclamation are not implemented.', 'Fixed R4 has no terminal-copy saving. Distributed ownership is not established.'),
 'C04': ('Actual disjoint features or complete projection contributions before nonlinear consumers.', 'No concrete removed-collective layout with native BF16 boundaries survives the screen.', 'A valid generic rewrite is not a measured removed message.'),
 'C05': ('Causal guesses or one exact warm-up, with matching immutable depth history.', 'Current-token KV needs private versions and complete finalization.', 'Serial diagnostic recovery is not physical parallel service or adaptive skipping.'),
 'C06': ('A genuinely ready base contribution and the later actual fresh delta.', 'Dense delta computation, merge layout, and BF16 cast boundaries remain charged.', 'A smaller norm is not a certificate for cheaper exact computation.'),
 'C07': ('Known draft IDs, exact ancestor versions, and matching depth history.', 'The proposed ownership/message ABI is explicit. Distributed commit and original-gate verification remain unvalidated.', 'P1 v2 smoke failed. No successful eager service time, four-owner makespan, or compiled comparison exists.'),
 'C08': ('Both named BF16 versions and the receiver base.', 'No complete codec/receiver-base lifetime or compressed serving protocol ran.', 'Small-message primitive cost is scoped. It does not reject all large-payload representations.'),
 'C09': ('The actual selected normalized hidden state, or a valid bound on its unfinished coordinates.', 'A concrete implementation-qualified BF16 certificate and owner/reduction map remain missing.', 'Greedy certification does not imply a sampling guarantee.'),
 'C10': ('Committed original normalized history inputs at each depth and layer.', 'Concrete checkpoint owners, publication, scratch, and restoration schedule remain missing.', 'Avoided local capacity or real transfer must exist before rematerialization can save it.'),
 'C11': ('The actual complete linear input and resident encoded weights.', 'A coding matrix, rank assignment, decoder, and BF16 numerical contract remain missing.', 'A real removable native tail is unmeasured. No artificial straggler may supply it.'),
 'C12': ('Committed actual K/V for updates, then the actual query for contraction.', 'No concrete summary representation, feature dimension, or distributed layout is established.', 'Approximation error, actual attention cost, and free-running quality remain unknown.'),
 'C13': ('Actual loop-normalized hidden state and baseline-rounded hazards.', 'The packet/lifetime contract is explicit. No GPU overlap implementation ran.', 'Adaptive-only baseline opportunity. Fixed R4 has no gate wait to remove.'),
 'C14': ('Known prompt IDs, actual predecessor hidden, and same-depth causal prefix KV.', 'A tied-aware generic cyclic control matches physical owners, aliases, messages, and depth KV.', 'No cyclic Ouro GPU timing. Generic trace identity is not a PP-versus-TP performance theorem.'),
 'C15': ('Actual historical normalized inputs and current position-correct query/weights.', 'Variant-specific state and message equations exist. BF16/model parity remains untested.', 'Card retained-K generic status and ledger direct-contraction cost status have different scope.'),
 'C16': ('One actual current row per independent request with its preceding gate decision.', 'A complete cross-owner attention plan and online trace are absent.', 'The CPU generic union equals the candidate union. No extra same-token frontier follows.'),
 'C17': ('Final coordinate tiles from their actual producer, not unreduced guessed features.', 'Fresh tile/sum scratch is described. No distributed layout or preserved-BF16 implementation exists.', 'The simple cast rewrite fails a constructed case. No model-quality failure follows.'),
 'C18': ('The complete actual read set, state versions, and exact equality certificate.', 'Attention read-state equality and all replayed side effects lack a runtime protocol.', 'Zero bounded whole-MLP-row hits is not a universal no-hit theorem.'),
 'C19': ('Original immutable KV words, every base/residual page, then the actual query.', 'Base retention, eviction, decoder scratch, and adaptive alias ownership remain unimplemented.', 'Key-only coding is not K+V traffic or latency. The card predates the negative read-demand screen.'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_fields(text):
    sections = list(re.finditer(r'^## (\d+)\. ([^\n]+)\n', text, re.M))
    if sections:
        return {int(m[1]): {'heading': m[2], 'text': text[m.end():sections[i+1].start() if i+1 < len(sections) else len(text)].strip()}
                for i, m in enumerate(sections)}
    return {int(m[1]): {'heading': m[2].strip(), 'text': m[3].strip()}
            for m in re.finditer(r'^\| (\d+)\. ([^|]+) \| (.*) \|$', text, re.M)}


def build():
    inputs = {}
    def read(path):
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        b = p.read_bytes()
        inputs[str(path)] = {'sha256': digest(b), 'bytes': len(b)}
        return b.decode()
    spec = read(AUTHORITATIVE)
    snapshot = read('reports/spec_snapshot.md')
    old = json.loads(read('artifacts/interim_candidate_evidence_matrix_v1.json'))
    old_rows = {r['id']: r for r in old['rows']}
    statuses = {}
    for i, row in enumerate(map(json.loads, read('artifacts/candidate_status.jsonl').splitlines())):
        key = (row.get('updated_utc', ''), i)
        if row['id'] not in statuses or key > statuses[row['id']][0]:
            statuses[row['id']] = (key, row)
    rows = []
    for path in sorted((ROOT / 'candidates').glob('C*.md')):
        ident = re.match(r'C\d+', path.name)[0]
        relative = str(path.relative_to(ROOT))
        fields = parse_fields(read(relative))
        assert set(fields) >= set(range(1, 21)), ident
        review = old_rows[ident]['reviewed_synthesis']
        for source in review['sources']:
            read(source)
        earliest, ownership, limit = DETAILS[ident]
        scope = dict(review['observed_scope'])
        if ident == 'C07':
            scope.update({'P1_v2_smoke': 'FAILED_RUNNER_AND_GUARD_PARITY_UNRESOLVED',
                          'P2': 'No candidate component supplied by the current fixture.',
                          'P4': 'No successful fixture service measurement at this snapshot.',
                          'fixed_R4_evidence': 'Contracts and failed eager smoke, not closed-loop speculation.',
                          'original_adaptive_evidence': 'Distributed original-gate verification UNVALIDATED.'})
        rows.append({'id': ident, 'card_path': relative, 'fields': fields,
                     'ten_question_field_presence': {q: {'fields': fs, 'present': all(f in fields for f in fs),
                                                         'semantic_completion': 'NOT_INFERRED_FROM_LABELS'}
                                                     for q, (fs, _) in QUESTIONS.items()},
                     'earliest_available_inputs': earliest, 'ownership_gap_or_boundary': ownership,
                     'evidence_limit': limit, 'observed_scope': scope,
                     'mode_cells': {f'{size}_{mode}': 'Candidate live E2E NOT_RUN. Read observed_scope for narrower diagnostics.'
                                    for size in ['Base1.4B', 'Base2.6B'] for mode in ['fixed_R4', 'original_adaptive']},
                     'nearest_prior': review['nearest_prior'], 'evidence_paths': review['sources'],
                     'status_record': statuses[ident][1]})
    assert len(rows) == 19 and set(DETAILS) == set(statuses)
    assert sorted(f for fs, _ in QUESTIONS.values() for f in fs) == list(range(1, 21))
    extra = [
        'reports/native_long_context_observation_v1.md', 'analysis/long_context_complete_v2/summary.json',
        'reports/native_quality_tp2_completion_v1.md', 'artifacts/native_quality_tp1_tp2_calibration_v1.json',
        'reports/c07_v2_eager_capture_port_contract_v1.md',
        'outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/guard_status.json',
        'outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/status.json',
        'outputs/quality_native_async_base14_P2_v1/guard_status.json',
        'outputs/quality_native_async_base14_P2_v1/1.4B_native_fixed_TP2_async/status.json',
        'outputs/quality_native_async_base26_P2_v1/guard_status.json',
        'outputs/quality_native_async_base26_P2_v1/2.6B_native_fixed_TP2_async/status.json',
        'reports/quality_and_exactness.md', 'reports/model_mode_scaling_matrix.md',
        'reports/source_verification.md', 'reports/search_map.md', 'reports/reproduce.md',
        'reports/scope_and_prior_evidence.md', 'reports/dependency_red_team.md',
        'reports/novelty_matrix.md', 'reports/negative_results.md',
        'figures/ownership_dags.mmd', 'scripts/verify_dag.py',
        'artifacts/ledger_current.json', 'artifacts/time_ledger.jsonl',
        'research_state.json', 'reports/next_actions.md',
        'tests/test_contract_validator_scope_v1.py',
        'figures/contract_frontier_and_c14_control_v1.mmd',
        'reports/contract_ownership_dag_supplement_v1.md',
        'templates/candidate_card.md',
        'outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/service.json',
        'outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/preparation.json',
        'outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/capture_finish.json',
        'outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/rank0/fixture.archive-v1/manifest.json',
    ]
    for source in extra:
        read(source)
    spec2 = importlib.util.spec_from_file_location('audit_contract_scope_fixture_v1', ROOT / 'tests/test_contract_validator_scope_v1.py')
    fixture = importlib.util.module_from_spec(spec2)
    sys.modules[spec2.name] = fixture
    spec2.loader.exec_module(fixture)
    counterexamples = []
    for case in fixture.scope_cases():
        counterexamples.append({**{k: v for k, v in case.items() if k != 'nodes'},
                                'nodes': [asdict(n) for n in case['nodes']],
                                'existing_checker_accepts': fixture.DAG.validate(case['nodes'])})
    for source, meta in inputs.items():
        p = Path(source)
        if not p.is_absolute():
            p = ROOT / p
        assert digest(p.read_bytes()) == meta['sha256'], f'Input changed during audit: {source}'
    return {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Interim contract-completeness audit. No final verdict or duration completion claim.',
        'authoritative_spec_matches_snapshot': spec == snapshot,
        'ten_question_grouping': {q: {'spec_fields': fs, 'question': question} for q, (fs, question) in QUESTIONS.items()},
        'grouping_authority': 'Reviewer grouping of all 20 spec fields. No separately named ten-question list appeared in the authoritative specification.',
        'historical_status_caution': {
            'paths': ['reports/search_map.md', 'reports/scope_and_prior_evidence.md'],
            'status': 'COMPLETED_FACTORIAL_WORDING_UNSUPPORTED_OR_STALE_FOR_THIS_AUDIT',
            'rule': 'Do not infer current status, counts, or conclusions of the protected separate factorial experiment.',
            'separate_experiment_current_status': 'NOT_EVALUATED',
        },
        'candidate_rows': rows, 'validator_scope_counterexamples': counterexamples,
        'counterexample_limit': 'These inputs characterize validator coverage. They do not show that previously generated graphs are invalid.',
        'template_candidate_card_exists': (ROOT / 'templates/candidate_card.md').exists(),
        'current_figures_directory_files': sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'figures').glob('*') if p.is_file()),
        'inputs': inputs, 'inputs_unchanged_during_build': True,
        'gpu_execution': 'NOT_RUN', 'model_execution': 'NOT_RUN',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = build()
    # Generated analysis only. Refuse to replace an existing destination.
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'output': str(target), 'candidates': len(result['candidate_rows']),
                      'input_files': len(result['inputs']),
                      'all_twenty_field_labels_present': True,
                      'semantic_completion': 'NOT_INFERRED',
                      'counterexamples': len(result['validator_scope_counterexamples'])}))


if __name__ == '__main__':
    main()
