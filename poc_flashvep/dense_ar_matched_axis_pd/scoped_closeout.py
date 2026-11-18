"""Close the user-revised scope without completing deferred GPU matrices."""
import collections
import json
import time
from . import common
from .common import ROOT, digest, preserve, write
from .runtime_freeze import check


def coverage(expected, measured):
    assert len(expected) == len(set(expected)), 'Duplicate planned start'
    missing = sorted(set(expected)-set(measured))
    assert not missing, ('Incomplete revised scope', missing)
    return len(expected)


def read(path):
    return json.loads(path.read_text())


def main():
    from .batch_matched_monitor import snapshot
    from .batch_matched import audit as batch_audit
    from .fixed_cohort_audit import audit as cohort_audit
    from .recovery import resolve_name
    preserve(); check()
    current = snapshot()
    assert current['all_scopes_complete'] and not current['critical'], current
    assert not current['controllers'] and current['lease']['status'] == 'RELEASED'
    transition = read(ROOT/'state/user_batch_matched_scope_revision_20261118.json')
    assert transition['status'] == 'TRANSITION_COMPLETE' and transition['GPU_jobs_terminated'] == 0
    data = read(ROOT/'analysis/independent_start_statistics.json')
    integrity = read(ROOT/'analysis/partial_integrity_validation.json')
    names = {r['name'] for r in data['starts']}
    assert integrity['passed'] and {r['name'] for r in integrity['runs']} == names
    assert read(ROOT/'analysis/tabular_export_manifest.json')['starts'] == len(names)
    designs = [read(ROOT/'state'/p) for p in
               ('batch_matched_design_20261118.json', 'cohort2_online_design.json')]
    assert [coverage(d['names'], names) for d in designs] == [98, 42]
    gates = []
    for design, audit in zip(designs, (batch_audit, cohort_audit)):
        for name in design['names']:
            result = audit(name)
            assert result['passed']
            gates.append(name)
    qualification = []
    for marker in ('batch1_full_state_qualified.json', 'fixed_cohort2_v3_qualification_complete.json'):
        value = read(ROOT/'state'/marker)
        assert value['passed'] and len(value['names']) == 7
        for name in value['names']:
            state = read(ROOT/'analysis'/(name+'_state_gate.json'))
            assert state['passed'], name
        qualification.append(dict(marker=marker, names=value['names'], count=18))
    previous = read(ROOT/'state/predecessor_vs_228a_source_audit.json')
    assert previous['equal'] == previous['files'] == 523
    primary = read(ROOT/'state/matrix_manifest.json')['expected']
    repeat_manifest = ROOT/'state/confirmation_manifest.json'
    if repeat_manifest.exists():
        repeats = read(repeat_manifest)['expected']
    else:
        from .campaign import key
        frozen = read(ROOT/'state/confirmation_design_before_performance.json')
        arms = [('Agg', 0), ('Native', 0)]+[(a, m) for m in frozen['layers'] for a in ('Layer', 'Token')]
        repeats = [key(a, m, frozen['count'], cap, frozen['QPS'], seed, 'confirmation', frozen['admission'])
                   for seed in frozen['seeds'] for cap in frozen['outputs'] for a, m in arms]
    deferred_discovery = [resolve_name(n) for n in primary if resolve_name(n) not in names]
    deferred_confirmation = [resolve_name(n) for n in repeats if resolve_name(n) not in names]
    stages = dict(collections.Counter(r['stage'] for r in data['starts']))
    assert len(deferred_discovery) == 34 and len(deferred_confirmation) == 120
    assert stages.get('admission_control') == 52
    assert (ROOT/'state/last2_supplement_cancelled.json').exists()
    assert not any(n.startswith('last2_') for n in names)
    report = dict(time=time.time(), passed=True, final=True,
        scope_revision='USER_REQUESTED_ACTUAL_BATCH_MATCHING', starts=len(names), stages=stages,
        actual_batch1_starts=98, actual_preparation1_decode2_starts=42,
        actual_geometry_verified_names=gates, state_qualification=qualification,
        deferred_original_discovery=deferred_discovery,
        deferred_original_confirmation=deferred_confirmation,
        historical_Native_bridge=dict(status='NOT_RUN', planned_starts=6),
        original_manifests_completed=False, cancelled_Last2_full_sweep_GPU_starts=0,
        per_start_integrity_SHA256=digest(ROOT/'analysis/partial_integrity_validation.json'),
        source_audit_SHA256=digest(ROOT/'state/predecessor_vs_228a_source_audit.json'),
        scope='Close only the revised actual-batch protocols. Preserve the incomplete original matrices, failed prototypes and historical evidence. Full-state tests use N18 per new arm. N144 timing verifies outputs, layer/query coverage and lifecycle without full KV hashing. Completion does not establish a favorable axis crossover or sustainable capacity.')
    write(ROOT/'analysis/final_integrity_validation.json', report)
    marker = ROOT/'state/CAMPAIGN_CLOSED.json'
    if not marker.exists():
        write(marker, dict(time=time.time(), status='CLOSED',
            reason='The user-revised actual-batch protocols completed. Original deferred matrices remain NOT_RUN. Close GPU admission for final analysis and publication.',
            actual_batch1_starts=98, actual_preparation1_decode2_starts=42,
            historical_discovery_starts=198, historical_admission_controls=52,
            deferred_original_discovery_starts=34, deferred_original_confirmation_starts=120,
            deferred_historical_Native_bridge_starts=6, original_manifests_completed=False,
            Last2_sweep_cancelled=True, Last2_sweep_GPU_starts=0))
    else:
        assert read(marker)['original_manifests_completed'] is False
    target = ROOT/'state/final_GPU_resource_restoration.json'
    if not target.exists():
        common.bind_legacy()
        from ..dense_ar_layer_token_pd import resource_closeout as proven
        for key in ('ROOT', 'QUIET_LOCK', 'resources', 'preserve', 'write', 'digest'):
            setattr(proven, key, getattr(common, key))
        proven.main()
    assert read(target)['passed']
    write(ROOT/'state/science_closeout.json', dict(time=time.time(),
        status='REVISED_GPU_SCOPE_CLOSED_WITH_EVIDENCE_LIMITATIONS',
        final_integrity_SHA256=digest(ROOT/'analysis/final_integrity_validation.json'),
        resource_restoration_passed=True,
        completion_restore_incident=dict(observed_all_GPUs_idle=True,
            observation_local_time='2026-11-18T23:05:11+09:00',
            cause='The actual-batch monitor returned on completion without calling burner restoration.',
            action='Restore all eight owned burners. Add a failing completion-path regression test and the missing guarded restoration call.',
            performance_results_changed=False),
        publication='PENDING_SEPARATE_REPORT_AND_REMOTE_SHA_VERIFICATION'))
    write(ROOT/'state/final_progress_manifest.json', dict(time=time.time(), status='CLOSED_GPU_SCOPE',
        batch1_completed=98, batch1_planned=98, cohort2_completed=42, cohort2_planned=42,
        earlier_discovery_completed=198, earlier_discovery_planned=232,
        earlier_confirmation_status='DEFERRED_BY_USER_SCOPE_REVISION_NOT_RUN',
        historical_Native_bridge_status='DEFERRED_BY_USER_SCOPE_REVISION_NOT_RUN',
        current_GPU_use='OWNED_BURNERS_0_TO_7',
        safe_resume='Do not restart deferred GPU controllers or duplicate completed starts. Reports, archive and private-index publication are separate CPU steps.'))
    preserve(); check()
    print('REVISED_SCOPE_CLOSEOUT', len(names), 98, 42, flush=True)


if __name__ == '__main__':
    main()
