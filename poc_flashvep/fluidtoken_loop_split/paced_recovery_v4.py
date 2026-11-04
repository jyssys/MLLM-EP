"""Test exact output-paced local work after the failed independent ITL repeat."""
import copy
from .common import *
from .run import execute
from .recovery_qualification import save


def main(version='v4', kv_blocks=None):
    initialize()
    prior_gate = json.loads((ROOT / 'state/audit_replacement_confirmation.json').read_text())
    assert prior_gate['status'] == 'RECOVERY_FAILED'
    target = ROOT / 'state' / f'paced_recovery_{version}.json'
    write(ROOT / 'state' / f'paced_recovery_{version}_plan.json', dict(
        hypothesis='A full-R32 local request produces a token-latency spike near the frozen p99 limit.',
        failed_repeat='allphase_slots8_audit_replacement_joint_s4_profile512',
        measured_ITL_p99_ms=194.8770748078823, frozen_ITL_ms=192.8212538175284,
        intervention='Use the existing staged D scheduler. Submit at most one exact local Prefill stage per observed background output step.',
        unchanged='Keep P execution, BF16, full R32 work, KV ownership, SLO thresholds, and the profiler perturbation gate.',
        configuration=dict(D_slots=8, graph_rows=8, nominal_QPS=1.971, probe_admission_guard=8,
                           D_KV_blocks=kv_blocks or 28288),
        memory_control='The optional 26752-block pool preserves six complete 2048/16 reservations and releases 3.8671875 GiB for staged workspace. Requalification is mandatory. The failed 28288-block state test remains preserved.',
        admission_note='Only one independent local probe can be in flight. The native KV guard still applies.',
        independent_starts=[4, 2, 3], reason='Retest the failed seed first. Do not select only passing historical starts.',
        limit='This is a D-local scheduling intervention, not P-to-D migration. Staged execution and pacing are combined.'))
    from ..looped_pd_autonomous_headroom.campaign import gate
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_atomic_prefill import evidence
    from .state_comparison import compare, states
    reference = 'recovery_huginn_async_evidence_slots8_graph8_state8'
    cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (reference + '.json')).read_text()))
    cfg.update(name=f'recovery_huginn_paced8_{version}_state8', D_local_probe_output_pacing=True)
    if kv_blocks is not None:
        cfg['D_kv_blocks'] = kv_blocks
    save(cfg)
    complete = execute(cfg['name'])
    record = dict(name=cfg['name'], complete=complete)
    if complete:
        with quiet_analysis(cfg['name'] + '_qualification'):
            record.update(work_and_transfer=gate(cfg['name']), output=evidence.compare(reference, cfg['name']),
                          state=compare(reference, cfg['name']))
            left, right = states(reference, 16), states(cfg['name'], 16)
            record['local_prompt_state_exact'] = bool(left) and left == right
    if not (complete and record['work_and_transfer'] and record['output']['passed']
            and record['state']['exact'] and record['local_prompt_state_exact']):
        write(target, dict(status='BLOCKED_CORRECTNESS', qualification=record))
        return
    base = json.loads((ROOT / 'configs/recovery_huginn_allphase_slots8_admitted_v2_joint_s1_clean512.json').read_text())
    base.update(D_local_probe_output_pacing=True, D_probe_max_active_decode=8)
    if kv_blocks is not None:
        base['D_kv_blocks'] = kv_blocks
    from .geometry_confirmation import pair
    results = []
    for start in (4, 2, 3):
        trial = pair(base, f'paced8_{version}', start, roles=('prefill',))
        results.append(trial)
        write(target, dict(status='IN_PROGRESS', qualification=record, results=results))
        if not trial['passed']:
            write(target, dict(status='RECOVERY_FAILED', qualification=record, results=results))
            return
    write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
        status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
        independent_repeats=True, exact_state_and_output=True,
        qualified_configuration=base, geometry_state_gate=record, results=results,
        D_kernel_activity='NOT_MEASURED_IN_ACCEPTED_RUNS',
        scope='Output-paced exact local requests. Three new paired starts include the previous failed arrival seed. No migrated serving work is qualified.'))
    write(target, dict(status='RECOVERY_READY', qualification=record, results=results))
    from .conditional_split_qualification import main as qualify
    from .static_screen import main as screen
    qualify()
    screen()


if __name__ == '__main__':
    main()
