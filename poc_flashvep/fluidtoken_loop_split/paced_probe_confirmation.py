"""Qualify bounded exact recurrent probes only if admission controls fail."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save
from .geometry_confirmation import pair


def main():
    initialize()
    target = ROOT / 'state/paced_probe_confirmation_v3.json'
    write(ROOT / 'state/paced_probe_plan_v3.json', dict(
        hypothesis='An indivisible full-R32 local prompt can exceed token latency slack despite spare average D capacity.',
        intervention='Use existing staged D Prefill. Admit at most one probe submission after an observed background output step.',
        admission='The all-phase guard permits up to the native four-slot limit. The earlier two-request guard completed only four of twelve probes. Native KV and scheduling guards remain active.',
        limitations='This is scheduling of independent probes, not P-to-D migration or a method speedup. P scheduling remains unchanged.',
        gates='Exact background and local state first. Then unchanged whole-run and same-window gates, three independent starts.',
        order=[dict(D_slots=4, graph_rows=4, QPS=1.971), dict(D_slots=4, graph_rows=4, QPS=1.85)]))
    write(target, dict(status='WAITING_FOR_ADMISSION_CONTROL'))
    pending = ROOT / 'state/admitted_probe_confirmation.json'
    while time.time() < LONG_START_CUTOFF:
        if pending.exists():
            state = json.loads(pending.read_text())
            if state['status'] in ('RECOVERY_READY', 'NOT_NEEDED'):
                write(target, dict(status='NOT_NEEDED'))
                return
            if state['status'] == 'RECOVERY_FAILED_OR_INCOMPLETE':
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    from ..looped_pd_autonomous_headroom.campaign import gate
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_atomic_prefill import evidence
    from .state_comparison import compare, states
    reference = 'recovery_huginn_async_evidence_slots4_graph4_state8'
    cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (reference + '.json')).read_text()))
    cfg.update(name='recovery_huginn_paced_probe_state8', D_local_probe_output_pacing=True)
    save(cfg)
    complete = execute(cfg['name'])
    record = dict(complete=complete, name=cfg['name'])
    if complete:
        with quiet_analysis(cfg['name'] + '_state_gate'):
            record.update(state_gate=gate(cfg['name']), output=evidence.compare(reference, cfg['name']),
                          state=compare(reference, cfg['name']))
            left, right = states(reference, 16), states(cfg['name'], 16)
            record['local_prompt_state_exact'] = bool(left) and left == right
    if not (complete and record['state_gate'] and record['output']['passed']
            and record['state']['exact'] and record['local_prompt_state_exact']):
        write(target, dict(status='BLOCKED_CORRECTNESS', qualification=record))
        return
    base = json.loads((ROOT / 'configs/recovery_huginn_async_evidence_slots4_graph4_joint_s1_clean512.json').read_text())
    base.update(D_local_probe_output_pacing=True, D_probe_admission_scope='all_decode_phases',
                D_probe_max_active_decode=4, D_probe_completion_contract='admitted_exact_v2')
    results = []
    for rate in (1.971, 1.85):
        base['arrival_rate'] = rate
        label = 'paced_probe_native_slot_guard_q' + str(rate).replace('.', 'p')
        trials = []
        for start in (1, 2, 3):
            if time.time() >= LONG_START_CUTOFF:
                break
            trial = pair(base, label, start, roles=('prefill',))
            trials.append(trial)
            write(target, dict(status='IN_PROGRESS', qualification=record, results=results, current=trials))
            if not trial['passed']:
                break
        results.append(dict(label=label, trials=trials))
        if len(trials) == 3 and all(row['passed'] for row in trials):
            write(ROOT / 'state/huginn_final_recovery_gate.json', dict(
                status='RECOVERY_READY', same_run_gates=True, whole_run_decode_SLO=True,
                independent_repeats=True, exact_state_and_output=True,
                qualified_configuration=copy.deepcopy(base), geometry_state_gate=record, results=trials,
                D_kernel_activity='NOT_MEASURED_IN_ACCEPTED_RUNS',
                scope='One exact probe submission per observed background output step. This does not qualify migrated serving work.'))
            write(target, dict(status='RECOVERY_READY', qualification=record, results=results))
            from .conditional_split_qualification import main as qualify
            from .static_screen import main as screen
            qualify()
            screen()
            return
    write(target, dict(status='RECOVERY_FAILED_OR_INCOMPLETE', qualification=record, results=results))


if __name__ == '__main__':
    main()
