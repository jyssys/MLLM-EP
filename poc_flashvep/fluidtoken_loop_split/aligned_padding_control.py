"""Separate fixed-projection padding from a claimed Huginn Loop advantage."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    target = ROOT / 'state/aligned_padding_control.json'
    source = ROOT / 'state/static_screen.json'
    write(target, dict(status='WAITING_FOR_STATIC_SCREEN'))
    while time.time() < LONG_START_CUTOFF:
        if source.exists():
            state = json.loads(source.read_text())
            if state['status'] == 'STATIC_SCREEN_FINISHED':
                break
            if state['status'].startswith('BLOCKED'):
                # A host-audit replacement can still requalify this gate.
                replacement = ROOT / 'state/audit_replacement_confirmation.json'
                pending = json.loads(replacement.read_text()).get('status') if replacement.exists() else None
                if pending not in ('WAITING', 'IN_PROGRESS'):
                    write(target, dict(status='BLOCKED_METHOD_GATE'))
                    return
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    from .conditional_split_qualification import recovery_ready
    if not recovery_ready(json.loads((ROOT / 'state/huginn_final_recovery_gate.json').read_text())):
        write(target, dict(status='BLOCKED_BY_HUGINN_GATE'))
        return
    rows = [row for row in state['results'] if row.get('model') == 'huginn']
    if len(rows) != 5 or not all(row.get('correct') for row in rows):
        write(target, dict(status='BLOCKED_CONNECTOR_CORRECTNESS'))
        return
    from .static_qualification import connector_gate
    from .state_comparison import compare
    from .static_screen import metrics
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    ledger = []
    for count, heavy in ((8, True), (32, True), (32, False)):
        for axis, label in (('loop', 'loop_partial'), ('token', 'token_partial')):
            prefix = 'static_gate_v4_' if heavy else 'static_screen_v2_'
            original = f'{prefix}huginn_{label}{count}'
            reference = f'{prefix}huginn_atomic_native{count}'
            cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (original + '.json')).read_text()))
            cfg.update(name=f'aligned_padding_huginn_{axis}_{"state" if heavy else "clean"}{count}',
                       phase='ALIGNED_256_ROW_PROJECTION_CONTROL',
                       cut=28 if axis == 'loop' else 32, prefix_tokens=1792)
            if time.time() >= LONG_START_CUTOFF:
                write(target, dict(status='NOT_RUN_TIME_BUDGET', results=ledger))
                return
            save(cfg)
            complete = execute(cfg['name'])
            row = dict(name=cfg['name'], axis=axis, heavy=heavy, complete=complete, passed=False)
            if complete:
                with quiet_analysis(cfg['name'] + '_analysis'):
                    row['output'] = evidence.compare(reference, cfg['name'])
                    row['full_work'] = evidence.full_work(cfg['name'])
                    if heavy:
                        row['connector'] = connector_gate(cfg['name'])
                        row['state'] = compare(reference, cfg['name'])
                    else:
                        row['metrics'] = metrics(cfg['name'])
                    row['passed'] = bool(row['output']['passed'] and row['full_work']['passed']
                        and (not heavy or row['connector'] and row['state']['exact']))
            ledger.append(row)
            write(target, dict(status='IN_PROGRESS', results=ledger))
            if not row['passed']:
                write(target, dict(status='FAILED_OR_UNQUALIFIED', results=ledger))
                return
    write(target, dict(status='COMPLETE_N32_CONTROL', results=ledger,
        scope='Huginn Loop28 versus Token1792 matches assigned recurrent rows and selected-projection padded rows. Attention work is not equal. No dynamic or replicated capacity claim.'))


if __name__ == '__main__':
    main()
