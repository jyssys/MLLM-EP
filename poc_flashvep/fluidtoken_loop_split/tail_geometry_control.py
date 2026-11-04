"""Test observed Decode batches beyond the qualified four-row graph limit."""
import copy
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    target = ROOT / 'state/tail_geometry_control.json'
    write(ROOT / 'state/tail_geometry_plan.json', dict(
        hypothesis='Rare five/six-row Decode submissions exceed the four-row graph limit and can cause eager tail spikes.',
        evidence='Clean start 3 has 341 five-row and 60 six-row recurrent submissions. Its ITL p99 is 263.483 ms.',
        controls=['D slots 8 / graph rows 4', 'D slots 4 / graph rows 4', 'D slots 8 / graph rows 8'],
        fixed='Same 2048/16 prompts, arrival seed 2026111603, 1.971 QPS, native transport, KV pool, and probe policy.',
        gates='New exact-state qualification precedes changed geometry. Full-run SLO follows. Same-window GPU gates remain separate.',
        reason='Whole-run tail failures also occur without profiling. Export overhead alone cannot explain them.'))
    from ..looped_pd_autonomous_headroom.campaign import gate
    from ..looped_pd_atomic_prefill import evidence
    from .state_comparison import compare
    from .static_screen import metrics
    from .probe_reference import verify
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    reference_state = 'recovery_huginn_graph_state8'
    base_state = json.loads((ROOT / 'configs' / (reference_state + '.json')).read_text())
    base_clean = json.loads((ROOT / 'configs/recovery_huginn_buffer16_confirm3_clean512.json').read_text())
    ledger = []
    for label, slots, rows in [('native8_graph4', 8, 4), ('slots4_graph4', 4, 4), ('slots8_graph8', 8, 8)]:
        record = dict(label=label, D_slots=slots, graph_rows=rows)
        if label != 'native8_graph4':
            cfg = copy.deepcopy(base_state)
            cfg.update(name='recovery_huginn_' + label + '_state8', D_slots=slots, graph_rows=rows,
                       observe_native_graph_coverage=True)
            save(cfg)
            complete = execute(cfg['name'])
            record['state_run'] = cfg['name']
            if complete:
                with quiet_analysis(cfg['name'] + '_qualification'):
                    record['state_gate'] = gate(cfg['name'])
                    record['state_parity'] = compare(reference_state, cfg['name'])
                    record['output'] = evidence.compare(reference_state, cfg['name'])
            if not complete or not (record['state_gate'] and record['state_parity']['exact'] and record['output']['passed']):
                record['status'] = 'BLOCKED_CORRECTNESS'
                ledger.append(record)
                write(target, dict(status='IN_PROGRESS', results=ledger))
                continue
        cfg = copy.deepcopy(base_clean)
        cfg.update(name='recovery_huginn_' + label + '_tail512', D_slots=slots, graph_rows=rows,
            observe_native_graph_coverage=True, kernel_activity_seconds=0., phase='MATCHED_CLEAN_DECODE_GEOMETRY_CONTROL')
        save(cfg)
        complete = execute(cfg['name'])
        record.update(name=cfg['name'], complete=complete)
        if complete:
            with quiet_analysis(cfg['name'] + '_tail_gate'):
                record['metrics'] = metrics(cfg['name'])
                record['output'] = evidence.compare('recovery_huginn_qualified_admission_no_probe512', cfg['name'])
                record['probe'] = verify(cfg['name'])
            record['passed'] = (record['metrics']['Decode_SLO_screen'] and record['output']['passed']
                and record['probe']['exact'] and record['probe']['full_work'])
        else:
            record['passed'] = False
        ledger.append(record)
        write(target, dict(status='IN_PROGRESS', results=ledger))
    write(target, dict(status='GEOMETRY_SCREEN_COMPLETE', results=ledger,
        scope='Single matched clean start per geometry. No same-run P saturation claim until a separate paired GPU capture.'))


if __name__ == '__main__':
    main()
