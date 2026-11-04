"""Refresh exact-state and finite-service checks with the current source."""
import copy
from .common import *
from .run import execute


def save(cfg):
    path = ROOT / 'configs' / (cfg['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg
    else:
        write(path, cfg)


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.campaign import gate
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom import verify_long_probes as probes
    probes.execute = execute
    ledger = []
    base = json.loads((ROOT / 'configs/recovery_huginn_original_clean384.json').read_text())
    for label, graph in (('eager', 0), ('graph', 4096)):
        cfg = copy.deepcopy(base)
        for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
            cfg.pop(key, None)
        cfg.update(name='recovery_huginn_' + label + '_state8', count=8,
                   P_prefill_graph_rows=graph, state_probe=True, verify_transfer=True,
                   phase='CORRECTNESS_NOT_CLEAN_TIMING')
        for key in ('request_caps', 'prompt_ids', 'arrival_offsets'):
            cfg[key] = cfg[key][:8]
        save(cfg)
        complete = execute(cfg['name'])
        row = dict(name=cfg['name'], complete=complete,
                   exact_state_and_work=complete and gate(cfg['name']))
        if label == 'graph' and complete:
            row['output_parity'] = evidence.compare('recovery_huginn_eager_state8', cfg['name'])
        ledger.append(row)
        write(ROOT / 'state/recovery_qualification.json', ledger)
        if not row['exact_state_and_work'] or (label == 'graph' and not row['output_parity']['passed']):
            return
    cfg = json.loads((PREVIOUS / 'configs/capacity_short_huginn_L2048_v2.json').read_text())
    cfg.update(name='capacity_recovery_huginn_L2048',
               phase='INDEPENDENT_FINITE_SERVICE_REFERENCE',
               jobs=[dict(batch=batch, cap=1, repeats=10) for batch in (1, 2)])
    save(cfg)
    ledger.append(dict(name=cfg['name'], complete=execute(cfg['name'])))
    write(ROOT / 'state/recovery_qualification.json', ledger)
    ledger.append(dict(probe_correctness=probes.verify('recovery_huginn_original_clean384')))
    write(ROOT / 'state/recovery_qualification.json', ledger)


if __name__ == '__main__':
    main()
