"""Gate bulk host bookkeeping before matched serving controls."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .campaign import gate


def main():
    validation = ROOT/'state/bulk_written_cpu_validation.json'
    while time.time() < LONG_START_CUTOFF and not validation.exists():
        time.sleep(5)
    assert validation.exists() and json.loads(validation.read_text())['all_exact']
    activate()
    from ..looped_pd_atomic_prefill import evidence
    ledger = []
    for model, rate, cap in [('huginn', 1.3, 16), ('ouro', 6.355, 32)]:
        reference = f'prefill_graph_{model}_native_graph_gate8'
        cfg = copy.deepcopy(json.loads((ROOT/'configs'/(reference+'.json')).read_text()))
        cfg.update(name=f'bulk_written_{model}_gate8', P_graph_bulk_written=True)
        name = put(cfg, 8)
        write(ROOT/'state/bulk_written_controls.json', dict(stage='CORRECTNESS', name=name, time=time.time()))
        passed = execute(name) and gate(name, reference)
        ledger.append(dict(model=model, name=name, phase='correctness', passed=passed))
        write(ROOT/'state/bulk_written_control_results.json', ledger)
        if not passed:
            continue
        previous = None
        for label in ('eager', 'graph', 'bulk'):
            candidate = copy.deepcopy(cfg)
            candidate.update(name=f'bulk_written_{model}_{label}_clean24', cap=cap,
                             phase='qps_pilot', verify_transfer=False, state_probe=False,
                             arrival_seed=2026110361, seed=2026110362,
                             P_graph_bulk_written=label == 'bulk',
                             P_prefill_graph_rows=cfg['P_prefill_graph_rows'] if label != 'eager' else 0)
            cname = put(candidate, 24, rate, split='confirmation')
            write(ROOT/'state/bulk_written_controls.json', dict(stage='MATCHED_CLEAN', name=cname, time=time.time()))
            complete = execute(cname)
            record = dict(model=model, name=cname, phase='matched_clean', complete=complete)
            if complete:
                record['work'] = evidence.full_work(cname)
                record['parity'] = evidence.compare(previous, cname) if previous else None
                complete = record['work']['passed'] and (record['parity'] is None or record['parity']['passed'])
                record['passed'] = complete
            ledger.append(record)
            write(ROOT/'state/bulk_written_control_results.json', ledger)
            if not complete:
                break
            previous = cname
    write(ROOT/'state/bulk_written_controls.json', dict(stage='COMPLETE_REVIEW_REQUIRED', results=ledger, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/bulk_written_controls.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
