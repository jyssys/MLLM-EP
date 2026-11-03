"""Test a larger legal P batch before retaining a compute-pressure claim."""
from .common import *
from .prepare import put
from .run import execute


def main():
    ledger = []
    for model, source in (('ouro', 'capacity_bulk_written_ouro_matched'),
            ('huginn', 'capacity_short_huginn_L2048_v2')):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        cfg.update(name=f'capacity_batch4_{model}_L{cfg["length"]}',
            capacity=4, slots=4, P_slots=4, token_budget=4*cfg['length'],
            P_prefill_graph_rows=4*cfg['length'],
            jobs=[dict(batch=b, cap=1, repeats=3) for b in (1, 2, 4)],
            worker='capacity_worker', capacity_mode='P_ONLY', P_graph_local_capacity=True,
            scope='Stronger finite local P batching control. Native output reference and full depth remain mandatory. Not a sustainable capacity estimate.')
        name = put(cfg, 4, None, 'confirmation')
        write(ROOT/'state/batch_capacity_control.json', dict(stage='CAPACITY', name=name, time=time.time()))
        okay = execute(name)
        file = ROOT/'raw'/name/'cohorts.jsonl'
        rows = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
        ledger.append(dict(name=name, model=model, complete=okay,
            cohorts=[dict(batch=r['job']['batch'], repeat=r.get('repeat'), status=r['status'],
                P_service_s=r.get('prefill_host_service_s'),
                native_output_parity=r.get('native_output_parity'),
                full_depth_rows=r.get('full_depth_rows'), observed_rows=r.get('full_depth_rows_observed')) for r in rows]))
        write(ROOT/'state/batch_capacity_control_results.json', ledger)
    write(ROOT/'state/batch_capacity_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/batch_capacity_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
