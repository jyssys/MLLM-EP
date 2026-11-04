"""Analyze completed marker-free controls without importing device start times."""
import csv
from .common import *


def main():
    initialize()
    from ..fluidtoken_loop_split.static_screen import metrics
    from ..looped_pd_atomic_prefill import evidence
    from .work_accounting import summarize
    from .lifetime_gate import check
    ledger = json.loads((ROOT / 'state/measurement_validation.json').read_text())
    rows = []
    for model, plans in ledger['markers'].items():
        reference = f'j_{model}_native_marker_off128'
        for plan, pair in plans.items():
            for control in pair.get('controls', []):
                if control['enabled'] or not control['complete']:
                    continue
                name = control['name']
                folder = ROOT / 'raw' / name
                cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
                assert not cfg.get('observe_service_timing') and not cfg.get('kernel_activity_seconds')
                output = evidence.compare(reference, name)
                work = evidence.full_work(name)
                lifetime = check(name)
                accounting = summarize(name)
                row = dict(name=name, model=model, plan=plan,
                    output=output, full_work=work, lifetime=lifetime,
                    passed=all(value['passed'] for value in (output, work, lifetime)),
                    metrics=metrics(name), accounting=accounting,
                    Service_TTFT='NOT_MEASURED_IN_THIS_EXECUTION',
                    scope='One N128 start with execution-start markers disabled. Same controlled inputs, seed, nominal rate, and arrival trace. '
                        'No CUDA first-execution timestamp is available. Do not import Service TTFT from another run.')
                write_derived(folder / 'marker_free_serving_analysis.json', row)
                rows.append(row)
    write_derived(ROOT / 'state/marker_free_serving_analysis.json', dict(results=rows,
        scope='Separate marker-free request-latency controls. A rejected marker pair remains rejected. These controls do not supply Service TTFT.'))
    flattened = [dict(name=row['name'], model=row['model'], plan=row['plan'],
        output_work_lifetime_gate=row['passed'], Service_TTFT='NOT_MEASURED', **{
            key: value for key, value in row['metrics'].items() if not isinstance(value, (dict, list))}) for row in rows]
    if flattened:
        fields = sorted({key for row in flattened for key in row})
        with (ROOT / 'marker_free_serving_results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flattened)


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('marker_free_serving_analysis'):
        main()
