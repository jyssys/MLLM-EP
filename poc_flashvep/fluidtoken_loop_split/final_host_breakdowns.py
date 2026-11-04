"""Export disjoint host request components for the accepted recovery controls."""
import csv
from .common import *


def main():
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .host_breakdown import analyze
    state = json.loads((ROOT / 'state/huginn_final_recovery_gate.json').read_text())
    names = [trial[key] for trial in state['results'] for key in ('clean', 'profile')]
    names += ['recovery_huginn_paced8_v5_dual_activity512',
              'recovery_huginn_paced8_v5_s4_paired_no_probe512']
    rows = []
    with quiet_analysis('final_recovery_host_breakdowns'):
        for name in names:
            result = analyze(name)
            assert not result['failures'], (name, result['failures'])
            for component, value in result['components'].items():
                rows.append(dict(name=name, model='huginn', component=component,
                    measured_requests=result['measured_requests'],
                    summed_request_E2E_seconds=result['summed_request_E2E_seconds'], **value,
                    evidence='MEASURED_DISJOINT_HOST_REQUEST_PARTITION',
                    scope='Percent of summed request E2E, not exclusive GPU execution or server wall time.'))
        with (ROOT / 'recovery_host_E2E_components.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write(ROOT / 'recovery_host_E2E_components.json', rows)
    print(json.dumps(dict(runs=len(names), components=len(rows))))


if __name__ == '__main__':
    main()
