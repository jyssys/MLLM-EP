"""Serial host-observed E2E partitions. These are not GPU kernel shares."""
import csv
import json
from .common import ROOT, write
from .analyze import read_rows


def main():
    output, exclusions = [], []
    phases = ('harness_delivery', 'coordinator_and_P_queue', 'P_prefill_observed',
              'exposed_handoff_observed', 'D_resume_to_first_token', 'generation_after_first_token')
    for folder in sorted((ROOT/'raw').glob('quiet*')):
        if not (folder/'stats.json').exists():
            continue
        rows = read_rows(folder)
        metrics = {m['request_id']:m for m in json.loads((folder/'stats.json').read_text())['metrics']}
        totals = [0.]*len(phases)
        count = 0
        for row in rows:
            rid = f"r{row['run']}_q{row['sample_id']}"
            m = metrics.get(rid)
            if m is None:
                exclusions.append(dict(run=folder.name, sample_id=row['sample_id'], reason='Missing coordinator metric'))
                continue
            t, created = m['timings'], m['created_monotonic']
            points = [row['arrival'], row['admission'], created+t['prefill_started'],
                      created+t['prefill_compute_completed'], created+t['d_received'],
                      row['arrival']+row['ttft_ms']/1000, row['finish']]
            if any(right < left for left, right in zip(points, points[1:])):
                exclusions.append(dict(run=folder.name, sample_id=row['sample_id'], reason='Host observations are not monotonic', points=points))
                continue
            for i, (left, right) in enumerate(zip(points, points[1:])):
                totals[i] += 1000*(right-left)
            count += 1
        denominator = sum(totals)
        if denominator:
            for phase, total in zip(phases, totals):
                output.append(dict(run=folder.name, phase=phase, N=count, aggregate_request_ms=total,
                                   share_pct=100*total/denominator, denominator_ms=denominator,
                                   evidence='MEASURED_HOST_OBSERVATION',
                                   scope='Sum of request E2E intervals. Not a cohort critical-path or GPU-DMA partition.'))
    if output:
        with (ROOT/'e2e_component_shares.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(output)
    write(ROOT/'host_breakdown_exclusions.json', exclusions)


if __name__ == '__main__':
    main()
