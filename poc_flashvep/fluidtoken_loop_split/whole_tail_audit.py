"""Keep complete-run Decode tails separate from capture-window gates."""
from .common import *


def one(name):
    from .static_screen import metrics
    folder = ROOT / 'raw' / name
    values = metrics(name)
    capture = folder / 'activity_capture.json'
    value = json.loads(capture.read_text()) if capture.exists() else None
    export = (value['stop']['monotonic_s'], value['stop']['monotonic_s'] + value['stop_call_seconds']) if value else None
    probes = [json.loads(line) for line in (folder / 'D_probe_requests.jsonl').read_text().splitlines()]
    events = []
    for line in (folder / 'requests.jsonl').read_text().splitlines():
        request = json.loads(line)
        if request.get('warmup'):
            continue
        for e in request['token_events']:
            if e['first']:
                continue
            a, b = e['previous_time'], e['time']
            duration = 1000 * (b - a) / e['new_tokens']
            events.append(dict(sample=request['sample_id'], start=a, end=b, ITL_ms=duration,
                violation=duration > values['frozen_SLO']['ITL_ms'],
                overlaps_export=bool(export and a < export[1] and b > export[0]),
                overlaps_probe=any(a < p['finish'] and b > p['time'] for p in probes)))
    violations = [e for e in events if e['violation']]
    return dict(name=name, metrics=values, ITL_intervals=len(events), violations=len(violations),
        violations_overlapping_export=sum(e['overlaps_export'] for e in violations),
        violations_overlapping_probe=sum(e['overlaps_probe'] for e in violations),
        top_intervals=sorted(events, key=lambda e: -e['ITL_ms'])[:20],
        scope='Same-run temporal overlap, not proof of causation. Complete-run tails include all offered background outputs.')


def main():
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    state = json.loads((ROOT / 'state/admitted_buffer_recovery.json').read_text())
    rows = []
    with quiet_analysis('whole_run_tail_audit'):
        for q in state['qualification']:
            cfg = json.loads((ROOT / 'configs' / (q['name'] + '.json')).read_text())
            rows.append(dict(clean=one(cfg['matched_clean']), profiled=one(q['name'])))
        write(ROOT / 'state/whole_run_tail_audit.json', dict(pairs=rows,
            status='REVIEW_COMPLETE', scope='This audit does not modify or replace the retained same-window verdict.'))
    print(json.dumps([dict(clean=r['clean']['metrics']['ITL_p99_ms'],
                          profiled=r['profiled']['metrics']['ITL_p99_ms'],
                          violations=r['profiled']['violations'],
                          export_overlap=r['profiled']['violations_overlapping_export'],
                          probe_overlap=r['profiled']['violations_overlapping_probe']) for r in rows]))


if __name__ == '__main__':
    main()
