"""Compare a clean admission control with its unchanged clean workload."""
from .common import *


def main():
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_headroom_recovery.gates import percentile
    from .probe_reference import verify
    names = ['recovery_huginn_extended1_clean512', 'recovery_huginn_admitted_probe_clean512']
    with quiet_analysis('probe_admission_clean_comparison'):
        result = dict(evidence='MEASURED_HOST_LATENCY_AND_EXACT_OUTPUT_COMPARISON',
            scope='One clean paired start. No simultaneous GPU capture. Not a P-plus-D gate or method speedup.',
            output=evidence.compare(*names), runs=[])
        for name in names:
            folder = ROOT / 'raw' / name
            rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
            rows = [r for r in rows if not r.get('warmup')]
            itl = [1000 * (e['time'] - e['previous_time']) / e['new_tokens']
                   for r in rows for e in r['token_events'] if not e['first']]
            window = json.loads((folder / 'window.json').read_text())
            result['runs'].append(dict(name=name, completed=len(rows),
                TTFT_p95_ms=percentile([r['ttft_ms'] for r in rows], .95),
                TPOT_p95_ms=percentile([r['tpot_ms'] for r in rows], .95),
                ITL_p99_ms=percentile(itl, .99), maximum_ITL_ms=max(itl),
                E2E_p95_ms=percentile([r['e2e_ms'] for r in rows], .95),
                response_window_s=window['end'] - window['start'],
                probe_correctness=verify(name)))
        write(ROOT / 'state/probe_admission_clean_comparison.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
