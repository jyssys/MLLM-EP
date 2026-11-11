"""Require observed destination quantum bounds, not configuration intent."""
import argparse
from collections import Counter
from .common import *


def audit(name):
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    folder = ROOT/'raw'/name
    begin = json.loads((folder/'measurement_begin.json').read_text())['time']
    spans = [row for path in folder.glob('atomic_spans_*.json')
             for owner in [json.loads(path.read_text())] if owner['role'] == 'decode'
             for row in owner['prompt_spans'] if begin <= row['time']
             and 0 <= row['seed']-cfg['seed'] < cfg['count']]
    histogram = Counter((row['phase'], row['count'], row['loop_count']) for row in spans)
    loop_limit = cfg.get('D_prefill_loop_cap', cfg['depth'])
    token_limit = cfg.get('D_prefill_token_cap', cfg['length'])
    violations = [row for row in spans if row['loop_count'] > loop_limit or row['count'] > token_limit]
    coverage = len({row['seed'] for row in spans if row['loop_count'] > 0})
    result = dict(name=name, measured_prompt_spans=len(spans),
        request_coverage=coverage, expected_requests=cfg['count'],
        loop_limit=loop_limit, token_limit=token_limit, violations=len(violations),
        histogram=[dict(phase=phase, rows=rows, loops=loops, spans=count)
                   for (phase,rows,loops),count in sorted(histogram.items())],
        passed=coverage == cfg['count'] and not violations,
        scope='Observed D prompt spans with measured sampling seeds. This checks the submitted work quantum, not GPU kernel preemption or execution time.')
    target = ROOT/'audit'/'quantum'
    target.mkdir(parents=True, exist_ok=True)
    write(target/(name+'.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    initialize()
    for name in args.names:
        print(json.dumps(audit(name)))


if __name__ == '__main__':
    main()
