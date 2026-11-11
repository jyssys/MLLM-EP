"""Compare each event diagnostic with its matched marker-free control."""
import argparse
from .common import *
from .cost_analysis import summarize


METRICS = ('TTFT_p95_ms', 'E2E_p95_ms', 'TPOT_p95_ms', 'ITL_p99_ms',
           'maximum_ITL_ms', 'completed_QPS')
FLAGS = {'name', 'phase', 'sparse_device_trace', 'sparse_decode_period',
         'host_cost_trace', 'profile_python', 'clean_confirmation',
         'matched_marker_free_control'}


def compare(name, threshold=.05):
    initialize()
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    control = cfg['matched_marker_free_control']
    original = json.loads((ROOT/'configs'/(control+'.json')).read_text())
    differing = sorted(key for key in cfg.keys() | original.keys()
                       if key not in FLAGS and cfg.get(key) != original.get(key))
    a,b = [analysis().metrics(n) for n in (control,name)]
    ratios = {key:b[key]/a[key] for key in METRICS
              if a.get('complete') and b.get('complete') and a.get(key) and b.get(key) is not None}
    passes = len(ratios) == len(METRICS) and not differing and all(
        abs(value-1.) <= threshold for value in ratios.values())
    result = dict(diagnostic=name, marker_free=control, compared_configuration_differences=differing,
        ratios_diagnostic_over_marker_free=ratios, allowed_absolute_relative_change=threshold,
        single_start_interference_screen_passed=passes,
        disposition=('EVENT_ENCLOSURE_DIAGNOSTIC_ONLY' if passes else 'INTERFERENCE_OR_COMPLETION_FAIL'),
        scope=('This is one matched Arrival trace, not a confidence bound. CUDA Event '
            'enclosures are not physical kernel unions even when this screen passes. '
            'Do not use the diagnostic as a marker-free serving result.'))
    (ROOT/'costs').mkdir(exist_ok=True)
    if b.get('complete'):
        summarize(name)
    write(ROOT/'costs'/(name+'_interference.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    for name in args.names:
        print(json.dumps(compare(name)))


if __name__ == '__main__':
    main()
