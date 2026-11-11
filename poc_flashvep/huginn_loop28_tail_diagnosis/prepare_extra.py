"""Keep observer versions explicit and retain the first instrumented controls."""
from .common import *


def main():
    initialize();mapping={};jobs=[]
    for arm in ('zero','full'):
        original=f'tail_{arm}_events_n64';name=f'tail_{arm}_events_cached_n64'
        cfg=json.loads((ROOT/'configs'/(original+'.json')).read_text())
        cfg.update(name=name,diagnosis_observer_revision='CACHED_MEASUREMENT_MARKER_V2')
        save(cfg);mapping[original]=name;jobs.append(name)
    write(ROOT/'state/trace_versions.json',dict(replacements=mapping,time=time.time(),
        reason='The first two event cells use a per-readiness-call file-existence test. The revised observer caches the measurement marker. Repeat both event controls under the same observer as later arms. Retain all first-version results as measurement diagnostics.',
        scope='Measurement-only source revision. No model, request placement, CUDA graph, arithmetic, or SLO changes. Selection is by observer version, not latency.'))
    jobs+=json.loads((ROOT/'state/kernel_tail_plan.json').read_text())['jobs']
    write(ROOT/'state/extra_diagnostic_plan.json',dict(jobs=jobs))
    print(' '.join(jobs))


if __name__=='__main__':main()
