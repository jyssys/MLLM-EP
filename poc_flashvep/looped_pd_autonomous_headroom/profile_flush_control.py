"""Separate collector buffering and graph-node overhead from serving failures."""
import copy
from .common import *
from .prepare import put
from .run import execute


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    source = 'graph_pressure_huginn_factor110_nsys_postwarm_stats_handshake'
    original = json.loads((ROOT/'configs'/(source+'.json')).read_text())
    results = []
    for granularity in ('node', 'graph'):
        cfg = copy.deepcopy(original)
        cfg.update(name=f'huginn_postwarm_flush0_{granularity}_diagnostic',
                   nsys_cuda_flush_ms=0, nsys_graph_trace=granularity)
        name = put(cfg, 24, cfg['arrival_rate'], split='confirmation')
        write(ROOT/'state/profile_flush_control.json', dict(stage='RUNNING', name=name, time=time.time()))
        row = dict(name=name, complete=execute(name), graph_granularity=granularity,
            source='Installed nsys profile --help: zero flush interval saves filled buffers. Graph-level tracing omits node activities.',
            scope='Collector diagnostic. Graph-level intervals are not kernel-union measurements.')
        if row['complete']:
            row['parity'] = evidence.compare(cfg['matched_clean'], name)
            if granularity == 'node':
                row['profile'] = analyze(name, cfg['matched_clean'])
            else:
                row['actual_kernel_union'] = 'NOT_MEASURED_GRAPH_NODES_DISABLED'
        results.append(row)
        write(ROOT/'state/profile_flush_control_results.json', results)
    write(ROOT/'state/profile_flush_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/profile_flush_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
