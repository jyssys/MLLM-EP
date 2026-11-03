"""Distinguish model/graph collection failures from a transport requirement."""
import copy
import tarfile
from collections import Counter
from .common import *
from .prepare import put
from .run import execute
from .analysis_lane import quiet_analysis


def main():
    ledger = []
    for model in ('huginn', 'ouro'):
        source = f'capacity_bulk_written_{model}_matched'
        cfg_path = ROOT/'configs'/(source+'.json')
        if not cfg_path.exists():
            continue
        for graph in (True, False):
            cfg = copy.deepcopy(json.loads(cfg_path.read_text()))
            cfg.update(name=f'collector_no_transport_{model}_graph{int(graph)}',
                phase='collector_no_transport_diagnostic', cuda_only_profile=True,
                capture_role='prefill_decode', capture_roles=['prefill_decode'],
                P_only_profiler=False, nsys=False, native_capture_seconds=0.,
                bounded_native_capture_seconds=0., autonomous_capture_seconds=0.,
                postwarm_capture=False, capture_seconds=0., timeout=600,
                jobs=[dict(batch=2, cap=1, repeats=12)])
            if not graph:
                cfg['P_prefill_graph_rows'] = 0
            name = put(cfg, 2, split='confirmation')
            if time.time() >= LONG_START_CUTOFF:
                break
            write(ROOT/'state/collector_no_transport_control.json', dict(stage='RUNNING', name=name, time=time.time()))
            row = dict(name=name, model=model, graph=graph, complete=execute(name),
                scope='Real model and valid KV, local barrier without NIXL. Diagnostic profiling, not clean capacity or PD headroom.')
            if row['complete']:
                with quiet_analysis(name+'_collector_counts'):
                    counts = Counter()
                    for archive_path in (ROOT/'raw'/name/'cuda_only_profile').glob('*/rank-*/cycle-*.tar.gz'):
                        with tarfile.open(archive_path) as bundle:
                            member = next(m for m in bundle.getmembers() if m.name.endswith('/trace.json') or m.name == 'trace.json')
                            raw = json.load(bundle.extractfile(member))
                        counts.update(e.get('cat', 'metadata') for e in raw['traceEvents'])
                    row['collector_event_counts'] = dict(counts)
                    row['kernel_activity_retained'] = counts['kernel'] > 0
            ledger.append(row)
            write(ROOT/'state/collector_no_transport_results.json', ledger)
    write(ROOT/'state/collector_no_transport_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/collector_no_transport_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
