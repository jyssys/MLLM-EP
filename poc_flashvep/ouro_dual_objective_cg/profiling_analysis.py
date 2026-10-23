"""Summarize event-audit paths without converting service sums to E2E shares."""
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    groups=defaultdict(list)
    for folder in sorted((ROOT/'raw').glob('dual_p2_EVENT_*')):
        if not (folder/'completed.json').exists():continue
        for file in folder.glob('*_OBS_*.json'):
            d=json.loads(file.read_text())
            if not isinstance(d,dict) or 'service_observations' not in d:continue
            for q in d['service_observations']:
                phases=sorted({s['phase'] for s in q['spans'] if s['phase'].startswith('prefill')})
                gs=sorted({s['G'] for s in q['spans'] if s['phase'].startswith('prefill')})
                key=(folder.name,d['label'],d['configured_N'],d['cell'].get('budget_ms'),','.join(phases),','.join(map(str,gs)))
                groups[key].append(q)
    rows=[]
    for key,vs in groups.items():
        service=[v['service_ms'] for v in vs]
        rows.append(dict(zip(('run_id','policy','width','B_ms','phases','actual_G'),key),units=len(vs),
            service_P50_ms=float(np.percentile(service,50)),service_P95_ms=float(np.percentile(service,95)),
            service_P99_ms=float(np.percentile(service,99)),
            staged_allocated_MiB=json.dumps(sorted({v['staged_bytes']/2**20 for v in vs})),
            graph_captures_inside_observed_units=sum(v['graph_after'].get('captures',0)-v['graph_before'].get('captures',0) for v in vs),
            graph_replays_inside_observed_units=sum(v['graph_after'].get('replays',0)-v['graph_before'].get('replays',0) for v in vs),
            graph_fallbacks_inside_observed_units=sum(v['graph_after'].get('fallbacks',0)-v['graph_before'].get('fallbacks',0) for v in vs),
            ready_before_issue_false=sum(not all(v['ready_before_issue'].values()) for v in vs),
            mean_host_submit_ms=float(np.mean([v['host_dispatch_ms'] for v in vs])),
            scope='LIVE_MEASURED_OBSERVER_READY_STREAM_SERVICE. Includes preparation/host gaps. No pure-kernel, HBM-traffic or additive E2E attribution.'))
    csv_file(ROOT/'event_path_profile.csv',rows)
    print('Event path profile cells',len(rows),'units',sum(r['units'] for r in rows),'capture increments',sum(r['graph_captures_inside_observed_units'] for r in rows))

if __name__=='__main__':main()
