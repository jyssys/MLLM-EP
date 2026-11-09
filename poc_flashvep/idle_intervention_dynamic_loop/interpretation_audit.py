"""Separate empirical model checks from unmeasured device safety claims."""
from collections import Counter
from .common import *
from .analyze import percentile
from .campaign_report import event_records,write_csv


def quantiles(values):
    return dict(count=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
        p99=percentile(values,.99),maximum=max(values,default=None))


def main():
    rows=[]
    for row in json.loads((ROOT/'serving_runs.json').read_text()):
        if not row['complete'] or row.get('evidence_class')!='MARKER_FREE':continue
        if not (row['name'].startswith(('confirm_','r1_','r6_')) and 'state' not in row['name']):continue
        folder=ROOT/'raw'/row['name']
        begin=json.loads((folder/'measurement_begin.json').read_text())['time']
        events=event_records(folder,begin)
        submitted=[r for r in events if r['event']=='D_QUANTUM_SUBMITTED' and r.get('choice')]
        completed=[r for r in events if r['event']=='D_QUANTUM_HOST_COMPLETION'
            and r.get('timing_revision')==2 and r.get('choice') and r['choice'].get('cost_ms')]
        ratios=[r['elapsed_host_ms']/r['choice']['cost_ms'] for r in completed]
        before_budget=[r for r in completed if r['choice'].get('budget_at_selection_ms') is not None]
        P=[r for r in events if r['event']=='SUPPLY_SAMPLE' and r.get('role')=='prefill']
        D=[r for r in events if r['event']=='SUPPLY_SAMPLE' and r.get('role')=='decode']
        decode_intervals=[r for r in events if r['event']=='DECODE_INTERVAL_SAMPLE']
        batches=sorted({r['batch'] for r in decode_intervals})
        rows.append(dict(name=row['name'],model=row['model'],method=row['method'],
            actual_offloads=row.get('offloaded_requests'),
            actual_grouped_loop_counts=dict(Counter(r['loops'] for r in submitted)),
            actual_submitted_token_widths=dict(Counter(r['rows'] for r in submitted)),
            quanta_with_ready_Decode=sum(r['choice'].get('decode_ready_at_selection',0)>0 for r in submitted),
            host_enclosure_over_candidate_cost=sum(value>1 for value in ratios),
            host_to_modeled_quantum_ratio=quantiles(ratios),
            host_enclosure_over_selection_budget=sum(r['elapsed_host_ms']>r['choice']['budget_at_selection_ms'] for r in before_budget),
            execution_budget_samples=quantiles([r['predicted_budget_ms'] for r in D]),
            P_executable_ready_samples=quantiles([r.get('executable_P_ready',0) or 0 for r in P]),
            P_pending_device_request_samples=quantiles([r.get('pending_P_device_requests',0) or 0 for r in P]),
            minimum_D_free_blocks=min((r['KV_free_blocks'] for r in D),default=None),
            Decode_delivery_intervals_by_batch={str(batch):quantiles([r['interval_ms'] for r in decode_intervals if r['batch']==batch]) for batch in batches},
            scope='Host enclosures include submission and polling. A cost or selection-budget overrun is not a measured GPU deadline miss. Sampled budgets are predictions, not physical idle durations. Ready Decode may wait within the predicted margin.'))
    write(ROOT/'interpretation_audit.json',rows)
    write_csv(ROOT/'interpretation_audit.csv',rows)
    print(json.dumps(dict(runs=len(rows))))


if __name__=='__main__':main()
