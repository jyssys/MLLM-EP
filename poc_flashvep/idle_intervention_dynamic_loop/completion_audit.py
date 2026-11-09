"""Compare the admission model with observed import-to-first-output waiting."""
from .common import *
from .campaign_report import event_records,write_csv


def main():
    rows=[]
    for receipt in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        if not json.loads(receipt.read_text()).get('complete'):continue
        folder=receipt.parent;name=folder.name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        if cfg.get('state_probe') or cfg.get('observe_device_execution') or cfg.get('calibration_serial'):continue
        begin=json.loads((folder/'measurement_begin.json').read_text())['time'];events=event_records(folder,begin)
        admissions={(r['tid'],r['epoch']):r for r in events if r['event']=='ADMISSION'}
        commits=[r for r in events if r['event']=='SOURCE_COMMIT']
        stats=folder/'stats.json'
        if not stats.exists():continue
        value=json.loads(stats.read_text());metrics=value.get('metrics',[])
        mapping={r.get('transfer_id'):r for r in metrics}
        for commit in commits:
            tid=commit['tid'];ad=admissions.get((tid,commit['epoch']),{});record=mapping.get(tid,{})
            timings=record.get('timings',{});first=timings.get('first_token');received=timings.get('d_received')
            submissions=[r for r in events if r['event']=='D_QUANTUM_SUBMITTED' and any(s['tid']==tid for s in r['spans'])]
            costs=[r.get('choice',{}).get('cost_ms') for r in submissions if r.get('choice')]
            rows.append(dict(name=name,model=cfg['model'],tid=tid,seed=commit['seed'],
                remaining=commit['remaining'],token_split=commit.get('token_split',False),
                admission_quantum_forecast_ms=ad.get('future_quantum_budget_ms'),
                admission_capacity_model_ms=ad.get('admission_budget_ms'),
                serial_remaining_cost_model_ms=ad.get('remaining_cost_ms'),
                observed_quantum_submissions=len(submissions),
                chosen_quantum_cost_bounds_sum_ms=sum(costs) if costs and all(c is not None for c in costs) else None,
                import_to_first_output_host_ms=1000*(first-received) if first is not None and received is not None else None,
                cut_commit_to_first_quantum_host_ms=1000*(submissions[0]['time']-commit['time']) if submissions else None,
                first_to_last_quantum_submission_host_ms=1000*(submissions[-1]['time']-submissions[0]['time']) if submissions else None,
                scope='Observed host completion intervals include queues, Decode work, submission, and communication dependencies. Serial cost estimates and sums of guarded candidate costs are models, not measured GPU occupancy.'))
    write(ROOT/'completion_model_audit.json',rows);write_csv(ROOT/'completion_model_audit.csv',rows)
    print(json.dumps(dict(migrated_requests_with_audit=len(rows))))


if __name__=='__main__':main()
