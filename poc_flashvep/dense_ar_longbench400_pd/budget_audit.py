"""Check actual query batches instead of trusting the printed B/S options."""
import json
import time
from .common import ROOT,START,write


def main():
    rows=[]
    failed=[]
    pending=[]
    allocation=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        config=folder/'configuration.json'
        if not config.exists():
            continue
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists():
            pending.append(folder.name)
            continue
        dispatcher=json.loads(receipt.read_text())
        cfg=json.loads(config.read_text())
        devices=1 if cfg.get('reference_single') else len(cfg['gpus'])
        allocation.append(dict(name=folder.name,complete=dispatcher['complete'],GPU_count=devices,
            worker_wall_seconds=dispatcher['elapsed_seconds'],
            worker_wall_GPU_hours=devices*dispatcher['elapsed_seconds']/3600))
        if not dispatcher['complete']:
            failed.append(folder.name)
            continue
        roles=[]
        for path in sorted(folder.glob('actual_queries_*.json')):
            queries=json.loads(path.read_text())['rows']
            maximum_queries=max((r['actual_batch_queries'] for r in queries),default=0)
            maximum_requests=max((r['requests'] for r in queries),default=0)
            ready=json.loads((folder/('actual_runtime_'+path.stem.removeprefix('actual_queries_')+'.json')).read_text())
            passed=ready['scheduler_B']==cfg['token_budget']==8192
            passed=passed and ready['scheduler_S']==cfg['max_num_seqs']==1024
            passed=passed and maximum_queries<=8192 and maximum_requests<=1024
            roles.append(dict(role=path.stem.removeprefix('actual_queries_'),passed=passed,
                query_records=len(queries),maximum_actual_queries=maximum_queries,
                maximum_actual_requests=maximum_requests,
                maximum_padded_queries=max((r['padded_queries'] for r in queries),default=0)))
        rows.append(dict(name=folder.name,method=cfg['method'],correctness_only=cfg.get('correctness_only',False),
            role_checks=roles,passed=bool(roles) and all(r['passed'] for r in roles)))
    invalid=[r for r in rows if not r['passed']]
    reference=[]
    for folder in sorted((ROOT/'raw').glob('CandidateNaturalReference_*')):
        for command in sorted(folder.glob('command_GPU*.json')):
            c=json.loads(command.read_text());status=folder/('GPU'+str(c['GPU']))/'status.json'
            if not status.exists():continue
            s=json.loads(status.read_text())
            if s['status'] not in ('COMPLETE','FAILED'):continue
            elapsed=s['time']-c['time'];assert elapsed>=0
            reference.append(dict(run=folder.name,GPU=c['GPU'],status=s['status'],
                command_to_terminal_generation_status_seconds=elapsed,GPU_hours=elapsed/3600))
    value=dict(time=time.time(),passed=not invalid,completed_starts_checked=len(rows),
        rows=rows,invalid=invalid,failed_not_qualified=failed,pending=pending,
        allocation_accounting=dict(standard_dispatcher_starts=len(allocation),
            standard_dispatcher_worker_wall_GPU_hours=sum(r['worker_wall_GPU_hours'] for r in allocation),
            standard_dispatcher_rows=allocation,natural_reference_shards=reference,
            natural_reference_status_enclosure_GPU_hours=sum(r['GPU_hours'] for r in reference),
            campaign_wall_elapsed_hours=(time.time()-START)/3600,
            time_limit_removed=json.loads((ROOT/'state/budget.json').read_text()).get('time_limit_removed',False),
            scope='Dispatcher wall time includes initialization,measurement and retirement. Multiply by resident device count,not by kernel busy. '
                'It excludes pair-lock waiting and burner execution. Natural-reference shard bounds use command and terminal status timestamps. '
                'They do not include every shutdown tail or custom preflight. This is partial resource accounting,not exclusive GPU compute time or billing.'),
        evidence='RUNTIME_CONFIGURATION_AND_QUERY_AUDIT',
        scope='Check printed Worker configuration and every logged actual query batch in completed starts. '
            'B8192/S1024 remains an upper bound,not actual batch geometry. Padded queries are reported separately. '
            'A budget pass does not qualify teacher numerics,task quality,observer overhead or serving safety. '
            'Failed and unfinished runs remain separate. Repeat after GPU closeout for the final denominator.')
    write(ROOT/'analysis/scheduler_actual_budget_audit_v1.json',value)
    print('ACTUAL_BUDGET_AUDIT',len(rows),'checked',len(invalid),'invalid',len(failed),'failed',len(pending),'pending',flush=True)
    assert not invalid,invalid[:3]


if __name__=='__main__':
    main()
