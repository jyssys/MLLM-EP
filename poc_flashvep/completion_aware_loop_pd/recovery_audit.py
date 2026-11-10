"""Audit matched recovery starts without merging them with the primary results."""
from .common import *
from .serving_audit import inspect,paired,csv_write


def main():
    configure()
    confirmation_names=json.loads((ROOT/'recovery_plan.json').read_text())['jobs']
    names=list(confirmation_names)
    names += [f'secondary_{m}_{a}_selection32' for m in ('ouro','huginn')
        for a in ('native','loop','token')]
    items=[]
    requests=[]
    completed=[]
    for name in names:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        row,req,loans=inspect(name)
        row['secondary_arm']=cfg['secondary_arm']
        row['cohort']='N128_CONFIRMATION' if name in confirmation_names else 'N32_SELECTION'
        row['pool_blocks']=cfg.get('D_kv_blocks')
        row['no_measured_packed_graph_capture']=cfg.get('no_measured_graph_capture')
        items.append(row)
        requests.extend(req)
        completed.extend(loans)
    write(ROOT/'recovery_serving_results.json',items)
    csv_write(ROOT/'recovery_serving_results.csv',items)
    write(ROOT/'recovery_request_inventory.json',requests)
    csv_write(ROOT/'recovery_request_inventory.csv',requests)
    write(ROOT/'recovery_completion_audit.json',completed)
    csv_write(ROOT/'recovery_completion_audit.csv',completed)
    ratios=paired([r for r in items if r['cohort']=='N128_CONFIRMATION'])
    write(ROOT/'recovery_paired_start_ratios.json',ratios)
    csv_write(ROOT/'recovery_paired_start_ratios.csv',ratios)
    print(json.dumps(dict(complete=sum(r.get('complete',False) for r in items),
        accounted_requests=len(requests),loans=len(completed))))


if __name__=='__main__':
    main()
