"""Retain every A/B/A outcome and compare the intervening zero-offload path."""
from .common import *
from .serving_audit import inspect,csv_write


def main():
    configure()
    plan=json.loads((ROOT/'state/late_ABA_plan.json').read_text())
    rows=[]
    for name in plan['jobs']:
        row,_,_=inspect(name)
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        row['secondary_arm']=cfg['late_control_arm'];rows.append(row)
    result=dict(plan=plan,runs=rows,
        scope='One fixed arrival trace in A/B/A order. This controls a simple temporal-drift explanation. It does not isolate individual Runtime components.')
    if all(r.get('complete') for r in rows):
        keys=('TTFT_p95_ms','ITL_p99_ms','maximum_ITL_ms','TPOT_p95_ms','completed_QPS','SLO_goodput')
        result['B_over_mean_A']={k:rows[1][k]/((rows[0][k]+rows[2][k])/2)
            if rows[0][k]+rows[2][k] else None for k in keys}
        result['A2_over_A1']={k:rows[2][k]/rows[0][k] if rows[0][k] else None for k in keys}
    write(ROOT/'late_ABA_results.json',result);csv_write(ROOT/'late_ABA_results.csv',rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('plan','runs')}))


if __name__=='__main__':
    main()
