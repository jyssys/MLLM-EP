"""Measure host queue states on the offered-arrival interval,not the drain."""
import argparse
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .c3_analysis import actual_quality,means_ci
from .c0_analysis import csv_save

FIELDS=('waiting_total','running','active_decode','decode_ready_waiting','decode_ready_including_first_query',
    'remote_KV_waiting','free_KV_blocks')

def weighted_states(rows,left,right,field):
    ordered=sorted(rows,key=lambda r:r['time']);values=[];weights=[]
    for index,row in enumerate(ordered):
        a=max(left,row['time']);b=min(right,ordered[index+1]['time'] if index+1<len(ordered) else right)
        if b>a and isinstance(row.get(field),(int,float)):
            values.append(row[field]);weights.append(b-a)
    if not values:return None
    values=np.asarray(values,float);weights=np.asarray(weights,float);order=np.argsort(values)
    v=values[order];w=weights[order];total=weights.sum()
    quantile=float(v[min(int(np.searchsorted(np.cumsum(w),.95*total)),len(v)-1)])
    grid=np.linspace(left+.1*(right-left),left+.9*(right-left),101)
    times=np.array([r['time'] for r in ordered]);index=np.searchsorted(times,grid,side='right')-1
    sampled=np.array([ordered[max(i,0)].get(field,np.nan) for i in index],float)
    valid=(index>=0)&np.isfinite(sampled)
    slope=float(np.polyfit(grid[valid]-left,sampled[valid],1)[0]) if valid.sum()>1 else None
    change=float(sampled[valid][-1]-sampled[valid][0]) if valid.any() else None
    growth='BACKLOG_GROWTH_OBSERVED' if slope is not None and slope>.05 and change>3 else 'NO_GROWTH_DETECTED_IN_THIS_FINITE_TRACE'
    return dict(time_weighted_mean=float(np.average(values,weights=weights)),time_weighted_p95=quantile,maximum=float(values.max()),
        field_units='KV blocks' if field=='free_KV_blocks' else 'requests',
        covered_seconds=float(total),requested_seconds=right-left,central_10_to_90percent_slope_units_per_second=slope,
        central_endpoint_change_units=change,finite_trace_growth='NOT_APPLICABLE_TO_FREE_MEMORY_BLOCKS' if field=='free_KV_blocks' else growth)

def analyze(plan_name):
    plan=json.loads((ROOT/'jobs'/plan_name).read_text());records=[]
    for block in plan['blocks']:
        for label,name in block['arms'].items():
            folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
            window=json.loads((folder/'window.json').read_text());left=window['start'];right=window['last_arrival']
            assert right>left
            quality=actual_quality(name)['passed']
            for role in ('prefill','decode','replica0','replica1'):
                path=folder/f'scheduler_queues_{role}.json'
                if not path.exists():continue
                rows=json.loads(path.read_text())['rows']
                fields={field:weighted_states(rows,left,right,field) for field in FIELDS}
                records.append(dict(name=name,label=label,QPS=block['qps'],split=block['split'],start=block['start'],role=role,
                    host_queue_fields=fields,quality_passed=quality,arrival_interval_seconds=right-left,
                    drain_seconds=window['end']-right,evidence='TRACE_ANALYSIS' if quality else 'DIAGNOSTIC_QUALITY_FAILURE'))
    result=dict(time=time.time(),plan_SHA256=digest(ROOT/'jobs'/plan_name),records=records,
        scope='Piecewise-constant scheduler observations. State changes and50ms snapshots are not GPU activity. '
            'The analysis excludes drain from the primary queue interval. Growth uses101 held-state samples on its central10–90%. '
            'No-growth in a finite trace is not proof of sustainable capacity. '
            'D decode-ready waiting excludes active decoders and remote-KV waiting. An empty queue does not establish a safe window. '
            'Replica queues in Agg are colocated prefill/decode states,not P/D-role queues.')
    stem=plan_name.removesuffix('.json')+'_host_queues_v2'
    write(ROOT/'analysis'/f'{stem}.json',result);csv_save(ROOT/'figures'/f'{stem}.csv',records);figures(result,stem)
    print('HOST_QUEUE_INTERVAL_TABLES',len(records),flush=True)

def figures(result,stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=(('prefill','waiting_total','P waiting'),('prefill','running','P running requests'),
        ('decode','waiting_total','D all waiting'),('decode','remote_KV_waiting','D remote-KV waiting'),
        ('decode','decode_ready_waiting','D decode-ready waiting'),('decode','active_decode','D active decode requests'))
    for split in sorted({r['split'] for r in result['records']}):
        fig,axes=plt.subplots(2,3,figsize=(14,8));table=[]
        for ax,(role,field,title) in zip(axes.flat,specs):
            rows=[r for r in result['records'] if r['split']==split and r['role']==role]
            for label in ('NativeStrong','Layer','StockToken','Hybrid','Deflection'):
                points=[]
                for qps in sorted({r['QPS'] for r in rows}):
                    group=[r for r in rows if r['QPS']==qps and r['label']==label and r['host_queue_fields'][field] is not None]
                    if not group:continue
                    value=means_ci([r['host_queue_fields'][field]['time_weighted_mean'] for r in group]);points.append((qps,value))
                    table.append(dict(split=split,label=label,QPS=qps,role=role,field=field,**value,
                        quality_passed=all(r['quality_passed'] for r in group),evidence='TRACE_ANALYSIS'))
                if points:
                    line=ax.plot([q for q,v in points],[v['mean'] for q,v in points],'o-',label=label,markersize=3)[0]
                    for q,value in points:
                        if value['CI95']:
                            lo,hi=value['CI95'];ax.errorbar(q,value['mean'],yerr=[[value['mean']-lo],[hi-value['mean']]],color=line.get_color(),capsize=2)
            ax.set(title=title,xlabel='Offered QPS (requests/s)',ylabel='Time-weighted requests');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'C3 {split} | host scheduler queues during arrivals')
        fig.text(.02,.015,'TRACE_ANALYSIS. Drain is excluded. Bars use independent starts when N≥3. All raw states and quality flags remain in the CSV.\n'
            'Running is not GPU busy. Decode-ready excludes active decode and remote-KV waiting. Queue zero is not a safe-window proof.',fontsize=8)
        fig.tight_layout(rect=(0,.08,1,.94));target=f'{stem}_{split}'
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{target}.{ext}',dpi=170)
        csv_save(ROOT/'figures'/f'{target}.csv',table);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='c3_main_serving_v2.json');args=p.parse_args();analyze(args.plan)
