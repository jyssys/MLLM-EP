"""Keep original, retained-graph, and exact-primary-policy evidence separate."""
from collections import defaultdict
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_looped_recurrence_tp_4models.contract import *
from poc_looped_recurrence_tp_4models.analysis import read,table,bootstrap


def main():
    root=ROOT/'results/rechecks';by_config={};requests=[];checks=[]
    for directory in sorted(root.glob('*')):
        if not (directory/'status.json').exists() or not read(directory/'status.json').get('success'):continue
        row=read(directory/'results.json');groups=defaultdict(list)
        for record in row['results']:groups[record['request_id']].append(record)
        values={}
        for request,records in groups.items():
            value=dict(config=directory.name,request_id=request,model=row['model'],R=row['R'],TP=row['TP'],family=row['family'],
                technical_repeats=len(records),E2E_ms=statistics.mean(r['wall_ms'] for r in records),
                prefill_ms=statistics.mean((r['metrics']['first_token_ts']-r['metrics']['scheduled_ts'])*1000 for r in records),
                TPOT_ms=statistics.mean((r['metrics']['last_token_ts']-r['metrics']['first_token_ts'])*1000/63 for r in records))
            values[request]=value;requests.append(value)
        by_config[directory.name]=values
        checks.append(dict(config=directory.name,exact_engine_kwargs_match=read(directory/'engine_contract.json')==read(ROOT/'results/raw'/directory.name/'engine_contract.json'),
            scope='Independent reverse-order model session. Original calibration requests. Not fresh held-out.'))
    comparisons=[]
    for name,rows in by_config.items():
        if '_tied_' not in name:continue
        native=by_config.get(name.replace('_tied_','_native_'))
        if not native:continue
        first=next(iter(rows.values()))
        for metric in ['prefill_ms','TPOT_ms','E2E_ms']:
            ratios=[native[request][metric]/r[metric] for request,r in rows.items()]
            differences=[(1-r[metric]/native[request][metric])*100 for request,r in rows.items()]
            comparisons.append(dict(config=name,R=first['R'],TP=first['TP'],prefix=1024,metric=metric,
                NativeVsUnrolledRatio=bootstrap(ratios),observed_tied_reduction_percent=bootstrap(differences),
                scope='Exact-primary-policy reverse-order recheck. Independent sessions, same eight calibration requests.'))
    table(ROOT/'results/same_policy_recheck_requests.csv',requests)
    write(ROOT/'results/same_policy_recheck_comparisons.json',comparisons);write(ROOT/'results/same_policy_recheck_contracts.json',checks)
    if not comparisons:return
    import csv
    with (ROOT/'results/paired_comparisons.csv').open() as stream:original=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,metric in zip(axes,['prefill_ms','TPOT_ms','E2E_ms']):
        for i,(R,TP) in enumerate([(4,1),(2,4)]):
            old=next(r for r in original if r['model']=='ouro' and r['comparison']=='NativeVsUnrolledRatio' and
                int(r['R'])==R and int(r['TP'])==TP and int(r['prefix'])==1024 and r['metric']==metric)
            new=next((r for r in comparisons if r['R']==R and r['TP']==TP and r['metric']==metric),None)
            for offset,stats,label,color in [(-.08,{k:float(old[k]) for k in ['mean','low','high']},'Original calibration','tab:blue'),
                (.08,None if new is None else new['NativeVsUnrolledRatio'],'Same-policy reverse-order recheck','tab:orange')]:
                if stats is None:continue
                ax.errorbar(i+offset,stats['mean'],yerr=[[stats['mean']-stats['low']],[stats['high']-stats['mean']]],
                    marker='o',capsize=3,color=color,label=label if i==0 else None)
        ax.axhline(1,color='k',linestyle='--');ax.set(title=metric,xticks=[0,1],xticklabels=['R4 TP1','R2 TP4'],ylabel='Native / Tied')
        lo,hi=ax.get_ylim();ax.set_ylim(min(.9,lo),max(1.1,hi));ax.legend(fontsize=7)
    fig.suptitle('1K discrepancy audit. Same calibration requests, not untouched confirmation.');fig.tight_layout()
    fig.savefig(ROOT/'figures/short_prefill_same_policy_rechecks.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print('Recheck configs',len(checks),'paired metrics',len(comparisons))


if __name__=='__main__':main()
