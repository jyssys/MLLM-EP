"""Aggregate real replicas from their common offered clock and immutable IDs."""
from collections import defaultdict
import json
import numpy as np
from .common import ROOT,write
from .analyze import csvfile

def main():
    groups=defaultdict(list)
    for path in (ROOT/'raw').glob('dp_*_r[12345678]/dp_result.json'):
        if (path.parent/'completed.json').exists():
            row=json.loads(path.read_text());row['job']=path.parent.name
            groups[row['cluster']].append(row)
    table=[];requests=[];parity=[];replica_rows=[]
    for cluster,replicas in sorted(groups.items()):
        width=replicas[0]['DP']
        if len(replicas)!=width:continue
        assert len(set(r['start_wall_s'] for r in replicas))==1
        start=replicas[0]['start_wall_s']
        cases=[dict(r,gpu=replica['gpu']) for replica in replicas for r in replica['requests']]
        assert len(cases)==32 and len(set(r['id'] for r in cases))==32
        for r in cases:assert int(r['id'].split('_')[1])%width==int(r['gpu'])
        end=max(r['finish_wall_s'] for r in replicas)
        for replica in replicas:
            replica_rows.append(dict(cluster=cluster,model=replica['model'],arm=replica['arm'],
                condition=replica.get('condition','LIBRARY_WARM_ONLY'),DP=width,gpu=replica['gpu'],
                request_count=len(replica['requests']),output_tokens=replica['output_tokens'],
                replica_finish_s=replica['finish_wall_s']-start,
                cluster_tail_after_replica_s=end-replica['finish_wall_s'],peak_HBM=replica['peak_HBM']))
        gaps=[(b-a)*1000 for r in cases for a,b in zip(r['times'],r['times'][1:])]
        record=dict(cluster=cluster,model=replicas[0]['model'],arm=replicas[0]['arm'],
            condition=replicas[0].get('condition','LIBRARY_WARM_ONLY'),
            DP=width,offered=32,completed=sum(r['reason'] in ['stop','chat_stop'] for r in cases),
            capped=sum(r['reason']=='length' for r in cases),output_tokens=sum(len(r['tokens']) for r in cases),
            makespan_s=end-start,requests_per_s=32/(end-start),
            tokens_per_s=sum(len(r['tokens']) for r in cases)/(end-start),
            TTFT_p50_ms=float(np.percentile([r['TTFT_s']*1000 for r in cases],50)),
            TTFT_p95_ms=float(np.percentile([r['TTFT_s']*1000 for r in cases],95)),
            E2E_p95_ms=float(np.percentile([r['E2E_s']*1000 for r in cases],95)),
            ITL_p99_ms=float(np.percentile(gaps,99)) if gaps else None,
            peak_HBM_max=max(r['peak_HBM'] for r in replicas),
            evidence='LIVE_MEASURED_DP_FINITE_BURST',independent_cluster_starts=1,
            capacity_qualified=False)
        record['requests_per_s_per_GPU']=record['requests_per_s']/width
        table.append(record)
        for r in cases:requests.append(dict(cluster=cluster,model=record['model'],arm=record['arm'],DP=width,**r))
    for row in table:
        reference=next((r for r in table if r['model']==row['model'] and r['arm']=='mixed' and r['DP']==1
            and r['condition']==row['condition']),None)
        if reference:row['makespan_speedup_vs_DP1_native']=reference['makespan_s']/row['makespan_s']
        else:row['makespan_speedup_vs_DP1_native']=None
    for model in ['ouro','huginn']:
        for width in [1,2,4,8]:
            for block in ['r1','r2','r3','r4','r5','r6','r7','r8']:
                a={r['id']:r for r in requests if r['model']==model and r['DP']==width and r['arm']=='mixed' and r['cluster'].endswith(block)}
                b={r['id']:r for r in requests if r['model']==model and r['DP']==width and r['arm']=='hybrid' and r['cluster'].endswith(block)}
                if not a or not b:continue
                assert a.keys()==b.keys()
                for rid in a:parity.append(dict(model=model,DP=width,block=block,id=rid,
                    tokens_exact=a[rid]['tokens']==b[rid]['tokens'],
                    length_exact=len(a[rid]['tokens'])==len(b[rid]['tokens']),reason_exact=a[rid]['reason']==b[rid]['reason']))
    csvfile('dp_scaleout.csv',table);csvfile('dp_request_metrics.csv',requests);csvfile('dp_output_parity.csv',parity)
    csvfile('dp_replica_metrics.csv',replica_rows)
    write(ROOT/'dp_summary.json',dict(clusters=len(table),offered=sum(r['offered'] for r in table),
        completed=sum(r['completed'] for r in table),capped=sum(r['capped'] for r in table),
        paired_outputs=len(parity),exact_outputs=sum(r['tokens_exact'] for r in parity)))
    print('DP_CLUSTERS',len(table),'OFFERED',sum(r['offered'] for r in table))

if __name__=='__main__':main()
