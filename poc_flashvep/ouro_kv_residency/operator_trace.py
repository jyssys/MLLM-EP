"""Export an actual two-GPU attention timeline without adding rank service times."""
import csv
import gzip
import json
import sqlite3
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write

def main():
    database=ROOT/'profiles/distributed_trace_v1_P2_s0.sqlite'
    db=sqlite3.connect(database);names=dict(db.execute('select id,value from StringIds'))
    kernels=db.execute('select start,end,deviceId,streamId,globalPid,demangledName,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    identities=db.execute('select id,uuid,busLocation from TARGET_INFO_GPU').fetchall()
    assert {r[2] for r in kernels}=={0,1}
    origin=min(r[0] for r in kernels);counts=defaultdict(int);durations=defaultdict(float)
    def family(name):
        if 'nccl' in name.lower():return 'NCCL_INCLUDING_WAIT'
        if 'flash_attn' in name.lower():return 'LOCAL_FA4'
        return 'PACK_AND_MERGE'
    with gzip.open(ROOT/'profiles/distributed_operator_trace.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['start_ns','end_ns','device','stream','globalPid','graphNodeId','family','name'])
        for s,e,gpu,st,pid,name,node in kernels:
            f=family(names[name]);writer.writerow([s-origin,e-origin,gpu,st,pid,node,f,names[name]])
            counts[(gpu,f)]+=1;durations[(gpu,f)]+=(e-s)/1e6
    write(ROOT/'distributed_trace_summary.json',dict(devices=identities,kernels=len(kernels),
        rows=[dict(device=g,family=f,kernels=n,service_ms=durations[(g,f)]) for (g,f),n in sorted(counts.items())],
        evidence='LIVE_MEASURED_OPERATOR',scope='Heavy trace. NCCL service includes waiting. Rank sums are not E2E latency.'))
    fig,ax=plt.subplots(figsize=(12,3.2));colors={'LOCAL_FA4':'#2864ad','NCCL_INCLUDING_WAIT':'#da7d24','PACK_AND_MERGE':'#3b9767'}
    for gpu in (0,1):
        for f,color in colors.items():
            values=[((s-origin)/1e6,(e-s)/1e6) for s,e,g,st,pid,name,node in kernels if g==gpu and family(names[name])==f and (s-origin)/1e6<1.5]
            ax.broken_barh(values,(gpu+.1,.6),facecolors=color,label=f if gpu==0 else None)
    ax.set_yticks([.4,1.4],['Physical GPU0','Physical GPU1']);ax.set_xlim(0,1.5)
    ax.set_xlabel('ms from first captured GPU kernel')
    ax.set_title('Actual CP2 graph: local FA4 + real NCCL + merge. P32769/B1. Heavy trace, not model E2E.')
    ax.legend(fontsize=8,loc='upper right');fig.tight_layout()
    (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/distributed_operator_Gantt.png',dpi=180);fig.savefig(ROOT/'figures/distributed_operator_Gantt.svg');plt.close(fig)
    components=[]
    for group in (ROOT/'distributed').glob('components_v1*'):
        ranks=[json.loads(p.read_text()) for p in sorted(group.glob('rank*_results.json'))]
        if not ranks:continue
        for rows in zip(*ranks):
            r=rows[0];c=r['components']
            if c is None:continue
            components.append(dict(run=group.name,P=r['P'],L=r['L'],B=r['B'],placement=r['placement'],
                full_wall_ms=max(x['graph']['wall_median_ms'] for x in rows),
                local_partial_floor_wall_ms=max(x['components']['local_partial_graph']['wall_median_ms'] for x in rows),
                communication_merge_floor_wall_ms=max(x['components']['communication_merge_graph']['wall_median_ms'] for x in rows),
                scope='Separate graph floors. Do not sum into an exclusive critical path.',evidence='LIVE_MEASURED_OPERATOR'))
    with (ROOT/'operator_component_floors.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(components[0]));writer.writeheader();writer.writerows(components)

if __name__=='__main__':main()
