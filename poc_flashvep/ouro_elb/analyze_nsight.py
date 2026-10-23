"""CUPTI service sums and interval unions stay distinct from clean E2E."""
import csv
import gzip
import json
import sqlite3
import subprocess
from collections import defaultdict
from .common import ROOT,write,digest

def merged(intervals):
    out=[]
    for start,end in sorted(intervals):
        if out and start<=out[-1][1]:out[-1][1]=max(end,out[-1][1])
        else:out.append([start,end])
    return out

def main():
    summaries=[];api_rows=[];kernel_rows=[];manifest=[];traces={}
    for D in (1,4):
        report=ROOT/'profiles'/f'p2_nsight_v1_D{D}_g0.nsys-rep'
        if not report.exists():continue
        database=report.with_suffix('.sqlite')
        if not database.exists():
            subprocess.run(['nsys','export','--type=sqlite','--output='+str(database),str(report)],check=True,stdout=subprocess.DEVNULL)
        c=sqlite3.connect(database)
        names=dict(c.execute('select id,value from StringIds'))
        kernels=c.execute('select start,end,streamId,demangledName,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
        copies=c.execute('select start,end,streamId from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
        api=c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME').fetchall()
        assert kernels
        start=min(s for s,e,*rest in kernels+copies);end=max(e for s,e,*rest in kernels+copies)
        union=merged([(s,e) for s,e,*rest in kernels+copies]);busy=sum(e-s for s,e in union)
        counters=defaultdict(lambda:[0,0])
        for s,e,name in api:counters[names[name]][0]+=1;counters[names[name]][1]+=e-s
        for name,(count,duration) in counters.items():api_rows.append(dict(D=D,API=name,calls=count,service_sum_ms=duration/1e6))
        kn=defaultdict(lambda:[0,0])
        for s,e,stream,name,node in kernels:kn[names[name]][0]+=1;kn[names[name]][1]+=e-s
        for name,(count,duration) in kn.items():kernel_rows.append(dict(D=D,kernel=name,calls=count,service_sum_ms=duration/1e6))
        summary=dict(D=D,context=256,N=1,kernels=len(kernels),GPU_memcopies=len(copies),
            kernel_service_sum_ms=sum(e-s for s,e,*rest in kernels)/1e6,
            GPU_extent_ms=(end-start)/1e6,GPU_kernel_copy_union_ms=busy/1e6,
            uncovered_extent_ms=(end-start-busy)/1e6,union_fraction=busy/(end-start),
            cudaGraphLaunch_calls=sum(v[0] for k,v in counters.items() if k.startswith('cudaGraphLaunch')),
            cudaEventSynchronize_calls=sum(v[0] for k,v in counters.items() if k.startswith('cudaEventSynchronize')),
            scope='NSIGHT_HEAVY_CAPTURE. Not clean latency or removable critical-path attribution.')
        summaries.append(summary);traces[D]=(start,kernels,copies)
        portable=ROOT/'profiles'/f'GPU_trace_D{D}.csv.gz'
        with gzip.open(portable,'wt',newline='') as f:
            w=csv.writer(f);w.writerow(['start_ns','end_ns','stream','kind','name'])
            for s,e,stream,name,node in kernels:w.writerow([s-start,e-start,stream,'kernel',names[name]])
            for s,e,stream in copies:w.writerow([s-start,e-start,stream,'memcopy',''])
        manifest.extend([dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in (report,database,portable)])
        c.close()
    for name,rows in (('nsight_summary.csv',summaries),('nsight_API_counts.csv',api_rows),('nsight_kernel_service.csv',kernel_rows)):
        if not rows:continue
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'nsight_manifest.json',manifest)
    if traces:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(len(traces),1,figsize=(12,3.2*len(traces)),squeeze=False)
        for ax,(D,(origin,kernels,copies)) in zip(axes.flat,sorted(traces.items())):
            streams=sorted({row[2] for row in kernels+copies})
            for index,stream in enumerate(streams):
                intervals=[((s-origin)/1e6,(e-s)/1e6) for s,e,st,*rest in kernels if st==stream and s-origin<80e6]
                ax.broken_barh(intervals,(index+.1,.6),facecolors='#2864ad')
                intervals=[((s-origin)/1e6,(e-s)/1e6) for s,e,st in copies if st==stream and s-origin<80e6]
                ax.broken_barh(intervals,(index+.1,.6),facecolors='#dd7733')
            ax.set_yticks([i+.4 for i in range(len(streams))],[f'CUDA stream {s}' for s in streams]);ax.set_xlim(0,80)
            ax.set_title(f'NSIGHT HEAVY TRACE, P256/N1/D{D}. Actual kernel (blue) and copy (orange) intervals. Not clean timing.')
            ax.set_xlabel('ms from first captured GPU operation')
        fig.tight_layout();fig.savefig(ROOT/'figures/nsight_actual_kernel_timeline.png',dpi=160);plt.close(fig)
    print(json.dumps(summaries))

if __name__=='__main__':main()
