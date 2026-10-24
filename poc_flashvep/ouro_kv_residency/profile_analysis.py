"""Keep trace-local service, union, and clean wall time separate."""
import csv
import gzip
import sqlite3
import subprocess
from collections import defaultdict
from .common import ROOT,write,digest
from ..ouro_elb.analyze_nsight import merged

def family(name):
    lower=name.lower()
    if 'flash_attn' in lower or 'flashattention' in lower:return 'ATTENTION_CORE'
    if 'index_put' in lower:return 'KV_WRITE_INDEX_KERNEL'
    if any(w in lower for w in ('gemm','gemv','cutlass','cublas','nvjet_')):return 'GEMM_UNRESOLVED_QKV_FFN'
    if any(w in lower for w in ('copy','gather_kernel','scatter_kernel')):return 'COPY_LAYOUT_METADATA_PARTIAL'
    if any(w in lower for w in ('metadata','slots_kernel')):return 'KV_METADATA'
    return 'NORM_ROPE_POINTWISE_OTHER_UNRESOLVED'

def main():
    summaries=[];kernels_summary=[];manifest=[];windows=[]
    for report in sorted((ROOT/'profiles').glob('profile_P*.nsys-rep')):
        database=report.with_suffix('.sqlite')
        if not database.exists():
            result=subprocess.run(['nsys','export','--type=sqlite','--output='+str(database),str(report)],capture_output=True,text=True)
            report.with_suffix('.export.log').write_text(result.stdout+result.stderr)
            assert result.returncode==0,str(report.with_suffix('.export.log'))
        db=sqlite3.connect(database)
        names=dict(db.execute('select id,value from StringIds'))
        kernels=db.execute('select start,end,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
        copies=db.execute('select start,end,streamId from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
        runtime=db.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME').fetchall()
        origin=min(s for s,e,*rest in kernels+copies);end=max(e for s,e,*rest in kernels+copies)
        intervals=defaultdict(list);counts=defaultdict(int)
        for s,e,stream,name in kernels:
            intervals[family(names[name])].append((s,e));counts[family(names[name])]+=1
        all_intervals=[(s,e) for s,e,*_ in kernels+copies]
        busy=sum(e-s for s,e in merged(all_intervals))
        item=dict(run=report.stem,trace_extent_ms=(end-origin)/1e6,kernel_service_sum_ms=sum(e-s for s,e,*_ in kernels)/1e6,
            kernel_copy_union_ms=busy/1e6,uncovered_extent_ms=(end-origin-busy)/1e6,kernels=len(kernels),copies=len(copies),
            cudaGraphLaunch=sum(names[n].startswith('cudaGraphLaunch') for s,e,n in runtime),
            attention_service_ms=sum(e-s for s,e in intervals['ATTENTION_CORE'])/1e6,
            attention_union_ms=sum(e-s for s,e in merged(intervals['ATTENTION_CORE']))/1e6,
            attention_trace_extent_fraction=sum(e-s for s,e in merged(intervals['ATTENTION_CORE']))/(end-origin),
            clean_critical_path_fraction='UNRESOLVED',HBM_traffic_bytes='NOT_MEASURED_IN_NSIGHT_SYSTEMS',
            evidence='LIVE_MEASURED_OPERATOR',lane='HEAVY_NSIGHT_TRACE_NOT_CLEAN_E2E')
        summaries.append(item)
        samples=sorted(e for s,e,st,name in kernels if 'ArgMaxOps' in names[name])
        for index,(begin,finish) in enumerate(zip(samples,samples[1:])):
            attention=[(max(s,begin),min(e,finish)) for s,e,st,name in kernels
                if family(names[name])=='ATTENTION_CORE' and s<finish and e>begin]
            other=[(max(s,begin),min(e,finish)) for s,e,st,name in kernels
                if family(names[name])!='ATTENTION_CORE' and s<finish and e>begin]
            au=merged(attention);ou=merged(other)
            overlap=sum(max(0,min(b,d)-max(a,c)) for a,b in au for c,d in ou)
            windows.append(dict(run=report.stem,index=index,attention_calls=len(attention),
                complete_R4_geometry=len(attention)==96,output_completion_interval_ms=(finish-begin)/1e6,
                attention_union_ms=sum(e-s for s,e in au)/1e6,
                attention_exclusive_GPU_service_ms=(sum(e-s for s,e in au)-overlap)/1e6,
                evidence='LIVE_MEASURED_OPERATOR',scope='Heavy-trace sampling-to-sampling window. Not clean host-delivered ITL or causal E2E saving.'))
        for name,times in intervals.items():
            kernels_summary.append(dict(run=report.stem,family=name,calls=counts[name],
                service_sum_ms=sum(e-s for s,e in times)/1e6,union_ms=sum(e-s for s,e in merged(times))/1e6))
        path=ROOT/'profiles'/f'{report.stem}_portable.csv.gz'
        with gzip.open(path,'wt',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['start_ns','end_ns','stream','family','name'])
            for s,e,st,name in kernels:writer.writerow([s-origin,e-origin,st,family(names[name]),names[name]])
            for s,e,st in copies:writer.writerow([s-origin,e-origin,st,'CUDA_MEMCPY',''])
        manifest.extend(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),bytes=p.stat().st_size) for p in (path,report,database))
        db.close()
    write(ROOT/'profile_summary.json',summaries)
    write(ROOT/'profile_kernel_families.json',kernels_summary)
    write(ROOT/'profile_complete_token_windows.json',windows)
    write(ROOT/'profile_artifact_manifest.json',manifest)
    with gzip.open(ROOT/'kernel_timing.csv.gz','wt',newline='') as output:
        writer=csv.writer(output);writer.writerow(['run','start_ns','end_ns','stream','family','name','evidence'])
        for path in sorted((ROOT/'profiles').glob('*_portable.csv.gz')):
            with gzip.open(path,'rt') as stream:
                rows=csv.reader(stream);next(rows)
                for row in rows:writer.writerow([path.name.removesuffix('_portable.csv.gz'),*row,'LIVE_MEASURED_OPERATOR_HEAVY_TRACE'])

if __name__=='__main__':main()
