"""Compress profiler evidence without deleting originals. Separate sums and unions."""
import collections
import gzip
import hashlib
import json
import tarfile
from .common import ROOT,write
from .analyze import csv_write


def union(intervals):
    end=None;total=0.
    for left,right in sorted(intervals):
        if end is None or left>end:total+=right-left;end=right
        elif right>end:total+=right-end;end=right
    return total


def category(name):
    lower=name.lower()
    if 'flash' in lower or 'attention' in lower:return 'attention_name_match'
    # The captured External-id chain maps nvjet_tst kernels to aten::mm.
    if any(s in lower for s in ('gemm','xmma','cutlass','matmul','nvjet_tst')):return 'GEMM_name_match'
    if 'norm' in lower:return 'normalization_name_match'
    return 'other_or_unclassified'


def main():
    rows=[];shapes=[];archives=[];mapped=[]
    for folder in sorted((ROOT/'raw').glob('profile_*')):
        status=folder/'status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':continue
        for path in folder.rglob('cycle-*.tar.gz'):
            with tarfile.open(path,'r:gz') as bundle:
                data=bundle.extractfile('trace.json').read()
            trace=json.loads(data)
            events=trace.get('traceEvents') if isinstance(trace,dict) else None
            if events is None:continue
            compressed=path
            archives.append(dict(file=str(path.relative_to(ROOT))+'::trace.json',archive=str(path.relative_to(ROOT)),
                                 sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),compressed_bytes=path.stat().st_size))
            kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e]
            copies=[e for e in events if e.get('cat') in ('gpu_memcpy','gpu_memset') and 'dur' in e]
            groups=collections.defaultdict(list)
            for event in kernels:groups[category(event['name'])].append(event)
            intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
            busy=union(intervals)
            duration=max((b for a,b in intervals),default=0)-min((a for a,b in intervals),default=0)
            for kind,items in groups.items():
                rows.append(dict(name=folder.name,trace=str(compressed.relative_to(ROOT)),category=kind,
                                 actual_kernel_invocations=len(items),kernel_duration_sum_ms=sum(e['dur'] for e in items)/1000,
                                 category_union_ms=union([(e['ts'],e['ts']+e['dur']) for e in items])/1000,
                                 all_kernel_busy_union_ms=busy/1000,kernel_window_ms=duration/1000,
                                 all_GPU_copy_events=len(copies),all_GPU_copy_duration_sum_ms=sum(e['dur'] for e in copies)/1000,
                                 evidence='MEASURED_PROFILE',
                                 limitation='Kernel-name classification. Overlapping sums are not exclusive E2E components. Copies are not identified as KV-only.'))
            cpu={e.get('args',{}).get('External id'):e for e in events if e.get('cat')=='cpu_op'}
            mapped_groups=collections.defaultdict(list)
            model='huginn' if 'huginn' in folder.name else 'ouro'
            width=5280 if model=='huginn' else 2048
            ffn=17920 if model=='huginn' else 5632
            for kernel in kernels:
                operator=cpu.get(kernel.get('args',{}).get('External id'))
                if not operator or operator['name']!='aten::mm':continue
                dimensions=operator.get('args',{}).get('Input Dims',[])
                if len(dimensions)!=2 or any(len(d)!=2 for d in dimensions):continue
                K,N=dimensions[1]
                kind=('FFN_GEMM' if (K,N) in ((width,ffn),(width,ffn*2),(ffn,width))
                      else 'reinjection_adapter_GEMM' if model=='huginn' and (K,N)==(2*width,width)
                      else 'attention_projection_GEMM' if (K,N) in ((width,width),(width,3*width))
                      else 'other_GEMM')
                mapped_groups[kind].append(kernel)
            for kind,items in mapped_groups.items():
                mapped.append(dict(name=folder.name,trace=str(compressed.relative_to(ROOT)),category=kind,
                                   directly_correlated_kernel_count=len(items),duration_sum_ms=sum(e['dur'] for e in items)/1000,
                                   evidence='MEASURED_EXTERNAL_ID_AND_MATRIX_SHAPE_CORRELATION',
                                   limitation='Direct aten::mm mappings only. Some CUDA Graph replay kernels have no matching CPU operation. Not complete module time or E2E shares.'))
            counts=collections.Counter()
            for e in events:
                if e.get('cat')=='cpu_op' and e.get('name') in ('aten::mm','aten::addmm','aten::linear'):
                    counts[(e['name'],json.dumps(e.get('args',{}).get('Input Dims','NOT_RECORDED')))]+=1
            for (operation,shape),count in counts.items():
                shapes.append(dict(name=folder.name,trace=str(compressed.relative_to(ROOT)),operation=operation,
                                   input_dims=shape,observed_CPU_invocations=count,
                                   limitation='CUDA Graph replays do not execute Python/aten dispatch again. Not total replayed GEMM count.'))
    csv_write('profiler_kernel_breakdown.csv',rows);csv_write('profiler_observed_GEMM_shapes.csv',shapes)
    csv_write('profiler_directly_correlated_GEMM_time.csv',mapped)
    write(ROOT/'configs/profiler_archives.json',archives)
    print('PROFILE_TRACES',len(archives),'KERNEL_ROWS',len(rows))


if __name__=='__main__':main()
