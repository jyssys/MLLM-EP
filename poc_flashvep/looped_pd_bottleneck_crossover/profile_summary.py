"""Intersect actual CUDA activity with explicit profiled phase ranges."""
import tarfile
from .common import *
from .summary_tables import save_csv
from ..looped_pd_abc.profile_analysis import union

def main():
    from ..looped_pd_abc import profile_analysis as prior
    prior.ROOT=ROOT;prior.analyze_profiles();prior.memory_and_transport()
    phases=[];shapes=[]
    for path in sorted((ROOT/'raw').glob('*/profile/**/*.tar.gz')):
        with tarfile.open(path) as archive:
            stream=archive.extractfile('trace.json')
            if stream is None:continue
            trace=json.load(stream)
        events=[e for e in trace.get('traceEvents',[]) if e.get('ph')=='X' and isinstance(e.get('dur'),(int,float))]
        device=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
        for span in events:
            if not span.get('name','').startswith('bc_phase_'):continue
            start=span['ts'];end=start+span['dur']
            intervals={kind:[] for kind in ('kernel','gpu_memcpy','gpu_memset')}
            for event in device:
                lo=max(start,event['ts']);hi=min(end,event['ts']+event['dur'])
                if hi>lo:intervals[event['cat']].append((lo,hi))
            kernels=union(intervals['kernel']);copies=union(intervals['gpu_memcpy']);busy=union(sum(intervals.values(),[]))
            phases.append(dict(run=path.relative_to(ROOT/'raw').parts[0],archive=str(path.relative_to(ROOT)),phase=span['name'],
                host_phase_ms=span['dur']/1000,kernel_busy_union_ms=kernels/1000,copy_union_ms=copies/1000,
                device_busy_union_ms=busy/1000,kernel_busy_fraction_of_profiled_host_phase=kernels/span['dur'],
                device_idle_or_unrecorded_fraction=1-busy/span['dur'],
                evidence='MEASURED_CUPTI_PROFILED_PHASE_NOT_CLEAN_SERVING_LATENCY',
                scope='Union of recorded device activity inside a CPU phase range. Do not add overlapping kernel and copy durations.'))
        counts={}
        for e in events:
            dims=e.get('args',{}).get('Input Dims')
            if not dims or e['name'] not in ('aten::linear','aten::mm','aten::addmm','aten::matmul'):continue
            key=e['name'],json.dumps(dims);value=counts.setdefault(key,[0,0.]);value[0]+=1;value[1]+=e['dur']
        for (name,dims),(count,duration) in counts.items():
            shapes.append(dict(run=path.relative_to(ROOT/'raw').parts[0],operator=name,input_dims=dims,count=count,
                host_inclusive_ms=duration/1000,evidence='MEASURED_PROFILE_CPU_OPERATOR_SHAPES',
                scope='Captured CPU operators only. CUDA Graph replay can omit individual operator entries. Not an exhaustive replay GEMM count.'))
    save_csv(ROOT/'profiled_phase_kernel_busy.csv',phases);save_csv(ROOT/'profiled_operator_shapes.csv',shapes)
    print('PROFILE_PHASES',len(phases))

if __name__=='__main__':main()
