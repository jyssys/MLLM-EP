"""Optional profiler-only diagnostics. Never merge these times into clean serving."""
import collections
import json
import subprocess
import tarfile
from .common import *
from .analyze import csv_write,metric
from ..adaptive_recurrence_cut.profile_timeline import merge,intersection

def capture():
    if remaining()<4800:
        write(ROOT/'state/profiling.json',dict(status='NOT_RUN_TIME_BUDGET',time=time.time()));return
    jobs=[]
    for model in ('ouro','huginn'):
        for arm in ('PD_G1','Loop_Static','Token_Static'):
            source=ROOT/'configs'/f'primary_{model}_w1_{arm.lower()}_start0.json'
            if not source.exists():continue
            cfg=json.loads(source.read_text());name=f'profile_{model}_{arm.lower()}_n2'
            parent=ROOT/'raw'/cfg['name']
            if not (parent/'status.json').exists() or metric(cfg,parent)[0]['status']!='COMPLETE':continue
            cfg.update(name=name,phase='profile',count=2,request_caps=[4,4],cap=4,
                prompt_ids=cfg['prompt_ids'][:2],arrival_offsets=[0.,.001],profile=True,estimated_seconds=150,
                cut=cfg['depth']//2 if arm=='Loop_Static' else cfg['depth'],prefix_fraction=.5)
            path=ROOT/'configs'/f'{name}.json'
            if not path.exists():write(path,cfg)
            jobs.append(path.name)
    if jobs:
        subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.dispatch',*jobs,'--quiet'],cwd=REPO)
    analyze()

def analyze():
    kernels=[];summaries=[];shapes=[];overlap=[]
    for folder in sorted((ROOT/'raw').glob('profile_*')):
        captures=[]
        for path in sorted(folder.rglob('cycle-*.tar.gz')):
            with tarfile.open(path,'r:gz') as archive:data=json.load(archive.extractfile('trace.json'))
            events=data['traceEvents'];span=[];names=collections.defaultdict(lambda:[0,0.])
            for e in events:
                if e.get('cat')=='kernel' and 'dur' in e:
                    span.append((e['ts'],e['ts']+e['dur']));names[e['name']][0]+=1;names[e['name']][1]+=e['dur']
                if e.get('cat')=='cpu_op' and e.get('name') in ('aten::mm','aten::addmm'):
                    shapes.append(dict(name=folder.name,rank=path.parent.name,operation=e['name'],
                        input_shapes=json.dumps(e.get('args',{}).get('Input Dims',[])),
                        scope='CPU_OPERATOR_SHAPE_METADATA. Graph replay can omit shapes. Not exclusive GPU GEMM time.'))
            united=merge(span)
            if not united:continue
            summaries.append(dict(name=folder.name,rank=path.parent.name,kernel_count=len(span),
                kernel_duration_sum_ms=sum(b-a for a,b in span)/1000,
                kernel_union_ms=sum(b-a for a,b in united)/1000,
                kernel_window_ms=(united[-1][1]-united[0][0])/1000,
                evidence='MEASURED_PROFILER_ONLY_NOT_CLEAN_E2E_OR_NIXL_DMA'))
            for name,(count,duration) in names.items():kernels.append(dict(name=folder.name,rank=path.parent.name,
                kernel_name=name,count=count,duration_sum_ms=duration/1000))
            if 'baseTimeNanoseconds' in data:captures.append((data['baseTimeNanoseconds'],united))
        if len(captures)==2:
            anchor=min(c[0] for c in captures)
            intervals=[[(a+(clock-anchor)/1000,b+(clock-anchor)/1000) for a,b in spans] for clock,spans in captures]
            overlap.append(dict(name=folder.name,cross_device_kernel_overlap_ms=intersection(*intervals)/1000,
                evidence='CLOCK_ALIGNED_KERNEL_UNION_INTERSECTION. Not critical-path speedup.'))
    csv_write(ROOT/'profiler_kernel_names.csv',kernels);csv_write(ROOT/'profiler_rank_summary.csv',summaries)
    csv_write(ROOT/'profiler_observed_mm_shapes.csv',shapes);csv_write(ROOT/'profiler_overlap.csv',overlap)
    write(ROOT/'state/profiling.json',dict(status='MEASURED' if summaries else 'NOT_AVAILABLE',rank_traces=len(summaries),time=time.time()))

if __name__=='__main__':capture()
