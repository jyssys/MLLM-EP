"""Clock-aligned kernel timelines. Concurrent kernels do not prove speedup."""
import json
import math
import tarfile
from .common import ROOT
from .analyze import csv_write


def merge(intervals):
    out=[]
    for left,right in sorted(intervals):
        if not out or left>out[-1][1]:out.append([left,right])
        else:out[-1][1]=max(out[-1][1],right)
    return out


def intersection(a,b):
    i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];overlap=[]
    for folder in sorted((ROOT/'raw').glob('profile_*')):
        captures=[]
        for path in sorted(folder.rglob('cycle-*.tar.gz')):
            with tarfile.open(path,'r:gz') as bundle:data=json.load(bundle.extractfile('trace.json'))
            if 'baseTimeNanoseconds' not in data:raise RuntimeError('No shared trace-clock anchor')
            spans=[(e['ts'],e['ts']+e['dur']) for e in data['traceEvents'] if e.get('cat')=='kernel' and 'dur' in e]
            captures.append((path.parent.name,data['baseTimeNanoseconds'],spans))
        if len(captures)!=2:continue
        base=min(c[1] for c in captures)
        aligned=[merge([(a+(clock-base)/1000,b+(clock-base)/1000) for a,b in spans]) for rank,clock,spans in captures]
        origin=min(a for spans in aligned for a,b in spans)
        aligned=[[(a-origin,b-origin) for a,b in spans] for spans in aligned]
        end=max(b for spans in aligned for a,b in spans);quantum=10000.;bins=math.ceil(end/quantum)
        grid=[]
        for rank,spans in enumerate(aligned):
            busy=[0.]*bins
            for a,b in spans:
                for index in range(int(a//quantum),min(bins,int(b//quantum)+1)):
                    busy[index]+=max(0.,min(b,(index+1)*quantum)-max(a,index*quantum))/quantum
            grid.append(busy)
            for index,fraction in enumerate(busy):rows.append(dict(name=folder.name,rank=rank,
                  left_ms=index*10,right_ms=(index+1)*10,kernel_busy_fraction=fraction,evidence='MEASURED_PROFILE_CLOCK_ALIGNED'))
        overlap.append(dict(name=folder.name,cross_device_kernel_overlap_ms=intersection(*aligned)/1000,
                            evidence='MEASURED_PROFILE_CONCURRENCY_NOT_SPEEDUP',
                            scope='Intersection of actual kernel unions, on the shared trace clock. Does not identify a request critical path.'))
        fig,ax=plt.subplots(figsize=(9,2.8),layout='constrained')
        image=ax.imshow(grid,aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='Blues',extent=(0,bins*10,1.5,-.5))
        ax.set_yticks([0,1],['P / GPU0','D / GPU1']);ax.set_xlabel('Milliseconds from first traced kernel, aligned clocks')
        ax.set_title(folder.name+'\nN2, cap4, profiler only. Ten-ms bins. Blank can include copy/wait time.')
        fig.colorbar(image,ax=ax,label='Traced kernel busy fraction')
        fig.savefig(ROOT/'figures'/f'{folder.name}_kernel_timeline.png',dpi=180)
        fig.savefig(ROOT/'figures'/f'{folder.name}_kernel_timeline.svg');plt.close(fig)
    csv_write('profile_kernel_timeline_bins.csv',rows);csv_write('profile_cross_device_overlap.csv',overlap)
    print('CLOCK_ALIGNED_PROFILE_PAIRS',len(overlap))


if __name__=='__main__':main()
