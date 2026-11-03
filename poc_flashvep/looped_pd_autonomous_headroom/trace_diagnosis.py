"""Attribute recorded gaps conservatively from the two qualified prior traces."""
import argparse
import collections
import tarfile
from .common import *


def merge(spans):
    result=[]
    for left,right in sorted(spans):
        if right<=left:continue
        if result and left<=result[-1][1]:result[-1][1]=max(right,result[-1][1])
        else:result.append([left,right])
    return result


def intersect(a,b):
    i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total


def one(name):
    folder=PREVIOUS/'raw'/name
    proof=json.loads((folder/'profile_analysis.json').read_text());assert proof['eligible_for_matched_clean']
    anchor=json.loads((folder/'cuda_only_clock_prefill.json').read_text())
    window=json.loads((folder/'window.json').read_text())
    left=window['start']+.2*(window['last_arrival']-window['start'])
    right=window['start']+.9*(window['last_arrival']-window['start'])
    path=next((folder/'cuda_only_profile/prefill').glob('rank-*/cycle-*.tar.gz'))
    with tarfile.open(path) as bundle:
        member=next(m for m in bundle.getmembers() if m.name=='trace.json' or m.name.endswith('/trace.json'))
        raw=json.load(bundle.extractfile(member))
    base=raw['baseTimeNanoseconds'];groups=collections.defaultdict(list);counts=collections.Counter();api=collections.Counter()
    kernel_names=collections.defaultdict(lambda:[0,0.])
    for event in raw['traceEvents']:
        if event.get('ph')!='X':continue
        category=event.get('cat','');event_name=event.get('name','')
        if category not in ('kernel','gpu_memcpy','gpu_memset','cuda_runtime','cuda_driver','user_annotation'):continue
        start=anchor['monotonic']+(base+1000*event['ts']-anchor['epoch_ns'])/1e9
        end=start+event['dur']/1e6
        if end<=left or start>=right:continue
        span=(max(start,left),min(end,right));counts[category]+=1
        if category in ('cuda_runtime','cuda_driver'):
            n=event_name.lower();api[event_name]+=1
            kind='API_launch' if 'launch' in n else 'API_sync_query' if any(x in n for x in ('synchron','eventquery','streamquery')) else 'API_memory' if 'memcpy' in n or 'memset' in n else 'API_allocation' if 'alloc' in n or 'free' in n else 'API_other'
            groups[kind].append(span)
        else:groups[category].append(span)
        if category=='kernel':
            kernel_names[event_name][0]+=1;kernel_names[event_name][1]+=span[1]-span[0]
    merged={k:merge(v) for k,v in groups.items()}
    occupied=merged.get('kernel',[]);gaps=[];cursor=left
    for start,end in occupied:
        if start>cursor:gaps.append([cursor,start])
        cursor=max(cursor,end)
    if cursor<right:gaps.append([cursor,right])
    duration=right-left;gap_seconds=sum(b-a for a,b in gaps)
    result=dict(name=name,source_run=folder.name,previous_campaign=True,window_start=left,window_end=right,
        P_kernel_union_percent=100*sum(b-a for a,b in occupied)/duration,observation_seconds=duration,
        gap_seconds=gap_seconds,recorded_event_counts=counts,
        CPU_API_gap_overlap_seconds={k:intersect(v,gaps) for k,v in merged.items() if k.startswith('API_')},
        recorded_copy_gap_overlap_seconds=intersect(merged.get('gpu_memcpy',[]),gaps),
        most_frequent_APIs=api.most_common(15),
        largest_kernel_duration_sums=[dict(kernel=k,invocations=v[0],summed_device_seconds=v[1]) for k,v in sorted(kernel_names.items(),key=lambda item:item[1][1],reverse=True)[:20]],
        evidence='TRACE_ANALYSIS_OF_QUALIFIED_PREVIOUS_NATIVE_GPU_CAPTURE',
        scope='CPU API categories can overlap across threads. Their overlap with a GPU gap is not exclusive causal attribution. The old trace does not measure new optimization speedup.')
    write(ROOT/'state'/('prior_trace_'+folder.name+'.json'),result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-target',action='store_true');a=p.parse_args()
    setup()
    rows=[one(n) for n in ('long_profile_ouro_PD_start1_P_only_n128','profile_warm_huginn_selected_both_24')]
    write(ROOT/'prior_trace_gap_diagnosis.json',rows)
    for r in rows:print(json.dumps({k:r[k] for k in ('source_run','P_kernel_union_percent','gap_seconds','CPU_API_gap_overlap_seconds','most_frequent_APIs')}),flush=True)
    if a.check_target:assert all(r['P_kernel_union_percent']>=80 for r in rows),'PRIOR_BASELINE_FAILS_P_KERNEL_SCREEN'


if __name__=='__main__':main()
