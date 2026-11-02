"""Measure device interval gaps without naming them exclusive CPU or DMA time."""
import collections
from .common import *
from .analyze import table
from .profile_analysis import union


def main():
    selected=[]
    for model in ('ouro','huginn'):
        paths=list((ROOT/'raw').glob(f'long_profile_{model}_*/profile_analysis.json'))
        if not paths:paths=list((ROOT/'raw').glob(f'profile_warm_{model}_*/profile_analysis.json'))
        choices=[]
        for p in paths:
            value=json.loads(p.read_text())
            if value.get('status')=='PARSED':choices.append((value.get('eligible_for_matched_clean',False),'both' in p.parent.name,p.parent))
        if choices:selected.append(max(choices,key=lambda r:r[:2])[2])
    totals=[];classes=[]
    for folder in selected:
        window=json.loads((folder/'window.json').read_text())
        left=window['start']+.2*(window['last_arrival']-window['start'])
        right=window['start']+.9*(window['last_arrival']-window['start'])
        for role in ('prefill','decode'):
            source=folder/f'aligned_kernels_{role}.json'
            proof=folder/f'profile_analysis_{role}.json'
            if not source.exists() or not proof.exists():continue
            p=json.loads(proof.read_text())
            value=json.loads(source.read_text())
            kernels=[r for r in value['kernels'] if r['end_monotonic']>left and r['start_monotonic']<right]
            intervals=sorted((max(left,r['start_monotonic']),min(right,r['end_monotonic'])) for r in kernels)
            merged=[]
            for start,end in intervals:
                if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
                else:merged.append([start,end])
            gaps=[];cursor=left
            for start,end in merged:
                if start>cursor:gaps.append(start-cursor)
                cursor=max(cursor,end)
            if cursor<right:gaps.append(right-cursor)
            busy=sum(b-a for a,b in merged)
            row=dict(name=folder.name,role=role,window_seconds=right-left,
                kernel_count=len(kernels),kernel_union_seconds=busy,kernel_union_percent=100*busy/(right-left),
                no_recorded_kernel_seconds=sum(gaps),no_recorded_kernel_percent=100*sum(gaps)/(right-left),
                maximum_observed_gap_ms=1000*max(gaps,default=0),
                profile_eligible=p['eligible_for_matched_clean'],
                evidence='MEASURED_RECORDED_DEVICE_INTERVAL_UNION',
                scope='Same-start central arrival interval. A gap can contain copies, synchronization, submission, or other unrecorded activity. It is not proved safe borrowing time or exclusive CPU overhead.')
            for bound in (.0001,.001,.01):
                row[f'gap_seconds_at_least_{bound:g}s']=sum(g for g in gaps if g>=bound)
                row[f'gap_count_at_least_{bound:g}s']=sum(g>=bound for g in gaps)
            totals.append(row)
            identities=collections.defaultdict(list)
            for k in kernels:identities[k['name']].append((max(left,k['start_monotonic']),min(right,k['end_monotonic'])))
            for name,spans in identities.items():
                classes.append(dict(name=folder.name,role=role,kernel_name=name,invocations=len(spans),
                    same_name_union_seconds=union(spans),summed_duration_seconds=sum(b-a for a,b in spans),
                    profile_eligible=p['eligible_for_matched_clean'],evidence='MEASURED_KERNEL_IDENTITIES_AND_DEVICE_DURATIONS',
                    scope='Names are not inferred model operations. Different-name unions can overlap. Do not sum them into exclusive E2E percentages.'))
    table('actual_kernel_gap_accounting.csv',totals)
    table('actual_kernel_identity_accounting.csv',classes)
    print(json.dumps(dict(starts=len(selected),device_windows=len(totals),kernel_identities=len(classes))))


if __name__=='__main__':main()
