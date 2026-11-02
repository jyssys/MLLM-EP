"""Separate actual kernel intervals from profiled device copies and memset."""
import collections
import tarfile
from .common import *
from .analyze import table
from .profile_analysis import union


def main():
    rows=[];details=[]
    for model in ('ouro','huginn'):
        paths=list((ROOT/'raw').glob(f'long_profile_{model}_*/profile_analysis.json'))
        if not paths:paths=list((ROOT/'raw').glob(f'profile_warm_{model}_*/profile_analysis.json'))
        choices=[]
        for p in paths:
            proof=json.loads(p.read_text())
            if proof.get('status')=='PARSED':choices.append((proof.get('eligible_for_matched_clean',False),'both' in p.parent.name,p.parent))
        if not choices:continue
        folder=max(choices,key=lambda r:r[:2])[2]
        window=json.loads((folder/'window.json').read_text())
        left=window['start']+.2*(window['last_arrival']-window['start'])
        right=window['start']+.9*(window['last_arrival']-window['start'])
        for role in ('prefill','decode'):
            archives=list((folder/'cuda_only_profile'/role).glob('rank-*/cycle-*.tar.gz'))
            anchor_path=folder/f'cuda_only_clock_{role}.json'
            if len(archives)!=1 or not anchor_path.exists():continue
            anchor=json.loads(anchor_path.read_text())
            with tarfile.open(archives[0]) as archive:
                member=next(m for m in archive.getmembers() if m.name.endswith('/trace.json') or m.name=='trace.json')
                raw=json.load(archive.extractfile(member))
            base=raw['baseTimeNanoseconds'];groups=collections.defaultdict(list);names=collections.Counter()
            for event in raw['traceEvents']:
                if event.get('ph')!='X':continue
                category=event.get('cat','')
                if category not in ('kernel','gpu_memcpy','gpu_memset'):continue
                device=event.get('args',{}).get('device')
                if device is not None and device!=anchor['logical_device']:continue
                begin=anchor['monotonic']+(base+1000*event['ts']-anchor['epoch_ns'])/1e9
                end=begin+event['dur']/1e6
                if end<=left or begin>=right:continue
                groups[category].append((max(left,begin),min(right,end)))
                names[(category,event.get('name','UNKNOWN'))]+=1
            all_intervals=[v for group in groups.values() for v in group]
            occupied=union(all_intervals);kernels=union(groups['kernel'])
            copies=union(groups['gpu_memcpy']);sets=union(groups['gpu_memset'])
            proof=json.loads((folder/f'profile_analysis_{role}.json').read_text())
            rows.append(dict(name=folder.name,model=model,role=role,observation_seconds=right-left,
                kernel_union_seconds=kernels,copy_union_seconds=copies,memset_union_seconds=sets,
                all_observed_device_activity_union_seconds=occupied,
                kernel_union_percent=100*kernels/(right-left),copy_union_percent=100*copies/(right-left),
                memset_union_percent=100*sets/(right-left),all_observed_activity_union_percent=100*occupied/(right-left),
                no_recorded_device_activity_percent=100*(1-occupied/(right-left)),
                overlap_between_category_unions_seconds=kernels+copies+sets-occupied,
                profile_eligible=proof.get('eligible_for_matched_clean',False),
                evidence='MEASURED_NATIVE_KINETO_DEVICE_EVENTS',
                scope='Overlapping category unions are not additive E2E components. Missing categories do not prove no fabric activity. CUPTI does not necessarily expose NIXL RDMA or remote GPU accesses. No-recorded activity is not certified safe compute slack.'))
            details.extend(dict(name=folder.name,role=role,category=category,event_name=name,count=count)
                for (category,name),count in names.items())
            del raw
    table('actual_device_activity_accounting.csv',rows)
    table('actual_device_activity_event_names.csv',details)
    print(json.dumps(dict(device_windows=len(rows),event_identities=len(details))))


if __name__=='__main__':main()
