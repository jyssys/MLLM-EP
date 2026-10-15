"""Count every observed native forward. Do not infer E2E shares from kernels."""
import csv
import gzip
import hashlib
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    directory=ROOT/'results/physics/TP4_R4_chunk0_v1'
    profile=read(directory/'native_kernel_profile.json')
    records=[]
    for path in sorted(directory.glob('torch_events.rank*.json.gz')):
        with gzip.open(path,'rt') as stream:events=json.load(stream)
        gpu=sorted((e for e in events if 'CUDA' in e['device_type']),key=lambda e:(e['start_us'],e['end_us']))
        markers=[e['start_us'] for e in gpu if e['name']=='_compute_slot_mappings_kernel']
        rank=int(path.name.split('rank')[1].split('.')[0])
        assert len(markers)==4,(rank,len(markers))
        for index,start in enumerate(markers):
            stop=markers[index+1] if index+1<len(markers) else float('inf')
            selected=[e for e in gpu if start<=e['start_us']<stop]
            names=[e['name'] for e in selected]
            writes=sum('reshape_and_cache' in name for name in names)
            attention=sum('FlashAttnFwdSm90' in name for name in names)
            collectives=[name for name in names if 'nccl' in name.lower() or 'cross_device_reduce' in name]
            records.append(dict(rank=rank,forward=index,phase='PREFILL' if index==0 else 'CACHED_DECODE',
                KV_write_count=writes,attention_main_count=attention,recognized_collective_kernel_count=len(collectives),
                kernel_count=len(names),kernel_order_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                expected_logical_depth=192,execution_count_PASS=writes==attention==192))
    assert len(records)==16
    with (ROOT/'results/native_forward_counts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(ROOT/'results/native_execution_audit.json',dict(
        all_count_checks_pass=all(r['execution_count_PASS'] for r in records),
        host_runtime_counts=profile['counts'],
        ranks=4,actual_observed_forwards_per_rank=4,observer_percent=profile['observer_percent'],
        timing_attribution='UNVALIDATED' if abs(profile['observer_percent'])>10 else 'DIAGNOSTIC_ONLY',
        scope='Actual separate trace. No last-replay multiplication. Counts do not measure a non-overlapped critical path.'))


if __name__=='__main__':main()
