"""Separate measured-window failures from logged context teardown errors."""
import collections
import datetime
import json
import re
from .common import ROOT,write,digest


def main():
    jobs=[]
    for receipt in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        folder=receipt.parent;result=json.loads(receipt.read_text())
        status_path=folder/'status.json';window_path=folder/'window.json'
        if not result.get('complete') or not status_path.exists() or not window_path.exists():
            continue
        status=json.loads(status_path.read_text());window=json.loads(window_path.read_text())
        complete=status['time'];start_estimate=complete-(window['end']-window['start'])
        counts=collections.Counter();examples={};JIT=[]
        for path in folder.glob('*.log'):
            for line in path.read_text(errors='replace').splitlines():
                if 'UCX  ERROR' in line and 'cuDevicePrimaryCtxGetState' in line:
                    stamp=re.search(r'^\[(\d+\.\d+)\]',line)
                    epoch=float(stamp[1]) if stamp else None
                    category='POST_COMPLETE_CUDA_CONTEXT_QUERY' if epoch and epoch>complete else 'UNQUALIFIED_CONTEXT_ERROR_TIME'
                    counts[category]+=1
                    examples.setdefault(category,dict(file=path.name,line=line,epoch=epoch,
                        after_COMPLETE_s=epoch-complete if epoch else None))
                elif re.search(r'CUDA error:|CUDA_ERROR_|illegal memory access|device-side assert|CUDA launch failure',line,re.I):
                    counts['UNCLASSIFIED_CUDA_FAILURE']+=1
                    examples.setdefault('UNCLASSIFIED_CUDA_FAILURE',dict(file=path.name,line=line))
                if 'Triton kernel JIT compilation during inference:' in line:
                    stamp=re.search(r'(\d\d-\d\d \d\d:\d\d:\d\d)',line)
                    epoch=None
                    if stamp:
                        local=datetime.datetime.strptime('2026-'+stamp[1],'%Y-%m-%d %H:%M:%S')
                        epoch=local.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=9))).timestamp()
                    category='PRE_WINDOW_JIT' if epoch and epoch+1<start_estimate else 'JIT_WINDOW_OR_BOUNDARY_UNCERTAIN'
                    counts[category]+=1;JIT.append(dict(file=path.name,category=category,epoch=epoch,
                        estimated_window_start=start_estimate,line=line))
        jobs.append(dict(name=folder.name,counts=dict(counts),examples=examples,JIT=JIT,
            coordinate_COMPLETE_epoch=complete,estimated_window_start_epoch=start_estimate,
            normal_worker_exit=result['normal_worker_exit'],
            scope='COMPLETE follows the final delivered output. Post-COMPLETE context queries are outside the measurement. Start epoch reconstruction uses the status/window relation and is approximate. JIT second-resolution boundary cases remain uncertain.'))
    header='/usr/local/cuda/include/cuda.h'
    write(ROOT/'analysis/CUDA_and_JIT_log_audit.json',dict(jobs=jobs,
        totals=dict(sum((collections.Counter(row['counts']) for row in jobs),collections.Counter())),
        local_header=dict(path=header,SHA256=digest(header),symbol='CUDA_ERROR_DEINITIALIZED = 4'),
        official_source='https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__TYPES.html',
        evidence='TRACE_ANALYSIS',scope='Do not call a process error-free when its teardown logs errors. Do not confuse a post-window context-deinitialization query with a serving kernel launch failure. Unknown failure timing remains unqualified. This audit does not establish physical GPU kernel busy/idle.'))
    print('CUDA_JIT_AUDIT',len(jobs),flush=True)


if __name__=='__main__':
    main()
