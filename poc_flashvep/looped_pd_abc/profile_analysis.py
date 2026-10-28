"""Separate profiler event accounting from clean request wall-time accounting."""
from collections import defaultdict
import json
import tarfile
from .common import ROOT
from .analyze import save_csv

def union(intervals):
 end=None;total=0.
 for start,stop in sorted(intervals):
  if end is None or start>end:total+=stop-start;end=stop
  elif stop>end:total+=stop-end;end=stop
 return total

def analyze_profiles():
 rows=[];operators=[]
 for archive in sorted((ROOT/'raw').glob('*/profile/**/*.tar.gz')):
  try:
   with tarfile.open(archive) as bundle:
    stream=bundle.extractfile('trace.json')
    if stream is None:continue
    trace=json.load(stream)
   events=trace.get('traceEvents',[]);bydevice=defaultdict(list)
   for e in events:
    if e.get('ph')!='X' or not isinstance(e.get('dur'),(float,int)):continue
    category=e.get('cat','');args=e.get('args',{})
    if category not in ('kernel','gpu_memcpy','gpu_memset'):continue
    device=str(args.get('device',e.get('pid','unknown')))
    bydevice[device].append(e)
   for device,entries in bydevice.items():
    intervals=[(e['ts'],e['ts']+e['dur']) for e in entries]
    kernels=[(e['ts'],e['ts']+e['dur']) for e in entries if e['cat']=='kernel']
    copies=[(e['ts'],e['ts']+e['dur']) for e in entries if e['cat']=='gpu_memcpy']
    span=max(b for a,b in intervals)-min(a for a,b in intervals)
    busy=union(intervals);kernel=union(kernels);copy=union(copies)
    both=kernel+copy-union(kernels+copies)
    row=dict(archive=str(archive.relative_to(ROOT)),device=device,event_count=len(entries),
     observed_device_span_ms=span/1000,device_busy_union_ms=busy/1000,
     device_span_idle_ms=(span-busy)/1000,kernel_union_ms=kernel/1000,copy_union_ms=copy/1000,
     copy_kernel_overlap_ms=both/1000,copy_overlap_percent=100*both/copy if copy else 'NOT_AVAILABLE',
     evidence='MEASURED_PROFILED_DEVICE_EVENT_UNIONS_NOT_CLEAN_E2E',
     limitation='Window runs from first to last recorded device event. Leading/trailing idle and unrecorded backend events are not counted.')
    rows.append(row)
    names=defaultdict(list)
    for e in entries:names[(e['name'],e['cat'])].append(e['dur'])
    for (name,category),durations in names.items():
     operators.append(dict(archive=row['archive'],device=device,kernel=name,category=category,count=len(durations),
      inclusive_ms=sum(durations)/1000,evidence='MEASURED_PROFILED_INCLUSIVE_DURATION_NOT_E2E_SHARE'))
  except (OSError,ValueError,KeyError,tarfile.TarError) as error:
   rows.append(dict(archive=str(archive.relative_to(ROOT)),error=repr(error),evidence='PROFILE_PARSE_FAILED'))
 save_csv(ROOT/'profile_device_summary.csv',rows);save_csv(ROOT/'profile_operators.csv',operators)
 return rows

def memory_and_transport():
 rows=[]
 for after in sorted((ROOT/'raw').glob('*/after_measured.json')):
  before=after.with_name('before_measured.json');config=after.with_name('configuration.json')
  if not before.exists() or not config.exists():continue
  a=json.loads(before.read_text());b=json.loads(after.read_text());args=json.loads(config.read_text())['arguments']
  if not isinstance(b,dict):continue
  for worker,v in b.items():
   if not isinstance(v,dict):continue
   fields=('weight_bytes','derived_weight_pack_bytes','reserved_kv_bytes','peak_allocated_bytes')
   row=dict(run=after.parent.name,model=args['model'],worker=worker,physical_GPU_mask=args['gpus'],
    evidence='MEASURED_WORKER_ALLOCATIONS_AND_COHORT_COUNTER_DELTAS')
   for field in fields:row[field]=v.get(field,'NOT_MEASURED')
   for field in ('issued_loop_rows','issued_submissions','transport_bytes_sent','transport_completed_count','transport_completion_observation_seconds'):
    row[field+'_delta']=v.get(field,0)-a.get(worker,{}).get(field,0)
   row['graph_workspace_bytes']='NOT_MEASURED_SEPARATELY'
   row['actual_active_KV_bytes']='NOT_MEASURED_FROM_POST_DRAIN_SNAPSHOT'
   rows.append(row)
 save_csv(ROOT/'worker_memory_transport.csv',rows)

def main():
 memory_and_transport();analyze_profiles()

if __name__=='__main__':main()
