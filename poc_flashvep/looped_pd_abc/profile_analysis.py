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
 rows=[];io=[]
 for after in sorted((ROOT/'raw').glob('*/after_measured.json')):
  before=after.with_name('before_measured.json');config=after.with_name('configuration.json')
  if not before.exists() or not config.exists():continue
  a=json.loads(before.read_text());b=json.loads(after.read_text());args=json.loads(config.read_text())['arguments']
  if not isinstance(b,dict):continue
  if 'requests' in b and 'used_blocks' in b:a={'local':a};b={'local':b}
  for worker,v in b.items():
   if not isinstance(v,dict) or 'weight_bytes' not in v:continue
   fields=('weight_bytes','derived_weight_pack_bytes','reserved_kv_bytes','peak_allocated_bytes')
   row=dict(run=after.parent.name,model=args['model'],worker=worker,physical_GPU_mask=args['gpus'],
    evidence='MEASURED_WORKER_ALLOCATIONS_AND_COHORT_COUNTER_DELTAS')
   for field in fields:row[field]=v.get(field,'NOT_MEASURED')
   for field in ('issued_loop_rows','issued_submissions','transport_bytes_sent','transport_completed_count','transport_completion_observation_seconds'):
    row[field+'_delta']=v.get(field,0)-a.get(worker,{}).get(field,0)
   row['graph_workspace_bytes']='NOT_MEASURED_SEPARATELY'
   row['actual_active_KV_bytes']='NOT_MEASURED_FROM_POST_DRAIN_SNAPSHOT'
   rows.append(row)
   for filename,stats in v.get('append_evidence_IO',{}).items():
    previous=a.get(worker,{}).get('append_evidence_IO',{}).get(filename,{})
    io.append(dict(run=after.parent.name,worker=worker,filename=filename,
     append_calls=stats['count']-previous.get('count',0),
     append_wall_ms=1000*(stats['total_seconds']-previous.get('total_seconds',0)),
     maximum_single_append_ms=1000*stats['max_seconds'],
     maximum_scope='Process lifetime, including warmup',
     evidence='MEASURED_CPU_LOGGING_COST_NOT_ADDITIVE_E2E_TIME'))
  iobefore=after.with_name('coordinator_IO_before.json');ioafter=after.with_name('coordinator_IO_after.json')
  if iobefore.exists() and ioafter.exists():
   x=json.loads(iobefore.read_text());y=json.loads(ioafter.read_text())
   for filename,stats in y.items():
    previous=x.get(filename,{})
    io.append(dict(run=after.parent.name,worker='coordinator',filename=filename,
     append_calls=stats['count']-previous.get('count',0),append_wall_ms=1000*(stats['total_seconds']-previous.get('total_seconds',0)),
     maximum_single_append_ms=1000*stats['max_seconds'],maximum_scope='Process lifetime, including warmup',
     evidence='MEASURED_CPU_LOGGING_COST_NOT_ADDITIVE_E2E_TIME'))
 save_csv(ROOT/'worker_memory_transport.csv',rows)
 save_csv(ROOT/'evidence_IO_cost.csv',io)

def nonblocking_evidence():
 """Report producer cost separately from background durability time."""
 rows=[]
 from .evidence_protocol import check
 for config in sorted((ROOT/'raw').glob('*/configuration.json')):
  folder=config.parent;data=json.loads(config.read_text())
  attempt=data.get('arguments',{}).get('attempt_id')
  for path in sorted(folder.glob('async_evidence_summary_*.json')):
   stats=json.loads(path.read_text())
   if stats.get('attempt_id')!=attempt:continue
   protocol=check(folder)
   rows.append(dict(run=folder.name,pid=stats['pid'],attempt_id=attempt,protocol_valid=protocol['valid'],
    enqueued_records=stats['enqueued_records'],durable_records=stats['background_written_records'],
    foreground_enqueue_ms=1000*stats['foreground_enqueue_seconds'],
    foreground_max_enqueue_ms=1000*stats['foreground_max_enqueue_seconds'],
    background_write_ms=1000*stats['background_write_seconds'],
    background_fsync_batches=stats['background_fsync_batches'],peak_queued_records=stats['peak_queued_records'],
    evidence='MEASURED_INSTRUMENTATION_COUNTERS_NOT_ADDITIVE_E2E_COMPONENTS',
    scope=stats['scope'],limitation='Producer counters exclude one-time writer initialization. Background I/O can still contend for shared CPU or storage.'))
 save_csv(ROOT/'nonblocking_evidence_cost.csv',rows)

def placement_residuals():
 rows=[]
 for path in sorted((ROOT/'raw').glob('*_C_*/closed_stats.json')):
  data=json.loads(path.read_text())
  for request in data.get('metrics',[]):
   if request['request_id'].startswith('r-1_'):continue
   worker=request.get('prefill_worker');pred=request.get('route_predictions',{}).get(worker)
   timings=request['timings']
   if not pred or not all(k in timings for k in ('admitted','prefill_started')):continue
   observed=timings['prefill_started']-timings['admitted']
   rows.append(dict(run=path.parent.name,request_id=request['request_id'],prefill_worker=worker,
    assigned_queue_work_prediction_ms=1000*pred['score'],observed_reservation_to_prefill_ms=1000*observed,
    assigned_P_phase_ms=1000*(timings['prefill_compute_completed']-timings['prefill_started']) if 'prefill_compute_completed' in timings else 'NOT_MEASURED',
    calibrated_completion_prediction_error_ms='NOT_MEASURED',
    evidence='MEASURED_CHOSEN_ROUTE_WITH_MODEL_PREDICTION',
    limitation='The rank score is pending work, not an ETA. The control timestamp is not a GPU first-kernel timestamp. These quantities cannot define a prediction error.'))
 save_csv(ROOT/'placement_residuals.csv',rows)

def main():
 memory_and_transport();nonblocking_evidence();placement_residuals();analyze_profiles()

if __name__=='__main__':main()
