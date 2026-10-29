"""Host critical-path partitions and explicitly frozen-timeline transfer oracles."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .analyze import save_csv,requests

PHASES=('coordinator_queue','reservation_control','P_assigned_prefill_including_backpressure',
 'uncovered_handoff','D_resume_coda_first_token','remaining_decode_phase','harness_delivery_poll_residual')

def partition_statistics(records):
 """The denominator is summed request wall time, not GPU execution time."""
 valid=[]
 for row in records:
  if str(row.get('phase_order_valid')).lower()!='true':continue
  if not all(row.get(name+'_ms') not in (None,'') for name in PHASES):continue
  elapsed=float(row['harness_E2E_ms'])
  values=[float(row[name+'_ms']) for name in PHASES]
  if elapsed<=0 or any(x<0 for x in values):continue
  if not np.isclose(sum(values),elapsed,rtol=1e-10,atol=1e-6):
   raise ValueError('Host partition does not equal harness E2E')
  valid.append((elapsed,values))
 if not valid:return []
 elapsed=np.asarray([x for x,y in valid]);values=np.asarray([y for x,y in valid])
 return [dict(phase=name,N=len(valid),phase_mean_ms=float(values[:,i].mean()),
  phase_p95_ms=float(np.percentile(values[:,i],95)),
  share_of_summed_request_E2E_percent=float(100*values[:,i].sum()/elapsed.sum()),
  mean_request_share_percent=float(np.mean(100*values[:,i]/elapsed)),
  evidence='MEASURED_HOST_CRITICAL_PATH_PARTITION',
  denominator='Sum of harness arrival-to-final-output request times in this run',
  limitation='Concurrent requests overlap. These shares are not GPU utilization, DMA shares, or additive cohort makespan components.')
  for i,name in enumerate(PHASES)]

def fixed_timeline_oracle(records):
 """Remove only the observed post-P handoff interval. No schedule is replayed."""
 before=[];after=[];tails=[]
 for row in records:
  if str(row.get('phase_order_valid')).lower()!='true':continue
  if not row.get('harness_E2E_ms'):continue
  elapsed=float(row['harness_E2E_ms']);tail=float(row['uncovered_handoff_ms'])
  if not 0<=tail<elapsed:continue
  before.append(elapsed);after.append(elapsed-tail);tails.append(tail)
 if not before:return None
 original=float(np.percentile(before,95));counterfactual=float(np.percentile(after,95))
 return dict(N=len(before),observed_p95_E2E_ms=original,
  removed_handoff_mean_ms=float(np.mean(tails)),removed_handoff_p95_ms=float(np.percentile(tails,95)),
  frozen_timeline_p95_E2E_ms=counterfactual,frozen_timeline_p95_ratio=original/counterfactual,
  evidence='ORACLE_FROZEN_TIMELINE_NOT_MEASURED_SPEEDUP',
  assumptions='Erase the entire observed post-P handoff interval. Retain every other request interval unchanged.',
  limitation='This is not a global serving bound. It omits queue feedback, CPU backpressure, contention, new metadata, and transfer scheduling changes.')

def full_depth_work():
 rows=[]
 for folder in sorted((ROOT/'raw').iterdir()):
  status=folder/'status.json';before=folder/'before_measured.json';after=folder/'after_measured.json'
  if not all(path.exists() for path in (status,before,after)):continue
  if json.loads(status.read_text()).get('status')!='COMPLETE':continue
  data=requests(folder)
  if not data or 'input_tokens' not in data[0]:continue
  a=json.loads(before.read_text());b=json.loads(after.read_text())
  if 'issued_loop_rows' in b:a={'local':a};b={'local':b}
  if not b or not all(isinstance(v,dict) and 'issued_loop_rows' in v and 'issued_loop_rows' in a.get(k,{}) for k,v in b.items()):continue
  actual=sum(v['issued_loop_rows']-a[k]['issued_loop_rows'] for k,v in b.items())
  required=sum((row['input_tokens']+max(row['output_tokens']-1,0))*row['depth'] for row in data)
  rows.append(dict(run=folder.name,model=data[0]['model'],N=len(data),issued_core_token_loop_rows=actual,
   required_core_token_loop_rows=required,difference=actual-required,
   equality=actual==required,evidence='MEASURED_SUBMISSION_COUNTER_VS_STRUCTURAL_FULL_DEPTH_ACCOUNTING',
   formula='Sum R*(actual prompt tokens + max(actual output tokens - 1, 0))',
   limitation='Counts logical submitted rows, not padded GPU GEMM rows. Boundary layers are excluded. Extra asynchronous work can precede EOS/cap readback.'))
 save_csv(ROOT/'full_depth_work_accounting.csv',rows)
 return rows

def main():
 source=ROOT/'e2e_phase_breakdown.csv'
 if not source.exists():return
 groups=defaultdict(list)
 with source.open() as stream:
  for row in csv.DictReader(stream):groups[row['name']].append(row)
 shares=[];oracles=[]
 for name,records in sorted(groups.items()):
  from .evidence_protocol import check
  protocol=check(ROOT/'raw'/name)
  interpretation=dict(evidence_io_mode=protocol['mode'],clean_method_timing_eligible=protocol['valid'])
  for row in partition_statistics(records):shares.append(dict(run=name,**row,**interpretation))
  row=fixed_timeline_oracle(records)
  if row:oracles.append(dict(run=name,**row,**interpretation))
 save_csv(ROOT/'e2e_phase_percent_summary.csv',shares)
 save_csv(ROOT/'handoff_frozen_timeline_oracle.csv',oracles)
 full_depth_work()

if __name__=='__main__':main()
