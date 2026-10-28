"""Summarize real evidence. Never fill missing GPU measurements with model values."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import ROOT,write

def load_rows(path):
 if not path.exists():return []
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def requests(folder):
 rows=load_rows(folder/'requests.jsonl');config=folder/'configuration.json'
 attempt=json.loads(config.read_text()).get('arguments',{}).get('attempt_id') if config.exists() else None
 if attempt:rows=[r for r in rows if r.get('attempt_id')==attempt]
 return [r for r in rows if not r.get('warmup',False)]

def save_csv(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()} for row in rows)

def summarize():
 inventory=[];summary=[];phase=[]
 for folder in sorted((ROOT/'raw').iterdir()):
  if not folder.is_dir():continue
  status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
  rows=requests(folder);inventory.append(dict(name=folder.name,status=status.get('status','NOT_AVAILABLE'),requests=len(rows),error=status.get('error','')))
  if status.get('status')!='COMPLETE' or not rows or 'ttft_ms' not in rows[0]:continue
  first=rows[0];r=dict(name=folder.name,model=first['model'],method=first['method'],gpus=first['gpus'],
   gpu_count=len(first['gpus'].split(',')),P=first['prefill_devices'] if first['method']=='pd' else 0,
   requests=len(rows),input_tokens_mean=float(np.mean([x['input_tokens'] for x in rows])),
   output_tokens=sum(x['output_tokens'] for x in rows),truncated=sum(x['truncated'] if 'truncated' in x else x['finish_reason']=='length' for x in rows),
   A=first.get('enable_recurrence_split',False),B=first.get('enable_frontier_handoff',False),C=first.get('placement','upstream'),
   evidence='MEASURED',scope=first['timing_scope'],model_revision=first['model_revision'],engine_revision=first['engine_revision'],
   depth=first['depth'],kv_layout=first['kv_layout'],precision=first['dtype'],
   D=len(first['gpus'].split(','))-first['prefill_devices'] if first['method']=='pd' else 0,
   colocated_replicas=len(first['gpus'].split(',')) if first['method']=='dp' else int(first['method'] in ('native','decode_local')))
  config=folder/'configuration.json'
  if config.exists():
   arguments=json.loads(config.read_text()).get('arguments',{})
   r.update(output_cap=arguments.get('cap','NOT_AVAILABLE'),arrival_rate=arguments.get('rate',0),
    scheduler_mode=arguments.get('scheduler_mode','refill'),split_cut=arguments.get('split_cut') if r['A'] else first['depth'],
    unique_input_hashes=len({json.dumps(x['input_ids']) for x in rows}))
  r['lane']=first.get('lane','parallel_discovery')
  r['official_huginn_protocol']=first.get('huginn_reference_compat',False)
  r['dory_adapter']=first.get('dory_adapter','initial_unpacked_adapter')
  for key in ('ttft_ms','e2e_ms','tpot_ms'):
   values=[x[key] for x in rows if key!='tpot_ms' or x['output_tokens']>1]
   r[key+'_mean']=float(np.mean(values)) if values else 'NOT_AVAILABLE'
   for q in (50,95,99):r[key+f'_p{q}']=float(np.percentile(values,q)) if values else 'NOT_AVAILABLE'
  r['TPOT_defined_requests']=sum(x['output_tokens']>1 for x in rows)
  r['coalesced_output_events']=sum(e['new_tokens']>1 for x in rows for e in x['token_events'])
  intervals=[(e['time']-e['previous_time'])*1000 for x in rows for e in x['token_events'] if not e['first'] and e['new_tokens']==1]
  for q in (50,95,99):r[f'ITL_ms_p{q}']=float(np.percentile(intervals,q)) if intervals else 'NOT_AVAILABLE'
  span=max(x['finish'] for x in rows)-min(x['arrival'] for x in rows)
  r['cohort_makespan_s']=span;r['requests_s']=len(rows)/span;r['output_tokens_s']=r['output_tokens']/span
  stats=folder/'closed_stats.json'
  before=folder/'before_measured.json';after=folder/'after_measured.json'
  if before.exists() and after.exists():
   a=json.loads(before.read_text());b=json.loads(after.read_text())
   for label,field in [('measured_transfer_bytes','transport_bytes_sent'),('measured_transfer_count','transport_completed_count'),
     ('measured_loop_rows','issued_loop_rows'),('measured_submissions','issued_submissions'),
     ('transfer_completion_observation_seconds','transport_completion_observation_seconds')]:
    if all(isinstance(v,dict) for v in b.values()):
     r[label]=sum(v.get(field,0)-a.get(k,{}).get(field,0) for k,v in b.items())
   r['transfer_counter_scope']='Measured cohort delta. Observation seconds are not GPU DMA time or critical-path time.'
  if stats.exists():
   data=json.loads(stats.read_text());wm=data.get('worker_metrics') or {}
   r['bytes_sent']=sum(v.get('bytes_sent',0) for v in wm.values());r['transfers']=sum(v.get('transfers',0) for v in wm.values())
   r['peak_HBM_bytes_max']=max((v['peak_allocated_bytes'] for v in wm.values() if 'peak_allocated_bytes' in v),default=data.get('local_peak_allocated','NOT_AVAILABLE'))
   for metric in data.get('metrics',[]):
    if metric['request_id'].startswith('r-1_'):continue
    t=metric['timings']
    if not all(k in t for k in ('admitted','prefill_started','prefill_compute_completed','d_received','first_token','finished')):continue
    components=[t['admitted'],t['prefill_started']-t['admitted'],t['prefill_compute_completed']-t['prefill_started'],
     t['d_received']-t['prefill_compute_completed'],t['first_token']-t['d_received'],t['finished']-t['first_token']]
    names=['coordinator_queue','reservation_control','P_assigned_prefill_including_backpressure','uncovered_handoff','D_resume_coda_first_token','remaining_decode_phase']
    line=dict(name=folder.name,request_id=metric['request_id'],coordinator_E2E_ms=t['finished']*1000,evidence='MEASURED_HOST_PHASE_PARTITION_NOT_PURE_GPU_TIME')
    for name,value in zip(names,components):line[name+'_ms']=value*1000;line[name+'_percent']=100*value/t['finished']
    phase.append(line)
  summary.append(r)
 save_csv(ROOT/'experiment_inventory.csv',inventory);save_csv(ROOT/'results_raw.csv',summary);save_csv(ROOT/'e2e_phase_breakdown.csv',phase)
 parity=[]
 for model in ('ouro','huginn','dory'):
  def preferred(label):
   third=ROOT/'raw'/f'{model}_parity3_{label}_P0_C1_s0'
   if (third/'status.json').exists():return third
   second=ROOT/'raw'/f'{model}_parity2_{label}_P0_C1_s0'
   return second if (second/'status.json').exists() else ROOT/'raw'/f'{model}_parity_{label}_P0_C1_s0'
  base={r['sample_id']:r for r in requests(preferred('native'))}
  for label,reference in [('PD',base),('A',base),('B',{r['sample_id']:r for r in requests(preferred('B_control'))})]:
   candidate={r['sample_id']:r for r in requests(preferred(label))}
   ids=sorted(reference.keys()&candidate.keys())
   if not ids:continue
   equal=sum(reference[i]['output_ids']==candidate[i]['output_ids'] for i in ids)
   parity.append(dict(model=model,method=label,equal=equal,N=len(ids),agreement_percent=100*equal/len(ids),
    reference='matched staged G1 / C256 PD' if label=='B' else 'native RLT',evidence='MEASURED_BOUNDED_NATURAL_GENERATION',
    output_cap=32,bounded_RLT_parity_pass=len(ids)>=100 and equal==len(ids),official_quality_validated=False))
 save_csv(ROOT/'parity.csv',parity)
 print('SUMMARY',len(summary),'PHASE_ROWS',len(phase),'PARITY',parity)

def calibrate():
 result={}
 for model in ('ouro','huginn','dory'):
  points=[];decode=[]
  for n in (128,512,2048):
   f=ROOT/'raw'/f'{model}_calibration_P{n}_C1_s0'
   alternate=ROOT/'raw'/f'{model}_calibration2_P{n}_C1_s0'
   if (alternate/'closed_stats.json').exists():f=alternate
   if not (f/'closed_stats.json').exists():continue
   stats=json.loads((f/'closed_stats.json').read_text())
   for row in stats['metrics'][1:]:
    t=row['timings'];points.append((n,t['prefill_compute_completed']-t['prefill_started']))
   for row in requests(f):decode.append((n,row['tpot_ms']/1000))
  if len({x for x,y in points})<2:continue
  p=np.polyfit(*np.asarray(points).T,1);d=np.polyfit(*np.asarray(decode).T,1)
  result[model]=dict(prefill_per_token=max(float(p[0]),0),prefill_intercept=max(float(p[1]),0),
   decode_per_context_token=max(float(d[0]),0),decode_intercept=max(float(d[1]),.0001),
   source='C1 warmed live PD calibration. No online future output lengths.',samples=len(points),evidence='MODEL_FIT_TO_MEASURED_PHASE_TIMES')
  ratios={'1':1.}
  for width in (2,4,8):
   values=[]
   for n in (128,512):
    f=ROOT/'raw'/f'{model}_calibration_P{n}_C{width}_s0'
    alternate=ROOT/'raw'/f'{model}_calibration2_P{n}_C{width}_s0'
    if (alternate/'closed_stats.json').exists():f=alternate
    rows=requests(f)
    if rows:
     base=max(.0001,result[model]['decode_intercept']+result[model]['decode_per_context_token']*n)
     values.extend(r['tpot_ms']/1000/base for r in rows)
   if values:ratios[str(width)]=float(np.median(values))
  result[model]['observed_decode_batch_ratios']=ratios
 write(ROOT/'configs/service_calibration.json',result);print('CALIBRATION',result)

def main():
 p=argparse.ArgumentParser();p.add_argument('--calibrate',action='store_true');a=p.parse_args()
 if a.calibrate:calibrate()
 summarize()

if __name__=='__main__':main()
