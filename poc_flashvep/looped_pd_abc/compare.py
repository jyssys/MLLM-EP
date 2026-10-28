"""Analyze completed GPU runs. Bootstrap intervals do not replace fidelity gates."""
import argparse
import json
import numpy as np
from .common import ROOT,write
from .analyze import load_rows,requests,save_csv,summarize

def complete(folder):
 path=folder/'status.json'
 return path.exists() and json.loads(path.read_text()).get('status')=='COMPLETE'

def wilson(correct,total):
 if not total:return [None,None]
 z=1.959963984540054;p=correct/total;d=1+z*z/total
 center=(p+z*z/(2*total))/d
 half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
 return [float(100*(center-half)),float(100*(center+half))]

def agreement(left,right):
 a={r['sample_id']:r for r in left};b={r['sample_id']:r for r in right}
 ids=sorted(a.keys()&b.keys());exact=lengths=inputs=depths=depth_compared=0;divergences=[]
 for i in ids:
  x=a[i];y=b[i];u=x['output_ids'];v=y['output_ids']
  exact+=u==v;lengths+=len(u)==len(v);inputs+=x['input_ids']==y['input_ids']
  if 'exit_depths' in x and 'exit_depths' in y:
   depths+=x['exit_depths']==y['exit_depths'];depth_compared+=1
  if u!=v:
   first=next((j for j,(p,q) in enumerate(zip(u,v)) if p!=q),min(len(u),len(v)))
   divergences.append(dict(sample_id=i,first_mismatch=first,left_length=len(u),right_length=len(v)))
 return dict(N=len(ids),exact=exact,length_equal=lengths,input_equal=inputs,
  depth_compared=depth_compared,depth_equal=depths if depth_compared else 'NOT_AVAILABLE',
  agreement_percent=100*exact/len(ids) if ids else None,wilson_95_percent=wilson(exact,len(ids)),
  first_divergences=divergences)

def reference_checks():
 result=[]
 for model,version,refname in [('ouro','parity','ouro_HF_4576_natural100'),
   ('huginn','parity2','huginn_HF_4576_natural100'),('dory','parity3','dory_official_B1_100')]:
  native=ROOT/'raw'/f'{model}_{version}_native_P0_C1_s0';reference=ROOT/'raw'/refname
  if not complete(native) or not complete(reference):continue
  a=requests(native);b=load_rows(reference/'requests.jsonl')
  if model=='dory':assert all(r['kv_layout']=='per_loop' and r['batch_size']==1 for r in b)
  row=agreement(a,b);detail=row.pop('first_divergences')
  row.update(model=model,native=native.name,reference=refname,evidence='MEASURED_BOUNDED_REFERENCE_GENERATION',
   output_cap=32,downstream_accuracy='NOT_MEASURED',reference_equal=row['exact']==row['N'])
  result.append(row);write(ROOT/'state'/f'{model}_reference_divergences.json',detail)
  extra=ROOT/'raw'/'ouro_HF_4576_explicit_last_R4_100'
  if model=='ouro' and complete(extra):
   fixed=agreement(a,load_rows(extra/'requests.jsonl'));mismatches=fixed.pop('first_divergences')
   fixed.update(model=model,native=native.name,reference=extra.name,
    evidence='MEASURED_BOUNDED_REFERENCE_GENERATION_EXPLICIT_LAST_R4',output_cap=32,
    downstream_accuracy='NOT_MEASURED',reference_equal=fixed['exact']==fixed['N'],
    protocol='Explicit zero-based exit_at_step=3. Separate from the unchanged original q=1 reference.')
   result.append(fixed);write(ROOT/'state'/'ouro_explicit_R4_reference_divergences.json',mismatches)
 save_csv(ROOT/'reference_agreement.csv',result)
 transfer=[]
 for folder in sorted((ROOT/'raw').iterdir()):
  if not folder.is_dir():continue
  rows=[r for f in folder.glob('KV_checks*.jsonl') for r in load_rows(f)]
  for i,r in enumerate(rows):
   transfer.append(dict(run=folder.name,check=i,passed=r['passed'],
    KV_valid_bytes=r['actual']['KV_valid_bytes'],hidden_bytes=r['actual']['hidden_bytes'],
    evidence=r['evidence'],includes_warmup=True))
 save_csv(ROOT/'transfer_identity.csv',transfer)
 return result

def bootstrap(blocks,metric='e2e_ms',quantile=95,samples=3000,seed=20261029):
 """Resample independent starts, then paired request IDs inside each start."""
 rng=np.random.default_rng(seed);values=[]
 arrays=[(np.asarray([a[metric] for a,b in block]),np.asarray([b[metric] for a,b in block])) for block in blocks]
 if not arrays:return dict(speedup=None,ci95=None)
 left=np.concatenate([a for a,b in arrays]);right=np.concatenate([b for a,b in arrays])
 ratio=float(np.percentile(left,quantile)/np.percentile(right,quantile))
 for _ in range(samples):
  p=[];q=[]
  for j in rng.integers(0,len(arrays),len(arrays)):
   a,b=arrays[j];ids=rng.integers(0,len(a),len(a));p.extend(a[ids]);q.extend(b[ids])
  values.append(np.percentile(p,quantile)/np.percentile(q,quantile))
 return dict(speedup=ratio,ci95=np.percentile(values,[2.5,97.5]).tolist())

def paired_runs(model,base,method,length,concurrency,kind='serving'):
 from .frozen_runs import preferred,matches
 blocks=[];exact=total=length_equal=0;raw_pairs=[]
 frozen_starts=0
 for repeat in range(3):
  names=[f'{model}_{label}_P{length}_C{concurrency}_s{repeat}' for label in (base,method)]
  names=[preferred(name) for name in names];folders=[ROOT/'raw'/name for name in names]
  if not all(complete(p) for p in folders):continue
  a,b=[{r['sample_id']:r for r in requests(p)} for p in folders]
  ids=sorted(a.keys()&b.keys())
  if not ids or len(ids)!=len(a) or len(ids)!=len(b):continue
  if not all(a[i]['input_ids']==b[i]['input_ids'] for i in ids):continue
  blocks.append([(a[i],b[i]) for i in ids]);total+=len(ids)
  frozen_starts+=all(matches(folder) for folder in folders)
  exact+=sum(a[i]['output_ids']==b[i]['output_ids'] for i in ids)
  length_equal+=sum(a[i]['output_tokens']==b[i]['output_tokens'] for i in ids)
  raw_pairs.append(names)
 if not blocks:return None
 E=bootstrap(blocks);T=bootstrap(blocks,'ttft_ms')
 left=[a for block in blocks for a,b in block];right=[b for block in blocks for a,b in block]
 return dict(model=model,kind=kind,baseline=base,candidate=method,input_tokens=length,concurrency=concurrency,
  starts=len(blocks),N=total,exact_outputs=exact,length_equal=length_equal,
  output_agreement_percent=100*exact/total,p95_E2E_speedup=E['speedup'],p95_E2E_speedup_ci95=E['ci95'],
  p95_TTFT_speedup=T['speedup'],p95_TTFT_speedup_ci95=T['ci95'],
  baseline_p95_E2E_ms=float(np.percentile([r['e2e_ms'] for r in left],95)),
  candidate_p95_E2E_ms=float(np.percentile([r['e2e_ms'] for r in right],95)),
  baseline_p95_TTFT_ms=float(np.percentile([r['ttft_ms'] for r in left],95)),
  candidate_p95_TTFT_ms=float(np.percentile([r['ttft_ms'] for r in right],95)),
  strict_same_output=exact==total,three_or_more_independent_starts=len(blocks)>=3,
  matched_frozen_runtime_starts=frozen_starts,
  evidence='MEASURED_WITH_PAIRED_HIERARCHICAL_BOOTSTRAP',raw_pairs=raw_pairs,
  limitation='Closed-loop bounded natural generation. This is not sustainable open-loop capacity.')

def comparisons():
 output=[]
 for model in ('ouro','huginn','dory'):
  for length,width in ((128,1),(512,4),(2048,8)):
   for method in ('A','B'):
    for base in ('DP','DP_Mixed','PD','PD_G1','PD_A_control','PD_A_coalesced'):
     row=paired_runs(model,base,method,length,width)
     if row:output.append(row)
  for base in ('C_upstream','C_queue','C_token','C_service','C_DP4','C_DP4_Mixed','C_PD1','C_PD3'):
   row=paired_runs(model,base,'C_loop',0,8,kind='C')
   if row:output.append(row)
  for method in ('A','B'):
   for base in ('DP','DP_Mixed','PD','PD_G1','PD_A_control','PD_A_coalesced'):
    row=paired_runs(model,'L128_'+base,'L128_'+method,512,4,kind='serving_long128')
    if row:output.append(row)
 save_csv(ROOT/'paired_comparisons.csv',output)
 return output

def slo_results():
 path=ROOT/'configs/frozen_slo.json'
 if not path.exists():return []
 grid=json.loads(path.read_text());rows=[]
 for folder in sorted((ROOT/'raw').iterdir()):
  if not folder.is_dir() or not complete(folder):continue
  raw=requests(folder)
  if not raw or 'token_events' not in raw[0]:continue
  model=raw[0]['model']
  if model not in grid:continue
  span=max(x['finish'] for x in raw)-min(x['arrival'] for x in raw)
  thresholds=dict(grid[model]['ITL_thresholds_ms'])
  thresholds.update({f'absolute_{limit}ms':limit for limit in grid[model].get('absolute_ITL_grid_ms',[])})
  for level,limit in thresholds.items():
   for ttft in grid[model]['TTFT_grid_ms']:
    passed=[]
    for x in raw:
     # The first token in a coalesced emission still exposes its full waiting gap.
     # Unknown internal token times must not erase a client-visible stall.
     intervals=[1000*(e['time']-e['previous_time']) for e in x['token_events'] if not e['first'] and e['new_tokens']>=1]
     normal=x['finish_reason'] in ('eos','length','stop') and x['output_tokens']>0
     passed.append(normal and x['ttft_ms']<=ttft and max(intervals,default=0)<=limit and x['e2e_ms']<=grid[model]['E2E_limit_ms'])
    count=sum(passed)
    rows.append(dict(run=folder.name,model=model,level=level,ITL_limit_ms=limit,TTFT_limit_ms=ttft,
     E2E_limit_ms=grid[model]['E2E_limit_ms'],correct_completions=count,N=len(raw),
     attainment_percent=100*count/len(raw),wilson_95_percent=wilson(count,len(raw)),
     SLO_goodput_requests_s=count/span,evidence='MEASURED_BOUNDED_REQUEST_SLO',
     sustainable_capacity='NOT_MEASURED' if not raw[0].get('arrival_rate') else 'REQUIRES_QUEUE_STABILITY_AND_RATE_SWEEP',
     length_terminations=sum(x['finish_reason']=='length' for x in raw),
     completion_scope='A length termination is valid only for this bounded workload. It does not establish answer completeness.'))
 save_csv(ROOT/'slo_results.csv',rows);return rows

def contention_results():
 data=[]
 for folder in sorted((ROOT/'raw').glob('ouro_contention*')):
  if not complete(folder):continue
  raw=requests(folder)
  if raw:data.append(dict(run=folder.name,gpus=raw[0]['gpus'],requests=len(raw),
   E2E_mean_ms=float(np.mean([r['e2e_ms'] for r in raw])),
   TPOT_mean_ms=float(np.mean([r['tpot_ms'] for r in raw])),
   evidence='MEASURED_CONTENDING_LANES' if 'parallel' in folder.name else 'MEASURED_ISOLATED_LANE'))
 save_csv(ROOT/'contention.csv',data);return data

def scale_discovery():
 rows=[]
 for prefix,length,width in [('scale4',512,8),('scale8',512,8),('pressure4',0,32),('pressure8',0,32)]:
  for model in ('ouro','huginn','dory'):
   candidates=('A1','A2','A4','B1','C_loop')
   for candidate in candidates:
    target=ROOT/'raw'/f'{model}_{prefix}_{candidate}_P{length}_C{width}_s0'
    if not complete(target):continue
    right=requests(target)
    for baseline in ('DP','DP_Mixed','PD1','PD2','PD3','PD4','PD1_G1','PD_A1','PD_A2','PD_A4','C_service'):
     control=ROOT/'raw'/f'{model}_{prefix}_{baseline}_P{length}_C{width}_s0'
     if not complete(control):continue
     left=requests(control);match=agreement(left,right)
     if match['input_equal']!=len(left) or len(left)!=len(right):continue
     before=float(np.percentile([r['e2e_ms'] for r in left],95));after=float(np.percentile([r['e2e_ms'] for r in right],95))
     rows.append(dict(model=model,regime=prefix,baseline=baseline,candidate=candidate,
      concurrency=width,starts=1,N=len(left),exact_outputs=match['exact'],length_equal=match['length_equal'],
      baseline_p95_E2E_ms=before,candidate_p95_E2E_ms=after,p95_E2E_speedup=before/after,
      confidence_interval='NOT_MEASURED_ONE_START',accepted_gain=False,
      evidence='MEASURED_SINGLE_START_DISCOVERY_NOT_ACCEPTED_SPEEDUP'))
 save_csv(ROOT/'scale_track_comparisons.csv',rows)

def heterogeneous_discovery():
 rows=[]
 for model in ('ouro','huginn','dory'):
  target=ROOT/'raw'/f'{model}_Chet_loop_P0_C16_s0'
  if not complete(target):continue
  right=requests(target)
  for label in ('upstream','queue','token','service','DP4_Mixed','PD1','PD3'):
   control=ROOT/'raw'/f'{model}_Chet_{label}_P0_C16_s0'
   if not complete(control):continue
   left=requests(control);match=agreement(left,right)
   if match['input_equal']!=len(left) or len(left)!=len(right):continue
   before=float(np.percentile([r['e2e_ms'] for r in left],95));after=float(np.percentile([r['e2e_ms'] for r in right],95))
   rows.append(dict(model=model,baseline=label,candidate='loop',starts=1,N=len(left),
    exact_outputs=match['exact'],length_equal=match['length_equal'],
    baseline_p95_E2E_ms=before,candidate_p95_E2E_ms=after,p95_E2E_speedup=before/after,
    confidence_interval='NOT_MEASURED_ONE_START',accepted_gain=False,
    workload='Synthetic padded questions, mixed 128/256/512/2048 prompt tokens, C16, cap64',
    evidence='MEASURED_SINGLE_START_DISCOVERY_NOT_ACCEPTED_SPEEDUP'))
 save_csv(ROOT/'heterogeneous_C_comparisons.csv',rows)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--comparisons',action='store_true');args=parser.parse_args()
 summarize();reference_checks();slo_results();contention_results();scale_discovery();heterogeneous_discovery()
 from .critical_path_analysis import main as critical_path_analysis
 critical_path_analysis()
 if args.comparisons:comparisons()

if __name__=='__main__':main()
