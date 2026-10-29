"""Paired SLO uncertainty. Bootstrap spans stay fixed within each sampled start."""
import json
import numpy as np
from .common import ROOT
from .analyze import requests,save_csv
from .frozen_runs import preferred,matches
from .scope import active_models

def qualifies(row,itl_limit,ttft_limit,e2e_limit):
 gaps=[1000*(event['time']-event['previous_time']) for event in row['token_events']
  if not event['first'] and event['new_tokens']>=1]
 return (row['finish_reason'] in ('eos','length','stop') and row['output_tokens']>0
  and row['ttft_ms']<=ttft_limit and max(gaps,default=0)<=itl_limit and row['e2e_ms']<=e2e_limit)

def pair(model,baseline,candidate,length,width,draws=3000):
 path=ROOT/'configs/frozen_slo.json'
 if not path.exists():return []
 contract=json.loads(path.read_text())[model];thresholds=dict(contract['ITL_thresholds_ms'])
 thresholds.update({f'absolute_{x}ms':x for x in contract['absolute_ITL_grid_ms']})
 grid=[(level,limit,ttft) for level,limit in thresholds.items() for ttft in contract['TTFT_grid_ms']]
 arrays=[];spans=[];exact=total=frozen=0;names=[];e2e=[[],[]]
 for repeat in range(3):
  run_names=[preferred(f'{model}_{label}_P{length}_C{width}_s{repeat}') for label in (baseline,candidate)]
  folders=[ROOT/'raw'/name for name in run_names]
  if not all((f/'status.json').exists() and json.loads((f/'status.json').read_text()).get('status')=='COMPLETE' for f in folders):continue
  data=[{r['sample_id']:r for r in requests(f)} for f in folders];ids=sorted(data[0])
  if not ids or set(ids)!=set(data[1]) or any(data[0][i]['input_ids']!=data[1][i]['input_ids'] for i in ids):continue
  if arrays and len(ids)!=arrays[0].shape[1]:continue
  flags=np.asarray([[[qualifies(data[side][i],limit,ttft,contract['E2E_limit_ms'])
   for level,limit,ttft in grid] for i in ids] for side in (0,1)],dtype=np.float64)
  arrays.append(flags);spans.append([max(r['finish'] for r in side.values())-min(r['arrival'] for r in side.values()) for side in data])
  exact+=sum(data[0][i]['output_ids']==data[1][i]['output_ids'] for i in ids);total+=len(ids)
  frozen+=all(matches(f) for f in folders);names.append(run_names)
  for side in (0,1):e2e[side].extend(data[side][i]['e2e_ms'] for i in ids)
 if not arrays:return []
 values=np.asarray(arrays);durations=np.asarray(spans);k=len(arrays);n=values.shape[2]
 # Values have dimensions [start, side, request, SLO cell].
 rng=np.random.default_rng(20261029);starts=rng.integers(0,k,size=(draws,k));indices=rng.integers(0,n,size=(draws,k,n))
 counts=[];rates=[]
 for side in (0,1):
  sampled=values[:,side][starts[:,:,None],indices]
  counts.append(sampled.sum(axis=(1,2)));rates.append(counts[-1]/durations[starts,side].sum(axis=1)[:,None])
 ratio=np.divide(rates[1],rates[0],out=np.full_like(rates[0],np.nan),where=rates[0]>0)
 counts_point=values.sum(axis=(0,2));rates_point=counts_point/durations.sum(axis=0)[:,None]
 delta_ci=np.percentile(rates[1]-rates[0],[2.5,97.5],axis=0)
 attainment_ci=np.percentile(100*(counts[1]-counts[0])/(k*n),[2.5,97.5],axis=0)
 output=[]
 for cell,(level,limit,ttft) in enumerate(grid):
  finite=int(np.isfinite(ratio[:,cell]).sum())
  interval=np.percentile(ratio[:,cell],[2.5,97.5]).tolist() if finite==draws else 'UNBOUNDED_OR_UNDEFINED_ZERO_BASELINE_DRAWS'
  output.append(dict(model=model,baseline=baseline,candidate=candidate,input_tokens=length,concurrency=width,
   starts=k,matched_frozen_runtime_starts=frozen,N=total,exact_outputs=exact,level=level,
   accepted_gain=False,uncertainty_scope='Exploratory: fewer than three independent starts' if k<3 else 'Three-start or larger conditional bootstrap. Scientific review still required.',
   ITL_limit_ms=limit,TTFT_limit_ms=ttft,E2E_limit_ms=contract['E2E_limit_ms'],
   baseline_qualifying_requests=int(counts_point[0,cell]),candidate_qualifying_requests=int(counts_point[1,cell]),
   baseline_SLO_goodput_requests_s=float(rates_point[0,cell]),candidate_SLO_goodput_requests_s=float(rates_point[1,cell]),
   SLO_goodput_ratio=float(rates_point[1,cell]/rates_point[0,cell]) if rates_point[0,cell]>0 else 'NOT_DEFINED_ZERO_BASELINE',
   SLO_goodput_ratio_ci95=interval,finite_ratio_bootstrap_draws=finite,bootstrap_draws=draws,
   SLO_goodput_difference_ci95_requests_s=delta_ci[:,cell].tolist(),
   attainment_difference_ci95_pp=attainment_ci[:,cell].tolist(),
   baseline_p95_E2E_ms=float(np.percentile(e2e[0],95)),candidate_p95_E2E_ms=float(np.percentile(e2e[1],95)),
   raw_pairs=names,evidence='MEASURED_WITH_PAIRED_HIERARCHICAL_SLO_BOOTSTRAP',
   limitation='The bootstrap conditions on observed cohort spans. It does not replay changed queueing or establish sustainable open-loop capacity.'))
 return output

def main():
 output=[]
 for model in active_models():
  for length,width in ((128,1),(512,4),(2048,8)):
   for candidate in ('A','B'):
    for baseline in ('DP','DP_Mixed','PD','PD_G1','PD_A_control','PD_A_coalesced'):
     output.extend(pair(model,baseline,candidate,length,width))
  for baseline in ('C_upstream','C_queue','C_token','C_service','C_DP4','C_DP4_Mixed','C_PD1','C_PD3'):
   output.extend(pair(model,baseline,'C_loop',0,8))
  for candidate in ('A','B'):
   for baseline in ('DP','DP_Mixed','PD','PD_G1','PD_A_control','PD_A_coalesced'):
    output.extend(pair(model,'L128_'+baseline,'L128_'+candidate,512,4))
 for baseline,candidate in [('Bpace_generic_confirm','Bpace_frontier_confirm'),('Bpace_generic_confirm','B')]:
  if 'dory' not in active_models():continue
  output.extend(pair('dory',baseline,candidate,2048,8))
 save_csv(ROOT/'paired_SLO_comparisons.csv',output)

if __name__=='__main__':main()
