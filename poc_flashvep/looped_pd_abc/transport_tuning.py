"""Conditional generic-pacing control for the observed Dory B SLO region."""
import json
from .common import ROOT,write
from .analyze import requests,save_csv
from .slo_compare import pair,qualifies

def paced(label,chunk,inflight,repeat=0,frontier=False):
 from .campaign import job
 group=(0,1) if repeat%2==0 else (6,7)
 extra=['--graphs','--wavefront','--chunk','256','--enable-boundary-adapter']
 if frontier:extra+=['--enable-frontier-handoff']
 result=job('dory',label,group,length=2048,count=24,cap=32,concurrency=8,extra=extra,repeat=repeat)
 result['module']='poc_flashvep.looped_pd_abc.transport_worker';result['quiet']=True
 result['env'].update(ABC_TRANSPORT_RUN=result['name'],ABC_TRANSPORT_CHUNK_MIB=str(chunk),ABC_TRANSPORT_INFLIGHT_MIB=str(inflight))
 return result

def run():
 from .dispatch import execute
 candidates=pair('dory','PD_G1','B',2048,8,draws=200)
 pilot=next((r for r in candidates if r['level']=='absolute_100ms' and r['TTFT_limit_ms']==10000),None)
 if not pilot or pilot['starts']<3 or pilot['matched_frozen_runtime_starts']<3 or pilot['exact_outputs']!=pilot['N'] or not isinstance(pilot['SLO_goodput_ratio'],float) or pilot['SLO_goodput_ratio']<1.10:
  write(ROOT/'state/B_generic_pacing_followup.json',dict(status='NOT_RUN_GATE_NOT_MET',pilot=pilot,
   reason='No three-start frozen, output-matched default-B signal above the predeclared exploratory threshold. C continues independently.'))
  return
 pairs=((8,32),(16,32),(16,64),(32,64),(64,64),(64,128),(16,256),(32,256),(64,256))
 jobs=[paced(f'Bpace_tune_c{chunk}_i{inflight}',chunk,inflight) for chunk,inflight in pairs]
 manifest=ROOT/'configs/B_generic_pacing_tuning.json';write(manifest,jobs);execute(manifest)
 from .frozen_runs import preferred
 canonical={r['sample_id']:r for r in requests(ROOT/'raw'/preferred('dory_PD_G1_P2048_C8_s0'))};scores=[]
 for job,(chunk,inflight) in zip(jobs,pairs):
  folder=ROOT/'raw'/job['name'];status=folder/'status.json'
  if not status.exists() or json.loads(status.read_text()).get('status')!='COMPLETE':continue
  data=requests(folder)
  if not data:continue
  span=max(r['finish'] for r in data)-min(r['arrival'] for r in data)
  count=sum(qualifies(r,100,10000,30000) for r in data)
  identical=(len(data)==len(canonical) and all(r['sample_id'] in canonical and r['input_ids']==canonical[r['sample_id']]['input_ids']
   and r['output_ids']==canonical[r['sample_id']]['output_ids'] for r in data))
  scores.append(dict(chunk_MiB=chunk,inflight_MiB=inflight,run=job['name'],N=len(data),SLO_qualifying=count,
   SLO_goodput_requests_s=count/span,selection_eligible=identical,reason='Exact canonical output match' if identical else 'Different output workload. Not selection eligible.'))
 save_csv(ROOT/'B_transport_tuning.csv',scores)
 write(ROOT/'configs/B_generic_pacing_tuning_results.json',scores)
 eligible=[row for row in scores if row['selection_eligible']]
 if not eligible:
  write(ROOT/'state/B_generic_pacing_followup.json',dict(status='BLOCKED_NO_OUTPUT_MATCHED_TUNING_CASE',pilot=pilot,tuning=scores));return
 # Tie-breaking prefers the less restrictive transport. No answer label enters this score.
 best=max(eligible,key=lambda row:(row['SLO_goodput_requests_s'],row['chunk_MiB'],row['inflight_MiB']))
 write(ROOT/'configs/B_generic_pacing_selection.json',dict(pilot=pilot,tuning=scores,selected=best,
  limitation='Pilot-selected secondary operating region. Tuning runs are not accepted speedup evidence.'))
 confirmation=[]
 for repeat in range(3):
  labels=[('Bpace_generic_confirm',False),('Bpace_frontier_confirm',True)]
  if repeat%2:labels.reverse()
  for label,frontier in labels:
   confirmation.append(paced(label,best['chunk_MiB'],best['inflight_MiB'],repeat,frontier))
 target=ROOT/'configs/B_generic_pacing_confirmation.json';write(target,confirmation);execute(target)
 profiles=[]
 for label,chunk,inflight,frontier in [('Bpace_profile_default',64,256,False),
   ('Bpace_profile_generic',best['chunk_MiB'],best['inflight_MiB'],False),
   ('Bpace_profile_frontier',best['chunk_MiB'],best['inflight_MiB'],True)]:
  j=paced(label,chunk,inflight,frontier=frontier)
  for option,value in [('--requests','8'),('--cap','8')]:j['args'][j['args'].index(option)+1]=value
  j['args'].append('--profile');profiles.append(j)
 profile_manifest=ROOT/'configs/B_generic_pacing_profiles.json';write(profile_manifest,profiles);execute(profile_manifest)
 outcomes={j['name']:json.loads((ROOT/'raw'/j['name']/'status.json').read_text()).get('status')
  if (ROOT/'raw'/j['name']/'status.json').exists() else 'NOT_MEASURED' for j in confirmation+profiles}
 success=all(value=='COMPLETE' for value in outcomes.values())
 write(ROOT/'state/B_generic_pacing_followup.json',dict(status='COMPLETE_REVIEW_REQUIRED' if success else 'PARTIAL_FAILURE_REVIEW_REQUIRED',
  selection=best,outcomes=outcomes,accepted_gain=False,
  warning='These are control and diagnostic attempts, not a GO decision. Reference fidelity and novelty gates remain.'))
