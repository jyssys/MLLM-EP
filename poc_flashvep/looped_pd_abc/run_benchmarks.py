"""Run independent A/B/C comparisons after qualification. Resume skips completed jobs."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write
from .campaign import job
from .dispatch import execute
from .analyze import summarize,calibrate,requests
from . import resources
from .scope import active_models

def wait_for_quiet_lease():
 deadline=time.monotonic()+1200
 while True:
  owned=resources.base.burners();active=[]
  for row in resources.base.processes():
   try:
    if os.getsid(row['pid'])!=owned.get(row['gpu']):active.append(row)
   except ProcessLookupError:pass
  if not active:return
  if time.monotonic()>deadline:raise TimeoutError('A non-burner GPU process remains. No process was signaled.')
  time.sleep(3)

def freeze_slo():
 result={}
 for model,version in [('ouro','parity'),('huginn','parity2'),('dory','parity3')]:
  rows=requests(ROOT/'raw'/f'{model}_{version}_native_P0_C1_s0')
  intervals=[1000*(e['time']-e['previous_time']) for r in rows for e in r['token_events'] if not e['first'] and e['new_tokens']==1]
  median=float(np.median(intervals))
  result[model]=dict(source_run=f'{model}_{version}_native_P0_C1_s0',decode_ITL_median_ms=median,
   ITL_thresholds_ms=dict(strict=5*median,moderate=10*median,relaxed=25*median),
   absolute_ITL_grid_ms=[100,150,200,250],TTFT_grid_ms=[1000,5000,10000],E2E_limit_ms=30000,
   definition='Request TTFT and every emitted single-token ITL must pass. Requested cap termination is a valid bounded serving completion.',
   limitation='This is not downstream answer completeness or sustainable open-loop capacity.',evidence='FROZEN_CONTRACT_FROM_NATIVE_DECODE_ONLY_INTERVALS')
 write(ROOT/'configs/frozen_slo.json',result)

def initial_checks():
 checks=[]
 for model,pair,cut in [('ouro',(0,1),2),('huginn',(2,3),16),('dory',(4,5),1)]:
  for label in ('A','B'):
   extra=['--graphs','--chunk','512','--wavefront']+(['--asynchronous'] if model=='ouro' else [])
   if label=='A':extra+=['--enable-recurrence-split','--split-cut',str(cut)]
   else:extra+=['--enable-frontier-handoff']
   j=job(model,'KV_identity_'+label,pair,length=512,count=2,cap=8,extra=extra)
   j['env'].update(ABC_VERIFY_KV='1',ABC_REVERSE_FRAGMENTS='1');checks.append(j)
 manifest=ROOT/'configs/KV_identity_checks.json';write(manifest,checks);execute(manifest)
 # Preserve independent gates. One failed track must not cancel another track.
 gates={}
 for j in checks:
  folder=ROOT/'raw'/j['name'];status=json.loads((folder/'status.json').read_text())
  rows=[json.loads(line) for path in folder.glob('KV_checks*.jsonl') for line in path.read_text().splitlines()]
  gates[j['name']]=dict(passed=status.get('status')=='COMPLETE' and len(rows)>=3 and all(r['passed'] for r in rows),
   checks=len(rows),status=status.get('status'))
 write(ROOT/'state/byte_identity_gates.json',gates)

def apply_identity_gates(manifest):
 path=ROOT/'state/byte_identity_gates.json'
 if not path.exists():return
 gates=json.loads(path.read_text());jobs=json.loads(manifest.read_text());kept=[];blocked=[]
 for j in jobs:
  if '--model' not in j['args']:kept.append(j);continue
  model=j['args'][j['args'].index('--model')+1]
  track='A' if '--enable-recurrence-split' in j['args'] else 'B' if '--enable-frontier-handoff' in j['args'] else None
  key=f'{model}_KV_identity_{track}_P512_C1_s0'
  if track and not gates.get(key,{}).get('passed',False):
   blocked.append(dict(run=j['name'],reason='Independent '+track+' byte-identity gate failed',model=model))
  else:kept.append(j)
 if blocked:
  write(manifest.with_suffix('.blocked.json'),blocked);write(manifest,kept)

def contention():
 jobs=[]
 for repeat in range(3):
  j=job('ouro','contention_quiet',(0,1),length=512,count=6,cap=32,extra=['--graphs','--asynchronous','--chunk','256'],repeat=repeat)
  j['quiet']=True;jobs.append(j)
  for p in range(4):
   pair=(2*p,2*p+1)
   j=job('ouro',f'contention_parallel_pair{p}',pair,length=512,count=6,cap=32,
    extra=['--graphs','--asynchronous','--chunk','256','--barrier',f'pair_contention_{repeat}','--barrier-count','4'],repeat=repeat)
   jobs.append(j)
 manifest=ROOT/'configs/contention_checks.json';write(manifest,jobs);execute(manifest)

def profiles():
 jobs=[]
 for model in active_models():
  opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
  boundary=['--enable-boundary-adapter'] if model!='ouro' else []
  for label in ('PD','A','B','C'):
   extra=opt+boundary+['--profile','--chunk','256']
   if label=='A':extra+=['--wavefront','--enable-recurrence-split','--split-cut',str({'ouro':2,'huginn':16,'dory':1}[model])]
   if label=='B':extra+=['--wavefront','--enable-frontier-handoff']
   if label=='C':extra+=['--wavefront','--placement','loop']
   j=job(model,'profile_'+label,(0,1),length=128,count=2,cap=4,extra=extra);j['quiet']=True;jobs.append(j)
 manifest=ROOT/'configs/profile_checks.json';write(manifest,jobs);execute(manifest)

def main():
 review=ROOT/'state/post_qualification_review.json'
 while not review.exists():time.sleep(3)
 wait_for_quiet_lease();calibrate();summarize();freeze_slo()
 write(ROOT/'state/main_campaign.json',dict(status='RUNNING',phase='byte_identity',pid=os.getpid(),time=time.time()))
 initial_checks();contention()
 for phase in ('A_sweep','serving','C','scale'):
  write(ROOT/'state/main_campaign.json',dict(status='RUNNING',phase=phase,pid=os.getpid(),time=time.time()))
  subprocess.run([PYTHON,'-m','poc_flashvep.looped_pd_abc.campaign',phase],cwd=REPO,check=True)
  manifest=ROOT/'configs'/f'{phase}_campaign.json';apply_identity_gates(manifest)
  execute(manifest);summarize()
 profiles();summarize()
 write(ROOT/'state/main_campaign.json',dict(status='COMPLETE_REVIEW_REQUIRED',pid=os.getpid(),time=time.time(),
  pending=['inspect failures and CIs','conditional open-loop extension','complete reports','scoped commit and push']))

if __name__=='__main__':main()
