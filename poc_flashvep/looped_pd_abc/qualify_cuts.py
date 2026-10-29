"""Qualify a tuned A cut before its repeated performance evaluation."""
import csv
import json
from .common import ROOT,write
from .analyze import requests
from .compare import complete,agreement
from .dispatch import execute
from .scope import active_models

DEFAULT={'ouro':2,'huginn':16,'dory':1}

def discovery_cut(model,length,sweep_prefix='A_sweep_'):
 if sweep_prefix!='A_sweep_':
  from .evidence_protocol import check
  import numpy as np
  candidates=[]
  for folder in sorted((ROOT/'raw').glob(f'{model}_{sweep_prefix}*_P{length}_C4_s0')):
   label=folder.name[len(model+'_'+sweep_prefix):-len(f'_P{length}_C4_s0')]
   if not label.isdigit() or not complete(folder) or not check(folder)['valid']:continue
   data=requests(folder)
   if data:candidates.append((float(np.percentile([r['e2e_ms'] for r in data],95)),int(label)))
  return min(candidates)[1] if candidates else DEFAULT[model]
 path=ROOT/'results_raw.csv';candidates=[]
 if path.exists():
  for row in csv.DictReader(path.open()):
   prefix=f'{model}_A_sweep_';suffix=f'_P{length}_C4_s0'
   if not row['name'].startswith(prefix) or not row['name'].endswith(suffix):continue
   label=row['name'][len(prefix):-len(suffix)]
   if label.isdigit():candidates.append((float(row['e2e_ms_p95']),int(label)))
 return min(candidates)[1] if candidates else DEFAULT[model]

def qualify(sweep_prefix='A_sweep_',qualification_prefix='cut',output='chosen_cuts.json'):
 from .campaign import job
 choices={m:{str(P):discovery_cut(m,P,sweep_prefix) for P in (512,2048)} for m in active_models()}
 jobs=[]
 for model,pair in [('ouro',(0,1)),('huginn',(2,3)),('dory',(4,5))]:
  if model not in active_models():continue
  opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
  for cut in sorted(set(choices[model].values())-{DEFAULT[model]}):
   extra=opt+['--wavefront','--enable-recurrence-split','--split-cut',str(cut)]
   jobs.append(job(model,f'{qualification_prefix}{cut}_qualification',pair,length=0,count=100,cap=32,extra=extra+['--chunk','2048']))
   check=job(model,f'{qualification_prefix}{cut}_identity',pair,length=512,count=2,cap=8,extra=extra+['--chunk','512'])
   check['env']['ABC_VERIFY_KV']='1';jobs.append(check)
 manifest=ROOT/'configs'/('tuned_cut_qualification.json' if output=='chosen_cuts.json' else 'clean_tuned_cut_qualification.json');write(manifest,jobs)
 if jobs:execute(manifest)
 output_path=ROOT/'configs'/output
 result=json.loads(output_path.read_text()) if output_path.exists() else {}
 for model in active_models():
  version={'ouro':'parity','huginn':'parity2','dory':'parity3'}[model]
  native=requests(ROOT/'raw'/f'{model}_{version}_native_P0_C1_s0');result[model]={}
  for P,cut in choices[model].items():
   if cut==DEFAULT[model]:
    result[model][P]=dict(cut=cut,qualified=True,reason='Existing 100-prompt and exact-byte qualification',proposed_cut=cut);continue
   pilot=ROOT/'raw'/f'{model}_{qualification_prefix}{cut}_qualification_P0_C1_s0'
   heavy=ROOT/'raw'/f'{model}_{qualification_prefix}{cut}_identity_P512_C1_s0'
   compare=agreement(native,requests(pilot))
   checks=[json.loads(line) for path in heavy.glob('KV_checks*.jsonl') for line in path.read_text().splitlines()]
   good=complete(pilot) and complete(heavy) and compare['N']==100 and compare['exact']==100 and len(checks)>=3 and all(r['passed'] for r in checks)
   result[model][P]=dict(cut=cut if good else DEFAULT[model],proposed_cut=cut,qualified=good,
    exact_outputs=compare['exact'],N=compare['N'],byte_checks=len(checks),
    reason='Tuned cut passed independent correctness qualification' if good else 'Unqualified tuned cut rejected. Retain the independently qualified original cut.')
 write(ROOT/'configs'/output,result)

if __name__=='__main__':qualify()
