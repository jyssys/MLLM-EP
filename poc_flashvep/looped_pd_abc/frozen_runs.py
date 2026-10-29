"""Confirm early telemetry epochs on one frozen runtime. Keep original attempts."""
import hashlib
import json
import zipfile
from .common import ROOT,write

RUNTIME_FILES=('worker.py','common.py','nvml_watch.py','replicas.py','split.py','frontier.py',
 'boundary.py','segments.py','placement.py','dory_adapter.py')

def matches(folder):
 from .evidence_protocol import check
 if (ROOT/'configs/evidence_protocol_epoch.json').exists() and not check(folder)['valid']:return False
 freeze=ROOT/'configs/runtime_freeze_final.json';configuration=folder/'configuration.json'
 if not freeze.exists() or not configuration.exists():return False
 lock=json.loads(freeze.read_text());config=json.loads(configuration.read_text())
 archive=folder/f"source-{config.get('arguments',{}).get('attempt_id')}.zip"
 if not archive.exists():return False
 with zipfile.ZipFile(archive) as bundle:
  epoch=ROOT/'configs/evidence_protocol_epoch.json'
  if epoch.exists():
   path='poc_flashvep/looped_pd_abc/telemetry.py'
   if path not in bundle.namelist() or hashlib.sha256(bundle.read(path)).hexdigest()!=json.loads(epoch.read_text())['telemetry_sha256']:return False
  if hashlib.sha256(bundle.read('rlt_changes.patch')).hexdigest()!=lock['patch_sha256']:return False
  for name in RUNTIME_FILES:
   path='poc_flashvep/looped_pd_abc/'+name
   if path not in bundle.namelist() or hashlib.sha256(bundle.read(path)).hexdigest()!=lock['files'].get(path):return False
 return True

def preferred(name):
 path=ROOT/'configs/frozen_serving_replacements.json'
 return json.loads(path.read_text()).get(name,name) if path.exists() else name

def matches_cut(folder,args):
 if '--enable-recurrence-split' not in args:return True
 config=folder/'configuration.json'
 if not config.exists():return False
 actual=json.loads(config.read_text()).get('arguments',{}).get('split_cut')
 return actual==int(args[args.index('--split-cut')+1])

def confirm():
 from .dispatch import execute
 from .scope import enabled
 path=ROOT/'configs/serving_campaign.json'
 if not path.exists():return
 jobs=[];mapping={}
 for job in json.loads(path.read_text()):
  if not enabled(job):continue
  name=job['name'];folder=ROOT/'raw'/name;status=folder/'status.json'
  if '--mode' not in job['args'] or 'qualification' in name or not status.exists():continue
  data=json.loads(status.read_text())
  if data.get('status') not in ('COMPLETE','FAILED'):continue
  arguments=list(job['args'])
  if '--enable-recurrence-split' in arguments and (ROOT/'configs/clean_chosen_cuts.json').exists():
   from .campaign import selected_cut
   model=arguments[arguments.index('--model')+1];length=int(arguments[arguments.index('--length')+1])
   arguments[arguments.index('--split-cut')+1]=str(selected_cut(model,length))
  if data.get('status')=='COMPLETE' and matches(folder) and matches_cut(folder,arguments):continue
  prefix,index=name.rsplit('_s',1);offset=int(index)+3
  while True:
   replacement=f'{prefix}_s{offset}';target=ROOT/'raw'/replacement;marker=target/'status.json'
   if not marker.exists() or json.loads(marker.read_text()).get('status')!='COMPLETE' or (matches(target) and matches_cut(target,arguments)):break
   offset+=3
  updated=dict(job);updated['name']=replacement;updated['args']=arguments
  updated['env']=dict(job.get('env',{}),ABC_ASYNC_EVIDENCE='1')
  updated['args'][updated['args'].index('--name')+1]=replacement
  updated['quiet']=True;jobs.append(updated);mapping[name]=replacement
 manifest=ROOT/'configs/frozen_confirmation_jobs.json';write(manifest,jobs)
 if jobs:execute(manifest)
 write(ROOT/'configs/frozen_serving_replacements.json',mapping)

if __name__=='__main__':confirm()
