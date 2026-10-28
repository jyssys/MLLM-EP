"""Confirm early telemetry epochs on one frozen runtime. Keep original attempts."""
import hashlib
import json
import zipfile
from .common import ROOT,write

RUNTIME_FILES=('worker.py','common.py','nvml_watch.py','replicas.py','split.py','frontier.py',
 'boundary.py','segments.py','placement.py','dory_adapter.py')

def matches(folder):
 freeze=ROOT/'configs/runtime_freeze_final.json';configuration=folder/'configuration.json'
 if not freeze.exists() or not configuration.exists():return False
 lock=json.loads(freeze.read_text());config=json.loads(configuration.read_text())
 archive=folder/f"source-{config.get('arguments',{}).get('attempt_id')}.zip"
 if not archive.exists():return False
 with zipfile.ZipFile(archive) as bundle:
  if hashlib.sha256(bundle.read('rlt_changes.patch')).hexdigest()!=lock['patch_sha256']:return False
  for name in RUNTIME_FILES:
   path='poc_flashvep/looped_pd_abc/'+name
   if path not in bundle.namelist() or hashlib.sha256(bundle.read(path)).hexdigest()!=lock['files'].get(path):return False
 return True

def preferred(name):
 path=ROOT/'configs/frozen_serving_replacements.json'
 return json.loads(path.read_text()).get(name,name) if path.exists() else name

def confirm():
 from .dispatch import execute
 path=ROOT/'configs/serving_campaign.json'
 if not path.exists():return
 jobs=[];mapping={}
 for job in json.loads(path.read_text()):
  name=job['name'];folder=ROOT/'raw'/name;status=folder/'status.json'
  if '--mode' not in job['args'] or 'qualification' in name or not status.exists():continue
  data=json.loads(status.read_text())
  if data.get('status')!='COMPLETE' or matches(folder):continue
  prefix,index=name.rsplit('_s',1);offset=int(index)+3
  while True:
   replacement=f'{prefix}_s{offset}';target=ROOT/'raw'/replacement;marker=target/'status.json'
   if not marker.exists() or json.loads(marker.read_text()).get('status')!='COMPLETE' or matches(target):break
   offset+=3
  updated=dict(job);updated['name']=replacement;updated['args']=list(job['args'])
  updated['args'][updated['args'].index('--name')+1]=replacement
  updated['quiet']=True;jobs.append(updated);mapping[name]=replacement
 manifest=ROOT/'configs/frozen_confirmation_jobs.json';write(manifest,jobs)
 if jobs:execute(manifest)
 write(ROOT/'configs/frozen_serving_replacements.json',mapping)

if __name__=='__main__':confirm()
