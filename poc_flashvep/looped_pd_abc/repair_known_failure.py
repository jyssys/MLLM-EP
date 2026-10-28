"""Retry only the fixed stats-import failure. Preserve each failed attempt."""
import json
import uuid
from .common import ROOT,write
from .dispatch import execute

def repair():
 manifest=ROOT/'configs/serving_campaign.json'
 if not manifest.exists():return
 state=ROOT/'state/known_instrumentation_retries.json'
 previous=json.loads(state.read_text()) if state.exists() else []
 jobs=[]
 for job in json.loads(manifest.read_text()):
  name=job['name'];folder=ROOT/'raw'/name;status=folder/'status.json'
  if name in previous or not status.exists():continue
  data=json.loads(status.read_text());log=ROOT/'logs'/f'{name}.log'
  if data.get('status')!='FAILED' or not log.exists():continue
  if "NameError: name 'os' is not defined" not in log.read_text():continue
  configuration=folder/'configuration.json'
  write(ROOT/'archives'/f'failed_stats_import_{name}_{uuid.uuid4().hex}.json',dict(status=data,
   configuration=json.loads(configuration.read_text()) if configuration.exists() else None,
   reason='The evidence-IO stats hook omitted an os import. No model computation changed. Retry uses the corrected module.'))
  jobs.append(job);previous.append(name)
 if jobs:
  dest=ROOT/'configs/fixed_stats_import_retries.json';write(dest,jobs);execute(dest)
 write(state,previous)

if __name__=='__main__':repair()
