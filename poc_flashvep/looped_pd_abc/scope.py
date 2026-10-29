"""Apply explicit user model deferrals without changing frozen model execution."""
import json
from .common import ROOT

KNOWN_MODELS=('ouro','huginn','dory')

def active_models():
 path=ROOT/'configs/model_scope.json'
 if not path.exists():return KNOWN_MODELS
 models=tuple(json.loads(path.read_text())['active_models'])
 if not models or len(set(models))!=len(models) or not set(models)<=set(KNOWN_MODELS):
  raise ValueError('Invalid explicit model scope')
 return models

def job_model(job):
 arguments=job.get('args',[])
 if '--model' in arguments:return arguments[arguments.index('--model')+1]
 explicit=job.get('env',{}).get('ABC_MANIFEST_MODEL')
 if explicit:return explicit
 for model in KNOWN_MODELS:
  if job.get('name','').startswith(model+'_'):return model
 return None

def enabled(job,models=None):
 model=job_model(job)
 return model is None or model in (active_models() if models is None else models)
