"""Run a bounded C shadow diagnostic before clean C comparisons."""
from .common import ROOT,write
from .scope import active_models

def run():
 from .campaign import job
 from .dispatch import execute
 jobs=[]
 for model in active_models():
  opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
  boundary=['--enable-boundary-adapter'] if model!='ouro' else []
  j=job(model,'C_shadow',range(4),length=0,count=24,cap=16,concurrency=16,
   extra=opt+boundary+['--prefill','2','--wavefront','--chunk','256','--placement','loop'])
  j['module']='poc_flashvep.looped_pd_abc.shadow_worker';j['quiet']=True
  j['env'].update(ABC_PROMPT_MANIFEST=str(ROOT/'configs'/f'{model}_C_heterogeneous_inputs.jsonl'),
   ABC_MANIFEST_MODEL=model,ABC_MANIFEST_RUN=j['name'])
  jobs.append(j)
 manifest=ROOT/'configs/C_shadow_diagnostics.json';write(manifest,jobs);execute(manifest)
