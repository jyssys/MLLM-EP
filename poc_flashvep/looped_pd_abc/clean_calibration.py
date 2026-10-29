"""Refit every C policy from a separate, nonblocking, isolated calibration."""
from .common import ROOT,write

def run():
 from .campaign import job
 from .dispatch import execute
 jobs=[]
 for model,pair in [('ouro',(0,1)),('huginn',(2,3)),('dory',(4,5))]:
  opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
  boundary=['--enable-boundary-adapter'] if model!='ouro' else []
  for length,width,count in [(n,1,6) for n in (128,512,2048)]+[(n,w,16) for w in (2,4,8) for n in (128,512)]:
   task=job(model,'calibration_async',pair,length=length,concurrency=width,count=count,cap=16,
    extra=opt+boundary+['--wavefront','--chunk','256'])
   task['quiet']=True;task['env']['ABC_ASYNC_EVIDENCE']='1';jobs.append(task)
 path=ROOT/'configs/clean_calibration_jobs.json';write(path,jobs);execute(path)

if __name__=='__main__':run()
