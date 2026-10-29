"""Repeat A tuning without foreground evidence I/O. Preserve the original sweep."""
from .common import ROOT,write
from .scope import active_models

def run():
 from .campaign import job
 from .dispatch import execute
 from .qualify_cuts import qualify
 jobs=[]
 for model,pair,cuts in [('ouro',(0,1),(1,2,3)),('huginn',(2,3),(1,8,16,24,31)),('dory',(4,5),(1,))]:
  if model not in active_models():continue
  opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
  boundary=['--enable-boundary-adapter'] if model!='ouro' else []
  for length in (512,2048):
   for label in ('DP','r0','rR','rR_G1',*cuts):
    mode='dp' if label=='DP' else 'decode_local' if label=='r0' else 'pd'
    extra=opt+['--chunk','2048']+(boundary if mode=='pd' else [])
    if isinstance(label,int):extra+=['--wavefront','--enable-recurrence-split','--split-cut',str(label)]
    if label=='rR_G1':extra+=['--wavefront']
    task=job(model,f'A_clean_sweep_{label}',pair,mode=mode,length=length,count=16,cap=8,concurrency=4,extra=extra)
    task['quiet']=True;jobs.append(task)
 path=ROOT/'configs/clean_A_cut_sweep.json';write(path,jobs);execute(path)
 qualify(sweep_prefix='A_clean_sweep_',qualification_prefix='async_cut',output='clean_chosen_cuts.json')

if __name__=='__main__':run()
