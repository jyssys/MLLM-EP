"""Resume only unfinished qualification work after the current dispatcher drains."""
from .common import ROOT,write,environment
from .campaign import job
from .dispatch import execute
from .analyze import summarize,calibrate

def plan():
 jobs=[]
 for model,pair,cut in [('huginn',(2,3),16),('dory',(4,5),1)]:
  opt=['--graphs'];boundary=['--enable-boundary-adapter']
  for label in ('native','PD','A','B','B_control'):
   extra=opt[:];mode='native' if label=='native' else 'pd'
   if label=='PD':extra+=boundary
   if label=='A':extra+=['--wavefront','--enable-recurrence-split','--split-cut',str(cut)]
   if label in ('B','B_control'):
    extra+=boundary+['--wavefront','--chunk','256']
    if label=='B':extra+=['--enable-frontier-handoff']
   jobs.append(job(model,'parity2_'+label,(pair[0],) if mode=='native' else pair,mode=mode,length=0,count=100,cap=32,extra=extra))
 for model in ('ouro','huginn'):
  name=model+'_HF_reference'
  jobs.append(dict(name=name,gpus=[7],module='poc_flashvep.looped_pd_abc.reference_probe',
   args=['--model',model,'--name',name,'--requests','4','--length','128','--cap','8']))
 jobs.append(job('huginn','native_HF_rng_control',(7,),mode='native',cap=8,count=4))
 jobs.append(dict(name='dory_official_B1_100',gpus=[6],module='poc_flashvep.looped_pd_abc.dory_reference',
  env={'ABC_DORY_PLUGIN':'1'},args=['--name','dory_official_B1_100','--requests','100','--cap','32','--batch-size','1']))
 for model,pair in [('huginn',(2,3)),('dory',(4,5))]:
  for n,width in [(128,1),(512,1),(2048,1),(128,2),(512,2),(128,4),(512,4),(128,8),(512,8)]:
   jobs.append(job(model,'calibration2',pair,length=n,count=6 if width==1 else 16,cap=16,concurrency=width,
    extra=['--graphs','--enable-boundary-adapter','--wavefront','--chunk','256']))
 manifest=ROOT/'configs/remaining_qualification.json';write(manifest,jobs)
 return manifest

def main():
 execute(plan())
 calibrate();summarize()
 write(ROOT/'state/qualification_review.json',dict(status='COMPLETE_REVIEW_REQUIRED',note='No speedup or paper accuracy acceptance follows automatically.'))

if __name__=='__main__':main()
