"""Qualify the final Dory adapter after generic GEMM/RoPE fidelity corrections."""
from .common import ROOT,write
from .campaign import job
from .dispatch import execute
from .analyze import summarize,calibrate

def main():
 jobs=[]
 for label in ('native','PD','A','B','B_control'):
  mode='native' if label=='native' else 'pd';extra=['--graphs']
  if label=='PD':extra+=['--enable-boundary-adapter']
  if label=='A':extra+=['--wavefront','--enable-recurrence-split','--split-cut','1']
  if label in ('B','B_control'):
   extra+=['--enable-boundary-adapter','--wavefront','--chunk','256']
   if label=='B':extra+=['--enable-frontier-handoff']
  jobs.append(job('dory','parity3_'+label,(4,) if mode=='native' else (4,5),mode=mode,length=0,count=100,cap=32,extra=extra))
 for n,width in [(128,1),(512,1),(2048,1),(128,2),(512,2),(128,4),(512,4),(128,8),(512,8)]:
  jobs.append(job('dory','calibration2',(4,5),length=n,count=6 if width==1 else 16,cap=16,concurrency=width,
   extra=['--graphs','--enable-boundary-adapter','--wavefront','--chunk','256']))
 manifest=ROOT/'configs/dory_final_qualification.json';write(manifest,jobs)
 execute(manifest);calibrate();summarize()
 write(ROOT/'state/post_qualification_review.json',dict(status='COMPLETE_REVIEW_REQUIRED'))

if __name__=='__main__':main()
