"""Match A's generic descriptor optimization in its full-prefill Ouro control."""
from .common import ROOT,write
from .dispatch import execute

def run():
 from .campaign import job
 jobs=[]
 for repeat in range(3):
  pair=(0,1) if repeat%2==0 else (6,7)
  for length,width,cap,count,label in [(128,1,32,24,'PD_A_coalesced'),(512,4,32,24,'PD_A_coalesced'),
    (2048,8,32,24,'PD_A_coalesced'),(512,4,128,16,'L128_PD_A_coalesced')]:
   jobs.append(job('ouro',label,pair,length=length,count=count,cap=cap,concurrency=width,repeat=repeat,
    extra=['--graphs','--asynchronous','--wavefront','--chunk',str(length),'--enable-boundary-adapter']))
 for j in jobs:j['quiet']=True
 jobs.append(dict(name='ouro_reference_gate_audit',gpus=[6],quiet=True,
  python='/home/esjung/.venvs/looped-pd-abc-hf457/bin/python',
  module='poc_flashvep.looped_pd_abc.ouro_gate_audit',args=['--name','ouro_reference_gate_audit'],
  env={'ABC_DIAGNOSTIC_BURN_OTHERS':'1'}))
 manifest=ROOT/'configs/coalesced_A_controls.json';write(manifest,jobs);execute(manifest)

if __name__=='__main__':run()
