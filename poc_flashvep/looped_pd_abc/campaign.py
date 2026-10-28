"""Create deterministic experiment manifests. This file does not generate results."""
import argparse
import json
from .common import ROOT,write

def selected_cut(model,length):
 """Use the frozen, qualified cut. Never read live future outputs."""
 path=ROOT/'configs/chosen_cuts.json';probe_length=512 if length<512 else length
 if path.exists():
  return json.loads(path.read_text())[model][str(probe_length)]['cut']
 return {'ouro':2,'huginn':16,'dory':1}[model]

def job(model,label,gpus,*,mode='pd',length=128,count=8,cap=32,concurrency=1,extra=(),repeat=0):
 name=f'{model}_{label}_P{length}_C{concurrency}_s{repeat}'
 args=['--model',model,'--mode',mode,'--gpus',','.join(map(str,gpus)),'--name',name,
  '--requests',str(count),'--length',str(length),'--cap',str(cap),'--concurrency',str(concurrency),'--warmup',*extra]
 env={'ABC_RLT_DORY':'1'} if model=='dory' else {}
 return dict(name=name,gpus=list(gpus),args=args,env=env)

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('parity','serving','smoke','scale','calibrate','C','A_sweep'));a=p.parse_args();jobs=[]
 if a.phase=='smoke':
  for model,gpus,cut in [('huginn',(0,1),16),('dory',(2,3),1)]:
   jobs.append(job(model,'A_repaired',gpus,cap=8,count=4,extra=['--wavefront','--chunk','256','--enable-recurrence-split','--split-cut',str(cut)]))
  jobs.append(dict(name='dory_official_compat4',gpus=[6],module='poc_flashvep.looped_pd_abc.dory_reference',env={'ABC_DORY_PLUGIN':'1'},args=['--name','dory_official_compat4','--requests','4','--cap','32']))
 elif a.phase=='parity':
  for model,pair,cut in [('ouro',(0,1),2),('huginn',(2,3),16),('dory',(4,5),1)]:
   opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
   boundary=['--enable-boundary-adapter'] if model!='ouro' else []
   jobs.append(job(model,'parity_native',(pair[0],),mode='native',length=0,count=100,cap=32,extra=opt))
   jobs.append(job(model,'parity_PD',pair,length=0,count=100,cap=32,extra=opt+boundary))
   jobs.append(job(model,'parity_A',pair,length=0,count=100,cap=32,extra=opt+['--wavefront','--enable-recurrence-split','--split-cut',str(cut)]))
   jobs.append(job(model,'parity_B',pair,length=0,count=100,cap=32,extra=opt+boundary+['--wavefront','--chunk','256','--enable-frontier-handoff']))
   jobs.append(job(model,'parity_B_control',pair,length=0,count=100,cap=32,extra=opt+boundary+['--wavefront','--chunk','256']))
 elif a.phase=='calibrate':
  for model,pair in [('ouro',(0,1)),('huginn',(2,3)),('dory',(4,5))]:
   opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
   boundary=['--enable-boundary-adapter'] if model!='ouro' else []
   for n in (128,512,2048):
    jobs.append(job(model,'calibration',pair,length=n,count=6,cap=16,extra=opt+boundary+['--wavefront','--chunk','256']))
   for width in (2,4,8):
    for n in (128,512):
     jobs.append(job(model,'calibration',pair,length=n,count=16,cap=16,concurrency=width,extra=opt+boundary+['--wavefront','--chunk','256']))
 elif a.phase=='C':
  for repeat in range(3):
   for model in ('ouro','huginn','dory'):
    opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
    boundary=['--enable-boundary-adapter'] if model!='ouro' else []
    group=range(4) if repeat%2==0 else range(4,8)
    for policy in ('upstream','queue','token','service','loop'):
     jobs.append(job(model,'C_'+policy,group,length=0,count=48,cap=64,concurrency=8,
      extra=opt+boundary+['--prefill','2','--wavefront','--chunk','256','--placement',policy],repeat=repeat))
    jobs.append(job(model,'C_DP4',group,mode='dp',length=0,count=48,cap=64,concurrency=8,extra=opt,repeat=repeat))
    jobs.append(job(model,'C_DP4_Mixed',group,mode='dp',length=0,count=48,cap=64,concurrency=8,extra=opt+['--scheduler-mode','mixed','--chunk','256'],repeat=repeat))
    for pcount in (1,3):
     jobs.append(job(model,f'C_PD{pcount}',group,length=0,count=48,cap=64,concurrency=8,extra=opt+boundary+['--prefill',str(pcount),'--chunk','256'],repeat=repeat))
 elif a.phase=='A_sweep':
  for model,pair,cuts in [('ouro',(0,1),(1,2,3)),('huginn',(2,3),(1,8,16,24,31)),('dory',(4,5),(1,))]:
   opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
   boundary=['--enable-boundary-adapter'] if model!='ouro' else []
   for P in (512,2048):
    for label in ('DP','r0','rR','rR_G1',*cuts):
     mode='dp' if label=='DP' else 'decode_local' if label=='r0' else 'pd'
     extra=opt+(['--chunk','2048'])
     if mode=='pd':extra+=boundary
     if isinstance(label,int):extra+=['--wavefront','--enable-recurrence-split','--split-cut',str(label)]
     if label=='rR_G1':extra+=['--wavefront']
     jobs.append(job(model,f'A_sweep_{label}',pair,mode=mode,length=P,count=16,cap=8,concurrency=4,extra=extra))
 elif a.phase=='serving':
  from .qualify_cuts import qualify
  qualify()
  jobs.append(dict(name='nixl_two_process_pair01',gpus=[0,1],module='poc_flashvep.looped_pd_abc.nixl_ipc_probe',
   args=['--name','nixl_two_process_pair01'],env={}))
  # Generic descriptor coalescing must not be attributed to loop-frontier timing.
  jobs.append(job('ouro','coalesced_qualification',(0,1),length=0,count=100,cap=32,
   extra=['--graphs','--asynchronous','--wavefront','--chunk','256','--enable-boundary-adapter']))
  # Each process starts independently. The repeated order alternates controls and methods.
  for repeat in range(3):
   for model in ('ouro','huginn','dory'):
    opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
    boundary=['--enable-boundary-adapter'] if model!='ouro' else []
    methods=['DP','DP_Mixed','PD','PD_G1','PD_A_control','A','B'] if repeat%2==0 else ['B','A','PD_A_control','PD_G1','PD','DP_Mixed','DP']
    for length,concurrency in [(128,1),(512,4),(2048,8)]:
     for label in methods:
      pair=(0,1) if repeat%2==0 else (6,7)
      mode='dp' if label in ('DP','DP_Mixed') else 'pd';extra=opt[:]
      if mode=='pd':extra+=boundary
      if label=='A':extra+=['--wavefront','--chunk',str(length),'--enable-recurrence-split','--split-cut',str(selected_cut(model,length))]
      elif label=='PD_A_control':extra+=['--wavefront','--chunk',str(length)]
      elif label=='B':extra+=['--wavefront','--chunk','256','--enable-frontier-handoff']
      elif label=='PD_G1':
       extra+=['--wavefront','--chunk','256']
       if model=='ouro':extra+=['--enable-boundary-adapter']
      elif label=='DP_Mixed':extra+=['--scheduler-mode','mixed','--chunk','256']
      else:extra+=['--chunk','256']
      jobs.append(job(model,label,pair,mode=mode,length=length,count=24,cap=32,concurrency=concurrency,extra=extra,repeat=repeat))
  # A longer natural-output regime prevents a cap-32-only PD conclusion.
  for repeat in range(3):
   for model in ('ouro','huginn','dory'):
    pair=(0,1) if repeat%2==0 else (6,7)
    opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
    boundary=['--enable-boundary-adapter'] if model!='ouro' else []
    methods=['DP','DP_Mixed','PD','PD_G1','PD_A_control','A','B']
    if repeat%2:methods.reverse()
    for label in methods:
     mode='dp' if label in ('DP','DP_Mixed') else 'pd';extra=opt[:]
     if mode=='pd':extra+=boundary
     if label=='A':extra+=['--wavefront','--chunk','512','--enable-recurrence-split','--split-cut',str(selected_cut(model,512))]
     elif label=='B':extra+=['--wavefront','--chunk','256','--enable-frontier-handoff']
     elif label=='PD_A_control':extra+=['--wavefront','--chunk','512']
     elif label=='PD_G1':
      extra+=['--wavefront','--chunk','256']
      if model=='ouro':extra+=['--enable-boundary-adapter']
     elif label=='DP_Mixed':extra+=['--scheduler-mode','mixed','--chunk','256']
     else:extra+=['--chunk','256']
     jobs.append(job(model,'L128_'+label,pair,mode=mode,length=512,count=16,cap=128,concurrency=4,extra=extra,repeat=repeat))
 elif a.phase=='scale':
  for repeat in range(3):
   for n in (2,4,8):
    group=(list(range(n)) if repeat%2==0 else list(range(6,8)) if n==2 else list(range(4,8)) if n==4 else [(g+repeat)%8 for g in range(8)])
    for model in ('ouro','huginn','dory'):
     opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
     boundary=['--enable-boundary-adapter'] if model!='ouro' else []
     counts=(0,1) if n==2 else (0,1,2,3) if n==4 else (0,1,2,4)
     for prefill in counts:
      label='DP' if not prefill else 'PD'+str(prefill)
      extra=opt+['--chunk','256']
      if prefill:extra+=boundary+['--prefill',str(prefill)]
      jobs.append(job(model,f'scale{n}_{label}',group,mode='pd' if prefill else 'dp',length=512,count=32,cap=32,concurrency=8,extra=extra,repeat=repeat))
 if a.phase in ('serving','scale','C','A_sweep'):
  for j in jobs:j['quiet']=True
 dest=ROOT/'configs'/f'{a.phase}_campaign.json';write(dest,jobs);print(dest,len(jobs))

if __name__=='__main__':main()
