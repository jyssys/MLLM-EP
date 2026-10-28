"""One-start C discovery with mixed prompt lengths. This is not accepted gain evidence."""
import json
from .common import ROOT,write

def jobs():
 from .worker import prompts
 from .campaign import job
 result=[];pattern=(128,2048,512,128,2048,128,512,2048)
 for model in ('ouro','huginn','dory'):
  questions=prompts(model,48,0);context=prompts(model,1,4096)[0];rows=[]
  for index,question in enumerate(questions):
   # Preserve the entire question and generation suffix, including the chat template.
   target=pattern[index%len(pattern)]
   if target<len(question):target=256
   tokens=context[:target-len(question)]+question
   assert len(tokens)==target
   rows.append(dict(model=model,sample_id=index,input_ids=tokens,input_tokens=target,
    source='Synthetic padded local training-validation question',answer_labels_used=False))
  manifest=ROOT/'configs'/f'{model}_C_heterogeneous_inputs.jsonl'
  with manifest.open('w') as stream:
   for row in rows:stream.write(json.dumps(row)+'\n')
  opt=['--graphs']+(['--asynchronous'] if model=='ouro' else [])
  boundary=['--enable-boundary-adapter'] if model!='ouro' else []
  for label in ('upstream','queue','token','service','loop','DP4_Mixed','PD1','PD3'):
   extra=opt[:];mode='dp' if label=='DP4_Mixed' else 'pd'
   if mode=='dp':extra+=['--scheduler-mode','mixed','--chunk','256']
   else:
    p=label[2:] if label.startswith('PD') else '2'
    extra+=boundary+['--prefill',p,'--chunk','256']
    if not label.startswith('PD'):extra+=['--wavefront','--placement',label]
   j=job(model,'Chet_'+label,range(4),mode=mode,length=0,count=48,cap=64,concurrency=16,extra=extra)
   j['module']='poc_flashvep.looped_pd_abc.workload_worker'
   j['env'].update(ABC_PROMPT_MANIFEST=str(manifest),ABC_MANIFEST_MODEL=model,ABC_MANIFEST_RUN=j['name'])
   j['quiet']=True;result.append(j)
 write(ROOT/'configs/C_heterogeneous_discovery.json',result)
 return result
