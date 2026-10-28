"""One bounded official-reference probe on an otherwise unused GPU."""
import argparse
import os
import subprocess
from .common import ROOT,REPO,environment,record_command
from . import resources

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--gpu',type=int,required=True)
 p.add_argument('--requests',type=int,default=4);p.add_argument('--length',type=int,default=128);p.add_argument('--cap',type=int,default=8);a=p.parse_args()
 name=a.model+('_HF_4576_reference' if a.requests==4 else '_HF_4576_natural100');g=[a.gpu]
 resources.base.stop(g)
 try:
  command=['/home/esjung/.venvs/looped-pd-abc-hf457/bin/python','-m','poc_flashvep.looped_pd_abc.reference_probe',
   '--model',a.model,'--name',name,'--requests',str(a.requests),'--length',str(a.length),'--cap',str(a.cap)]
  with (ROOT/'logs'/(name+'.log')).open('a') as f:
   child=subprocess.Popen(command,cwd=REPO,env=environment(g),stdout=f,stderr=subprocess.STDOUT)
   record_command(name,command,g,child.pid);result=child.wait()
  print(name,result,flush=True)
 finally:resources.base.restore(g)

if __name__=='__main__':main()
