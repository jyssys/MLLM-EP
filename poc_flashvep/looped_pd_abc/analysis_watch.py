"""Run CPU analysis after the isolated GPU campaign releases its resource pool."""
import json
import os
import subprocess
import time
import traceback
from .common import ROOT,REPO,PYTHON,write

def main():
 target=ROOT/'state/main_campaign.json';dest=ROOT/'state/analysis_watch.json'
 write(dest,dict(status='WAITING_FOR_GPU_CAMPAIGN',pid=os.getpid(),time=time.time()))
 try:
  while True:
   state=json.loads(target.read_text()) if target.exists() else {}
   if state.get('status')=='COMPLETE_REVIEW_REQUIRED':break
   controller=state.get('pid')
   if controller and not __import__('pathlib').Path(f'/proc/{controller}').exists():
    raise RuntimeError('GPU controller stopped before completion. Preserve logs and resume its completed-run-aware dispatcher.')
   time.sleep(30)
  write(dest,dict(status='ANALYZING_COMPLETED_GPU_RUNS',pid=os.getpid(),time=time.time()))
  with (ROOT/'logs/final_cpu_analysis.log').open('a') as log:
   for module,args in [('compare',['--comparisons']),('profile_analysis',[]),('render',[])]:
    subprocess.run([PYTHON,'-m','poc_flashvep.looped_pd_abc.'+module,*args],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
  write(dest,dict(status='ANALYSIS_COMPLETE_REVIEW_REQUIRED',pid=os.getpid(),time=time.time(),
   warning='Generated final-file names remain explicit drafts until scientific review. No commit or push occurs automatically.'))
 except BaseException as error:
  write(dest,dict(status='FAILED',pid=os.getpid(),time=time.time(),error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
