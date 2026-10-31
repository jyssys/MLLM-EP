"""Refresh CPU analysis after the managed GPU queue stops. Do not decide research gates."""
import subprocess
from .common import *
from .idle_restore import managed_jobs


def main():
    path=ROOT/'state/after_campaign_analysis.json'
    write(path,dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_GPU_QUEUE'))
    while managed_jobs():time.sleep(15)
    modules=('summary_tables','memory_accounting','classify','additional_analysis','final_tables','profile_summary','figures')
    results=[]
    for module in modules:
        result=subprocess.run([str(PYTHON),'-m','poc_flashvep.looped_pd_bottleneck_crossover.'+module],cwd=REPO)
        results.append(dict(module=module,returncode=result.returncode))
        write(path,dict(time=time.time(),pid=os.getpid(),status='CPU_ANALYSIS_RUNNING',results=results))
        if result.returncode:break
    write(path,dict(time=time.time(),pid=os.getpid(),
        status='ANALYSIS_COMPLETE_REQUIRES_FINAL_REVIEW' if len(results)==len(modules) and all(r['returncode']==0 for r in results) else 'CPU_ANALYSIS_FAILED',
        results=results,scope='No GPU jobs, research verdict, source edits, Git commit, or push are initiated by this helper.'))


if __name__=='__main__':main()
