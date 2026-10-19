"""Continue after the already-running campaign, then analyze complete evidence."""
import os
import subprocess
import time
from poc_depth_cdb_factorial.contract import *

def main():
    write(ROOT/'artifacts/continuation_controller.json',dict(pid=os.getpid(),started_host_s=time.perf_counter(),
        purpose='Wait for required quality/trace/sensitivity, run two extra complete repeats, then analyze.',
        user_stop_rule='Stop this controller and its campaign parents before stopping GPU workers if the user requests a pause.'))
    required=['quality_A0_v1','quality_DP4_v1','trace_A0_v1','trace_DP4_v1','sensitivity_DP4_v1']
    deadline=time.monotonic()+18000
    while True:
        complete=[]
        for name in required:
            path=ROOT/'outputs'/name/'guard_status.json'
            if path.exists():
                assert read(path)['success'],name
                complete.append(name)
        if len(complete)==len(required):break
        assert time.monotonic()<deadline,'The prerequisite campaign exceeded the controller deadline.'
        time.sleep(5)
    time.sleep(5)
    for module,extra in [('campaign',['--phase','extra']),('analyze',['--quality']),
        ('trace_analysis',[]),('equal_progress',[]),('audit_v3',[]),('functional_summary',[]),('figures',[])]:
        print('POST_PHASE',module,flush=True)
        subprocess.run([REFERENCE_PYTHON,'-m','poc_depth_cdb_factorial.scripts.'+module,*extra],cwd=PROJECT,check=True)
    write(ROOT/'artifacts/computation_complete.json',dict(success=True,
        primary_cohorts=125,quality_generations=1280,all_required_rows_measured=True,reports_pending_review=True))
    print('COMPUTATION_COMPLETE',flush=True)

if __name__=='__main__':main()
