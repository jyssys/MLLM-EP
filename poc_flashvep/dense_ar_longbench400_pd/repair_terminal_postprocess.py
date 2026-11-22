"""Reconstruct the last frozen repair after its GPU controller drains."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,preserve


def main():
    preserve()
    required=['request_registration_controlled_bridge_complete_v1.json',
        'decode_compile_disabled_control_terminal_v1.json']
    required += [f'Decode_compile_{axis}_gate_v{version}.json'
                 for version in (1,2,3) for axis in ('Layer','Hybrid')]
    last=None
    while not all((ROOT/'state'/name).exists() for name in required):
        progress=ROOT/'state/request_registration_controlled_bridge_progress_v1.json'
        count=len(json.loads(progress.read_text())['results']) if progress.exists() else 0
        if count!=last:
            print('LAST_REPAIR_GPU_PROGRESS',count,54,flush=True)
            last=count
        time.sleep(20)
    jobs=[('registration_controlled_bridge','analyze'),
          ('decode_only_compile_followup','analyze'),
          ('budget_audit',),('independent_metric_audit',),
          ('final_protocol_integrity_audit',),('figure_catalog',),('cpu_final_tests',)]
    results=[]
    for args in jobs:
        command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.'+args[0],*args[1:]]
        result=subprocess.run(command,cwd=REPO)
        results.append(dict(module=args[0],arguments=args[1:],returncode=result.returncode))
        write(ROOT/'state/last_repair_CPU_postprocess_progress_v1.json',dict(time=time.time(),results=results))
        assert result.returncode==0,results[-1]
    preserve()
    write(ROOT/'state/last_repair_CPU_postprocess_complete_v1.json',dict(time=time.time(),
        complete=True,results=results,
        scope='CPU reconstruction only. Strict failures and failed decode gates remain. '
              'Root figure/scientific review,source recheck,freeze,resource closeout,archive and publication remain.'))
    print('LAST_REPAIR_CPU_POSTPROCESS_COMPLETE',flush=True)


if __name__=='__main__':main()
