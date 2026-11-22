"""Wait for the final bounded GPU repair, then reconstruct its evidence."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,preserve


def main():
    preserve();last=None
    target=ROOT/'state/eager_compile_confirmation_v2_complete_v1.json'
    while not target.exists():
        path=ROOT/'state/eager_compile_confirmation_v2_progress_v1.json'
        count=len(json.loads(path.read_text())['results']) if path.exists() else 0
        if count!=last:
            print('FINAL_COMPILER_GPU_PROGRESS',count,21,flush=True);last=count
        time.sleep(20)
    jobs=[('eager_prefill_decode_campaign','analyze'),('final_compiler_artifacts','--figures'),
          ('budget_audit',),('independent_metric_audit',),('final_protocol_integrity_audit',),
          ('figure_catalog',),('cpu_final_tests',),('final_environment_recheck',)]
    rows=[]
    for args in jobs:
        result=subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.'+args[0],*args[1:]],cwd=REPO)
        rows.append(dict(module=args[0],arguments=args[1:],returncode=result.returncode))
        write(ROOT/'state/final_compiler_CPU_postprocess_progress_v1.json',dict(time=time.time(),results=rows))
        assert result.returncode==0,rows[-1]
    preserve()
    write(ROOT/'state/final_compiler_CPU_postprocess_complete_v1.json',dict(time=time.time(),complete=True,results=rows,
        scope='CPU reconstruction, not a scientific GO. Root review, final report hashes, resource closeout, '
              'lossless archive, commit, push and remote verification still remain.'))
    print('FINAL_COMPILER_CPU_RECONSTRUCTION_COMPLETE',flush=True)


if __name__=='__main__':main()
