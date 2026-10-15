"""Run bounded native diagnostics sequentially. Each GPU guard restores burners."""
import subprocess
import sys
from poc_looped_multigpu_kv_placement.contract import *


def main():
    base='results/raw/'
    jobs=[
        ('ouro_tp4_kernel_profile_v1','run_native_layout_sweep',['--tp','4','--profile-only','--trace-prefix','16384']),
        ('ouro_tp4_eager_source_v2','run_native_layout_sweep',['--tp','4','--profile-only','--eager','--trace-prefix','16384']),
        ('ouro_tp4_outer_events_v1','run_native_layout_sweep',['--tp','4','--events-only','--trace-prefix','16384','--limit','8']),
        ('ouro_tp4_natural_quality_v1','run_native_layout_sweep',['--tp','4','--quality-only']),
        ('ouro_replicas_tp1_dp4_v1','run_native_replicas',['--tp','1']),
        ('ouro_replicas_tp2_dp2_v1','run_native_replicas',['--tp','2']),
    ]
    records=[]
    for name,module,flags in jobs:
        relative=base+name;output=ROOT/relative
        if output.exists():
            records.append(dict(name=name,status='EXISTING_EVIDENCE_PRESERVED'))
            continue
        command=[sys.executable,'-m','poc_looped_multigpu_kv_placement.scripts.gpu_session',
            '--gpus',MASK,'--timeout','2400','--output',relative,'--',VLLM_PYTHON,
            '-m','poc_looped_multigpu_kv_placement.scripts.'+module,'--output',str(output),*flags]
        result=subprocess.run(command,cwd=PROJECT,check=False)
        records.append(dict(name=name,exit_code=result.returncode,command=command))
        write(ROOT/'results/remaining_checks.json',records)
        print(name,result.returncode,flush=True)


if __name__=='__main__':main()
