"""Qualify a compiled full-model baseline without compiling partial boundaries."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics


def main():
    jobs=[]
    for count in (8,32):
        jobs.append(base(f'compiled_opaque_norm_Continuous_B512_O16_N{count}_v1',arm='Colocated',count=count,cap=16,
            token_budget=512,chunk=512,opaque_phase_norm=True,ar_compile_mode=3,
            ar_execution_family='COMPILED_FULL_MODEL_OPAQUE_PHASE_NORMALIZATION',startup_timeout=900,wall_timeout=1600,
            phase='COMPILED_FULL_MODEL_CORRECTNESS_NOT_PARTIAL_LAYER_COMPILATION'))
    jobs.append(base('compiled_opaque_norm_Native_O16_N64_v1',cap=16,opaque_phase_norm=True,ar_compile_mode=3,
        ar_execution_family='COMPILED_FULL_MODEL_OPAQUE_PHASE_NORMALIZATION',startup_timeout=900,wall_timeout=1600,
        phase='COMPILED_NATIVE_GENERIC_RUNTIME_CONTROL'))
    jobs.append(base('compiled_opaque_norm_Continuous_B512_O16_N64_v1',arm='Colocated',cap=16,
        token_budget=512,chunk=512,opaque_phase_norm=True,ar_compile_mode=3,
        ar_execution_family='COMPILED_FULL_MODEL_OPAQUE_PHASE_NORMALIZATION',startup_timeout=900,wall_timeout=1600,
        phase='COMPILED_CONTINUOUS_GENERIC_RUNTIME_CONTROL'))
    results=[]
    try:
        for cfg in jobs:
            if not execute(save(cfg),restore=False):
                write(ROOT/'state/compiled_baseline_progress.json',dict(results=results,failed=cfg['name']));return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',cfg['name'],True)
            row=dict(name=cfg['name'],output=output,metrics=metrics(cfg['name']))
            results.append(row);write(ROOT/'state/compiled_baseline_progress.json',dict(results=results))
            print('COMPILED_FULL_MODEL',cfg['name'],output['passed'],flush=True)
            if not output['passed']:return
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
