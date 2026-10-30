"""Qualify the extra Huginn token fraction before any candidate serving data."""
import json
import subprocess
from .common import *
from .gate import compare

def qualify_final_token_grid():
    dest=ROOT/'state/final_token_grid_qualification.json'
    if dest.exists():
        return json.loads(dest.read_text()).get('fixed',{}).get('passed',False)
    specifications=[('gate8_huginn_token_static_v4.json','fair_grid_huginn_token_static_eighths',dict(prefix_fraction=.875)),
                    ('qualification32_huginn_token_dynamic_v1.json','qualification32_huginn_token_dynamic_v2',{})]
    jobs=[]
    for source,name,overrides in specifications:
        cfg=json.loads((ROOT/'configs'/source).read_text());cfg.update(overrides,name=name,gpus=[4,5])
        path=ROOT/'configs'/f'{name}.json'
        if not path.exists():write(path,cfg)
        jobs.append(path.name)
    result=subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.dispatch',*jobs],cwd=REPO)
    try:fixed=compare('gate8_huginn_pd_g1_v1','fair_grid_huginn_token_static_eighths')
    except Exception as error:fixed=dict(passed=False,error=repr(error))
    try:dynamic=compare('qualification32_huginn_pd_g1_v1','qualification32_huginn_token_dynamic_v2')
    except Exception as error:dynamic=dict(passed=False,error=repr(error))
    value=dict(passed=fixed['passed'] and dynamic['passed'],fixed=fixed,dynamic=dynamic,
               reason='Huginn Token candidates include the same28/32 fraction as Loop candidates before legal alignment.')
    write(dest,value)
    write(ROOT/'state/gate_qualification32_huginn_token_dynamic_v1.json',dict(dynamic,
          supersedes='qualification32_huginn_token_dynamic_v1',actual_final_candidate='qualification32_huginn_token_dynamic_v2'))
    if not fixed['passed']:
        path=ROOT/'state/gate_qualification32_huginn_token_static_v1.json'
        previous=json.loads(path.read_text())
        write(path,dict(passed=False,reason='EXTRA_TOKEN_FRACTION_FAILED',previous_fixture=previous,extra_fraction=fixed))
    return fixed['passed']

if __name__=='__main__':qualify_final_token_grid()
