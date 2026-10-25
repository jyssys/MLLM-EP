"""Preserve the failed Foil metadata attempts and run reviewed retry IDs."""
import json
from .common import ROOT,write
from .final_checks import dispatch

def main():
    original=json.loads((ROOT/'plan_independent_controls.json').read_text())
    jobs=[]
    for job in original['jobs']:
        if job['model']=='foil1':
            failed=ROOT/'raw'/job['id']/'failure.json'
            assert json.loads(failed.read_text())['type']=='TypeError'
            jobs.append(dict(job,id=job['id'].removesuffix('_v1')+'_v2'))
    path=ROOT/'plan_independent_controls_foil_retry_v2.json'
    write(path,dict(phase='independent_control',jobs=jobs))
    dispatch(path,'0,1,2,3,4,5,6,7')
    path=ROOT/'plan_strict_foil_load.json'
    write(path,dict(phase='p0',jobs=[dict(id='foil1_strict_load_final_v1',model='foil1',phase='p0')]))
    dispatch(path,'0')

if __name__=='__main__':
    main()
