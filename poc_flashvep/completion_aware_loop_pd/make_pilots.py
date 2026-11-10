"""Predeclare paired calibration pilots. Primary confirmation stays held out."""
from .calibrate import *


def main():
    initialize();jobs=[]
    for model in ('ouro','huginn'):
        for method in ('native','old','loop','token'):
            name=f'p1_{model}_{method}_clean24'
            cfg=(new_config(model,name,method,count=24) if method in ('loop','token')
                else config(model,name,method,count=24))
            if method in ('loop','token'):cfg['required_qualification']=f'q2r2_{model}_{method}_state32'
            cfg['phase']='PAIRED_N24_SELECTION_NOT_CONFIRMATION'
            save(cfg);jobs.append(name)
        for index in (2,3):
            cfg=config(model,f'aa{index}_{model}_native_clean128')
            cfg['phase']='MATCHED_NATIVE_AA_VARIATION_SAME_ARRIVAL_TRACE'
            save(cfg);jobs.append(cfg['name'])
        cfg=config(model,f'physical_{model}_both_kernel50ms128')
        cfg.update(kernel_activity_seconds=.05,capture_roles=['prefill','decode'],capture_arrival_fraction=.35,
            phase='SHORTER_CUPTI_DIAGNOSTIC_NOT_AUTOMATICALLY_QUALIFIED')
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/pilot_jobs.json',jobs);print(json.dumps(jobs))


if __name__=='__main__':main()
