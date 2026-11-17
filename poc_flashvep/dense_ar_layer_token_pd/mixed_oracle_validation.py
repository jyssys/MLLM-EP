"""Check one constrained static and one future-selected diagnostic plan."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,resources,write
from .flow_campaign import static_config
from .discovery import job


def main():
    path=ROOT/'analysis/strong_per_cap_fraction_oracle_v2_Joint95.json'
    while not path.exists():
        time.sleep(10)
    model=json.loads(path.read_text())
    selected=next(v for v in json.loads((ROOT/'analysis/flow_macro8_oracle_screen_v3.json').read_text()) if v['rate']==4.8)
    values=[]
    try:
        plans=[('flow_low_Static_Joint95_N128_v1',model['static']['best_with_Joint95']['plan'],False),
            ('flow_low_Offline_future_selected_plan_N128_diagnostic_v1',
                model['best_tested_dynamic_with_Joint95']['plan'],True)]
        for name,plan,offline in plans:
            cfg=static_config(name,selected,plan,
                phase='OFFLINE_PLAN_MODEL_VALIDATION' if offline else 'STRONG_STATIC_JOINT95_DISCOVERY',
                offline_selected_schedule=offline,evaluation_arm='OfflinePlanDiagnostic' if offline else 'MixedStaticValidation')
            value=job(cfg)
            if value:
                values.append(value);write(ROOT/'analysis/mixed_constrained_static_oracle_GPU_validation.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
