"""Warm actual batch geometries before testing a quieter observer."""
import json
from .common import *
from .prepare import save


def main():
    initialize();old=json.loads((ROOT/'state/native_confirmation_lease_v2_plan.json').read_text());jobs=[]
    template=json.loads((ROOT/'configs'/(old['jobs'][0]+'.json')).read_text())
    for label,observed in (('cleanA',False),('quiet',True),('cleanB',False)):
        c=dict(template,name='native_qwen8_L8000_O16_N128_s4_geometry_'+label+'_v3',
            runtime_observation=observed,runtime_observation_roles=['prefill'],physical_activity=observed,
            activity_quiet_destination=True,warmup_batch_sizes=[1,2,3,4],warmup_output_cap=16,
            phase='QUIET_DESTINATION_ACTIVITY_CONTROL',
            scope='D trace is intentionally censored to fenced empty-engine epochs. Uncaptured D activity is UNKNOWN, not idle. Whole-run and local tail interference still require controls.')
        jobs.append(save(c))
    write(ROOT/'state/quiet_confirmation_plan.json',dict(jobs=jobs,
        scope='Native A/quiet/B with the same deterministic real warm-up geometries. This alternative does not trace Decode kernels. It can establish only measured idle intervals and a lower bound on total inactivity.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()
