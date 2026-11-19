"""Repeat the existing C1 control without substituting it for online isolation."""
import itertools
import json
import time
import numpy as np
from .common import ROOT,write,save
from .qualification import configuration,records
from .run import execute
from .metrics import summarize


def main():
    design=ROOT/'configs/frozen_Native_clean_C1_confirmation_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),input_tokens=8192,output_tokens=16,count=8,
            independent_starts_per_arm=3,reuse_existing_start1=True,numerical_family='OFFICIAL_BATCH_INVARIANT',
            native_protocol='OFFICIAL_PROXY',threshold_upper_CI95=1.05,metrics=['ITL_p99_ms','TPOT_p95_ms'],
            scope='Same clean serial diagnostic as the existing control. No concurrent imports into active decode. No online isolation or sustainable-capacity claim.'))
    while not (ROOT/'state/UCX_protocol_probe_complete.json').exists():
        for file in ('STOP_REQUESTED.json','Main_driver_hold.json','followup_driver_hold.json','UCX_protocol_probe_hold.json'):
            assert not (ROOT/'state'/file).exists(),file
        time.sleep(5)
    teacher={r['prompt_id']:r['output_ids'] for r in records(ROOT/'raw/micro_Standard_L8192_O16_N8_C1_BI1_B8192_S1024_v1/requests.jsonl')}
    names={label:[f'micro_{label}_L8192_O16_N8_C1_BI1_B8192_S1024_v1'] for label in ('Standard','OfficialPD')}
    for repeat in (2,3):
        for label in ('Standard','OfficialPD') if repeat==2 else ('OfficialPD','Standard'):
            name=f'micro_{label}_L8192_O16_N8_C1_BI1_B8192_S1024_r{repeat}_v1'
            extra=dict(reference_single=True,gpus=[1]) if label=='Standard' else dict(nixl_manual_coalesce=False)
            cfg=configuration(name,'Native','diagnostic_serving_L8192.json',batch_invariant=True,
                output_cap=16,ignore_eos=True,state_capture=False,correctness_only=False,diagnostic_only=True,**extra)
            assert execute(save(cfg)),name
            actual=records(ROOT/'raw'/name/'requests.jsonl')
            assert len(actual)==8 and all(r['output_ids']==teacher[r['prompt_id']] for r in actual)
            lifetime=[json.loads(p.read_text())['passed'] for p in (ROOT/'raw'/name).glob('KV_lifetime_*.json')]
            assert len(lifetime)==(1 if label=='Standard' else 2) and all(lifetime)
            summarize(name);names[label].append(name)
    rows={label:[summarize(name) for name in part] for label,part in names.items()};results={}
    for key in ('ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms'):
        a=[r[key] for r in rows['OfficialPD']];b=[r[key] for r in rows['Standard']]
        draws=[np.mean([a[i] for i in x])/np.mean([b[i] for i in y]) for x in itertools.product(range(3),repeat=3) for y in itertools.product(range(3),repeat=3)]
        ci=[float(np.percentile(draws,p)) for p in (2.5,97.5)]
        results[key]=dict(Native_over_Standard=float(np.mean(a)/np.mean(b)),CI95=ci,
            noninferiority_upper_bound_passed=ci[1]<=1.05)
    passed=all(results[k]['noninferiority_upper_bound_passed'] for k in ('ITL_p99_ms','TPOT_p95_ms'))
    write(ROOT/'analysis/Native_clean_C1_confirmation_v1.json',dict(names=names,results=results,
        independent_starts_per_arm=3,bootstrap_combinations=729,passed=passed,
        evidence='MEASURED_DIAGNOSTIC',scope='Unpaired empirical start-level bootstrap. Clean C1 only. It does not establish decode protection during concurrent NIXL imports or on Mixed200. Single-GPU throughput is not compared with two-GPU throughput.'))
    write(ROOT/'state/Native_clean_C1_confirmation_complete.json',dict(time=time.time(),complete=True,passed=passed))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/Native_clean_C1_confirmation_hold.json',dict(time=time.time(),error=repr(error)))
        raise
