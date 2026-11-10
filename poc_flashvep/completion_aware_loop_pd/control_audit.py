"""Compare later zero-offload controls without calling them policy offloading."""
from .common import *
from .serving_audit import inspect,paired,csv_write


def main():
    configure();rows=[]
    for name in json.loads((ROOT/'state/zero_confirmation_jobs.json').read_text()):
        r,_,_=inspect(name);r['secondary_arm']='zero';rows.append(r)
    recovery=json.loads((ROOT/'recovery_serving_results.json').read_text())
    native=[r for r in recovery if r['cohort']=='N128_CONFIRMATION' and r['secondary_arm']=='native']
    comparisons=paired(rows+native,arms=('zero',),references=('native',))
    treatments=[r for r in recovery if r['cohort']=='N128_CONFIRMATION' and r['secondary_arm'] in ('loop','token')]
    matched=paired(rows+treatments,arms=('loop','token'),references=('zero',))
    write(ROOT/'zero_control_results.json',rows);csv_write(ROOT/'zero_control_results.csv',rows)
    write(ROOT/'zero_control_ratios.json',comparisons);csv_write(ROOT/'zero_control_ratios.csv',comparisons)
    write(ROOT/'matched_native_path_ratios.json',matched);csv_write(ROOT/'matched_native_path_ratios.csv',matched)
    print(json.dumps(dict(starts=len(rows),complete=sum(r.get('complete',False) for r in rows))))


if __name__=='__main__':
    main()
