"""Extend trace coverage to every primary C. Do not change clean plans."""
from poc_depth_cdb_factorial.contract import *

def main():
    cases=read(ROOT/'artifacts/speed_plan.json')['cases']
    for dp in [False,True]:
        selected=[dict(c,name='trace_'+c['name']) for c in cases if c['repeat']==0 and (c['arm']!='A0')==dp]
        destination=ROOT/f'artifacts/trace_{"DP4" if dp else "A0"}_full_plan.json'
        assert not destination.exists()
        write(destination,dict(cases=selected,local_batch_cap=16,
            note='All primary C values. Trace-only extension before trace measurements. Original trace plan remains preserved.'))
    historical=read(V3_ROOT/'artifacts/hindsight_confirmation_c64_frozen.json')
    old={c['arm']:c for c in historical['cases'] if c['repeat']==0}
    extra=[]
    for repeat in range(3):
        for arm in (['A1','A2','A3','A4'] if repeat%2==0 else ['A4','A3','A2','A1']):
            template=old['hindsight' if ARMS[arm]['hindsight'] else 'generic']
            extra.append(dict(name=f'sensitivity_v3matched_C64_{arm}_r{repeat}',kind='sensitivity',arm=arm,
                C=64,repeat=repeat,ids=template['ids'],cap=256,placements=template['placements']))
    write(ROOT/'artifacts/sensitivity_v3_matched_DP4_plan.json',dict(cases=extra,local_batch_cap=16,
        note='Both historical control and hindsight placements. This permits a matched v3 P95 comparison.',
        historical_manifest_sha256=sha(V3_ROOT/'artifacts/hindsight_confirmation_c64_frozen.json')))

if __name__=='__main__':main()
