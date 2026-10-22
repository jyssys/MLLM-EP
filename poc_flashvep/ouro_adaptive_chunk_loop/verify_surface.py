"""Check actual prompt/decode work and event-clock bounds before grid rollout."""
import argparse
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,write

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pattern',default='p2_harness_preflight*')
    parser.add_argument('--output',default='surface_work_clock_verification.json')
    a=parser.parse_args()
    reports=[]
    for folder in sorted((ROOT/'raw').glob(a.pattern)):
        if not (folder/'completed.json').exists():
            continue
        for file in folder.glob('*_T*.json'):
            data=json.loads(file.read_text())
            prompt=defaultdict(int);decode=defaultdict(int)
            quanta=data['quantum_observations']
            for q in quanta:
                for span in q['spans']:
                    if span['phase'] in ('prefill','prefilling_recurrent'):
                        prompt[span['id']]+=span['tokens']*span['group']
                    elif span['phase']=='recurrent':
                        decode[span['id']]+=span['tokens']*span['group']
            invariants=[]
            for r in data['requests']:
                invariants.append(dict(id=r['id'],prompt_work=prompt[r['id']],expected_prompt=4*r['input_tokens'],
                    decode_work=decode[r['id']],expected_decode=4*max(0,len(r['output_ids'])-1),
                    depth_R4=all(d==4 for d in r['depths']),token_clock_monotonic=all(b>=a for a,b in zip(r['token_times'],r['token_times'][1:]))))
            passed=all(r['prompt_work']==r['expected_prompt'] and r['decode_work']==r['expected_decode'] and r['depth_R4'] and r['token_clock_monotonic'] for r in invariants)
            reports.append(dict(run=data['run_id'],policy=data['policy'],C=data['configured_C'],passed=passed,
                requests=invariants,quantums=len(quanta),max_alignment_error_bound_ms=max(q['alignment_error_bound_ms'] for q in quanta),
                negative_scheduling_targets=sum(q['scheduling_impact_ms']<0 for q in quanta),
                scope='Recorded software work and event alignment. Not task accuracy or qualified serving capacity.'))
    write(ROOT/a.output,reports)
    print(json.dumps(dict(policy_cohorts=len(reports),passed=sum(r['passed'] for r in reports),
        max_alignment_error_bound_ms=max((r['max_alignment_error_bound_ms'] for r in reports),default=None)),indent=2))
    assert reports and all(r['passed'] for r in reports),'FULL_R4_OR_CLOCK_CHECK_FAILED'

if __name__=='__main__':
    main()
