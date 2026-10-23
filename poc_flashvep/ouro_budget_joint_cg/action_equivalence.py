"""Compare actual legal per-request plans, not just optimizer utilities."""
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .cost_predictor import csv_file

def plan(data):
    rows=defaultdict(list)
    for decision in data['chunk_decisions']:
        s=decision['state']
        rows[s['id']].append((decision['phase'],s['start'],s['depth'],decision['chosen_C'],decision['chosen_G']))
    return {key:sorted(value) for key,value in rows.items()}

def main():
    rows=[];cohorts=[]
    for folder in sorted((ROOT/'raw').glob('cb_p5_confirm_*')):
        if not (folder/'completed.json').exists():
            continue
        jobs=json.loads((folder/'identity.json').read_text())['job']
        for trial in jobs['trials']:
            a=json.loads((folder/f'G1_T{trial}.json').read_text());b=json.loads((folder/f'J1_T{trial}.json').read_text())
            left,right=plan(a),plan(b)
            for request in b['requests']:
                rid=request['id']
                rows.append(dict(run_id=folder.name,P=b['P'],budget_ms=b['configured_budget_ms'],trial=trial,id=rid,
                    exact_request_chunk_group_plan=left.get(rid)==right.get(rid),
                    G1_decisions=len(left.get(rid,[])),J1_decisions=len(right.get(rid,[])),
                    scope='Actual immutable new-chunk and continuation descriptors. Host timing excluded.'))
            def times(data):
                return [r['selector_ns']/1000 for r in data['chunk_decisions']]
            ta,tb=times(a),times(b)
            cohorts.append(dict(run_id=folder.name,P=b['P'],budget_ms=b['configured_budget_ms'],trial=trial,
                G1_selector_total_ms=sum(ta)/1000,J1_selector_total_ms=sum(tb)/1000,
                G1_selector_P50_us=float(np.percentile(ta,50)),J1_selector_P50_us=float(np.percentile(tb,50)),
                G1_admission_total_ms=a['admission_cpu']['total_ns']/1e6,J1_admission_total_ms=b['admission_cpu']['total_ns']/1e6,
                scope='Measured CPU call durations, not additive GPU critical-path attribution.'))
    csv_file(ROOT/'G1_J1_actual_plan_equivalence.csv',rows);csv_file(ROOT/'G1_J1_CPU_overhead.csv',cohorts)
    print('Exact actual plans',sum(r['exact_request_chunk_group_plan'] for r in rows),'/',len(rows))

if __name__=='__main__':
    main()
