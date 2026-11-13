"""Check report tables against independent-start records before publication."""
import re,statistics
from .common import *


def main():
    progress=json.loads((ROOT/'state/confirmation_progress.json').read_text())
    rows=progress['results'];checks=[];failures=[]
    labels={'TTFT p95 (ms)':'TTFT_p95_ms','ITL p99 (ms)':'ITL_p99_ms',
        'TPOT p95 (ms)':'TPOT_p95_ms','E2E p95 (ms)':'E2E_p95_ms','Output tokens/s':'output_tokens_s'}
    arms={'Native':'Native','Continuous DP2':'Colocated','Continuous':'Colocated',
        'Static Last8':'Static','Static Last0':'Static','Revised Dynamic':'Revised','Revised':'Revised'}
    for filename in ('dense_ar_decode_safe_elastic_layer_pd_final_summary.md','dense_ar_four_method_qps_sweep_results.md'):
        cap=16;header=None
        for number,line in enumerate((REPO/'poc_flashvep/reports'/filename).read_text().splitlines(),1):
            if line.startswith('## '):
                match=re.search(r'Output(\d+)',line)
                if match:cap=int(match[1])
                header=None
            if not line.startswith('|'):continue
            parts=[v.strip() for v in line.strip('|').split('|')]
            if parts[:2]==['QPS','Arm']:header=parts;continue
            if not header or len(parts)!=len(header):continue
            try:rate=float(parts[0])
            except ValueError:continue
            if parts[1] not in arms:continue
            selected=[r for r in rows if r['cap']==cap and r['rate']==rate and r['arm']==arms[parts[1]]]
            for index,label in enumerate(header):
                if label not in labels:continue
                text=parts[index];printed=float(text)
                measured=statistics.mean(r['metrics'][labels[label]] for r in selected)
                digits=len(text.split('.')[1]) if '.' in text else 0
                allowed=.5001*10**(-digits)
                check=dict(file=filename,line=number,cap=cap,rate=rate,arm=parts[1],metric=labels[label],
                    printed=printed,measured=measured,allowed_rounding=allowed,passed=abs(printed-measured)<=allowed)
                checks.append(check)
                if not check['passed']:failures.append(check)
    execution=json.loads((ROOT/'state/heldout_runtime_freeze.json').read_text())
    immutable={p:digest(REPO/p)==h for p,h in execution['hashes'].items()}
    output_failures=[r['name'] for r in rows if not r['output']['passed']]
    result=dict(time=time.time(),numeric_checks=len(checks),numeric_failures=failures,
        output_failures=output_failures,frozen_runtime_matches=all(immutable.values()),
        frozen_runtime_checks=immutable,completed_primary_and_reference=len(rows),
        not_run=progress.get('not_run',[]),
        passed=bool(checks) and not failures and not output_failures and all(immutable.values()),
        scope='Check printed start means at their reported precision. Separate scientific limits and confidence intervals remain in the reports.')
    write(ROOT/'state/report_numeric_and_freeze_audit.json',result)
    print('REPORT_NUMERIC_CHECKS',len(checks),'FAILURES',len(failures),'IMMUTABLE_RUNTIME',all(immutable.values()))
    assert result['passed'],failures


if __name__=='__main__':main()
