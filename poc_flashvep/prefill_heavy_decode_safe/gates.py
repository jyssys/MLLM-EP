"""Reject perturbed or incomplete traces before any physical-capacity claim."""
import argparse,json
from .common import ROOT,REPO,initialize,write


def interference(candidate,reference,fields):
    ratios={key:candidate[key]/reference[key] for key in fields}
    return dict(ratios=ratios,passed=all(.95<=v<=1.05 for v in ratios.values()),
        scope='Two-sided five-percent point screen. Independent-start variation requires a separate audit.')


def same_window(p,d):
    return p['run']==d['run'] and p['left']==d['left'] and p['right']==d['right'] and p['right']>p['left']


def model_contract(model,length,cap,special_tokens=0):
    return cap>=16 and length>0 and (model!='huginn' or length+cap+special_tokens<=4096)


def prior_audit():
    initialize()
    source=REPO/'poc_flashvep/reports/huginn_loop28_decode_tail_diagnosis_20261111/analysis/output_idle_diagnosis.json'
    value=json.loads(source.read_text());records=[]
    for row in value['rows']:
        records.append(dict(output=row['output_cap'],qualified=row['strict_capture_qualified'],
            screens=row.get('instrumentation'),capture_status=row.get('capture_status'),
            scope='Frozen prior diagnostic. Not new GPU execution.'))
    result=dict(source=str(source),passed=all(r['qualified'] for r in records),records=records,
        reason='Prior captures do not qualify a clean same-window Output16 physical-idle claim.')
    write(ROOT/'analysis/prior_profile_gate_reproduction.json',result)
    print(json.dumps(result));return result['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-prior',action='store_true',required=True)
    p.parse_args();raise SystemExit(0 if prior_audit() else 1)
