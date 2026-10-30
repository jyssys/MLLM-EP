"""Validate every fixed cut against one repeated full-PD reference."""
import argparse
import json
from .common import ROOT,write
from .gate import read


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=('ouro','huginn'));a=p.parse_args()
    model=a.model;depth=4 if model=='ouro' else 32
    name=f'gate_{model}_allcuts_L128';folder=ROOT/'raw'/name
    reference=ROOT/'raw'/f'gate_{model}_L128_cut{depth}_v3'
    baseline={r['seed']:r for r in read(reference/'requests.jsonl') if not r['warmup']}
    baseline_states={r['seed']:r for path in reference.glob('prompt_states_*.jsonl') for r in read(path)}
    candidates=[r for r in read(folder/'requests.jsonl') if not r['warmup']]
    states=[r for path in folder.glob('prompt_states_*.jsonl') for r in read(path)]
    byte_checks=[r for path in folder.glob('KV_checks_*.jsonl') for r in read(path)]
    results=[]
    for cut in sorted({r['cut'] for r in candidates}):
        rows=[r for r in candidates if r['cut']==cut]
        token_exact=sum(all(row[k]==baseline[row['seed']][k] for k in ('input_ids','output_ids','exit_depths','finish_reason')) for row in rows)
        selected=[r for r in states if r['cut']==cut]
        hidden_exact=sum(r['hidden_sha256']==baseline_states[r['seed']]['hidden_sha256'] for r in selected)
        KV_exact=sum(r['planes']==baseline_states[r['seed']]['planes'] for r in selected)
        work=json.loads((folder/f'cohort_cut{cut}_stats.json').read_text())
        issued=sum(work['after'][k]['issued_loop_rows']-work['before'][k]['issued_loop_rows'] for k in work['before'])
        required=sum((r['input_tokens']+r['output_tokens']-1)*depth for r in rows)
        bytes_for_cut=[r for r in byte_checks if r['cut']==cut]
        result=dict(cut=cut,N=len(rows),token_exact=token_exact,hidden_checks=len(selected),
                    hidden_exact=hidden_exact,KV_exact=KV_exact,receiver_checks=len(bytes_for_cut),
                    receiver_exact=sum(r['passed'] for r in bytes_for_cut),issued_loop_rows=issued,required_loop_rows=required)
        result['passed']=(len(rows)==token_exact==8 and len(selected)==hidden_exact==KV_exact>0
                          and issued==required and all(r['passed'] for r in bytes_for_cut)
                          and (cut==0 or bool(bytes_for_cut)))
        results.append(result)
    out=dict(model=model,reference=reference.name,candidate=name,results=results,passed=all(r['passed'] for r in results),
             evidence='MEASURED_HEAVY_CORRECTNESS',limitation='8 requests per cut, plus warm-up. Not downstream accuracy or universal losslessness.')
    write(ROOT/'state'/f'all_cut_gate_{model}.json',out);print(out)
    if not out['passed']:raise SystemExit(1)


if __name__=='__main__':main()
