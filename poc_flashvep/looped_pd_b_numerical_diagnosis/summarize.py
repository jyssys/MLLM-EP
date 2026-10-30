"""Index measured runs, exact parity, work counts, and byte fingerprints."""
import csv
import json
from pathlib import Path
from .common import ROOT,PRIOR,write


def rows(folder):
    path=folder/'requests.jsonl'
    return {r['sample_id']:r for line in path.read_text().splitlines()
            if (r:=json.loads(line)) and not r.get('warmup')} if path.exists() else {}


def comparison(left,right,source=ROOT):
    a,b=rows(source/'raw'/left),rows(source/'raw'/right)
    ids=sorted(a.keys()&b.keys())
    changes=[]
    for i in ids:
        x,y=a[i]['output_ids'],b[i]['output_ids']
        if x!=y:
            pos=next((j for j,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)))
            changes.append(dict(sample_id=i,first_divergence=pos))
    return dict(left=left,right=right,N=len(ids),exact=len(ids)-len(changes),
        input_exact=sum(a[i]['input_ids']==b[i]['input_ids'] for i in ids),
        seed_exact=sum(a[i].get('seed')==b[i].get('seed') for i in ids),
        finish_reason_exact=sum(a[i].get('finish_reason')==b[i].get('finish_reason') for i in ids),
        truncation_exact=sum(a[i].get('truncated')==b[i].get('truncated') for i in ids),
        depth_exact=sum(a[i].get('exit_depths')==b[i].get('exit_depths') for i in ids),
        complete=bool(ids) and len(ids)==len(a)==len(b),differences=changes)


def main():
    records=[]
    checks=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():continue
        status=folder/'status.json'
        if not status.exists():continue
        state=json.loads(status.read_text())
        config=folder/'serving_contract.json'
        cfg=json.loads(config.read_text()) if config.exists() else {}
        requests=rows(folder)
        item=dict(run=folder.name,status=state.get('status'),model=cfg.get('model'),variant=cfg.get('variant'),
                  arm=cfg.get('arm'),N=len(requests),planned=cfg.get('count'),GPUs=cfg.get('gpus'),
                  quiet=cfg.get('quiet',False),profile=cfg.get('nsys',False),
                  instruments=cfg.get('env',{}),error=state.get('error'))
        if state.get('status')=='COMPLETE' and (folder/'after_measured.json').exists():
            before=json.loads((folder/'before_measured.json').read_text())
            after=json.loads((folder/'after_measured.json').read_text())
            if all(isinstance(v,dict) for v in after.values()):
                item['issued_loop_rows']=sum(v.get('issued_loop_rows',0)-before.get(k,{}).get('issued_loop_rows',0) for k,v in after.items())
                depth=4 if cfg['model']=='ouro' else 32
                item['expected_loop_rows']=depth*sum(r['input_tokens']+r['output_tokens']-1 for r in requests.values())
                item['full_depth_work_pass']=item['issued_loop_rows']==item['expected_loop_rows'] and all(
                    len(r['exit_depths'])==r['output_tokens'] and all(d==depth for d in r['exit_depths']) for r in requests.values())
        paths=list(folder.glob('KV_checks-*.jsonl'))
        if paths:
            closed=json.loads((folder/'closed_stats.json').read_text())
            tids={r['transfer_id'] for r in closed['metrics'] if r['request_id'].startswith('r0_')}
            fingerprints=[r for p in paths for line in p.read_text().splitlines()
                          if (r:=json.loads(line)) and r['tid'] in tids]
            item['measured_KV_checks_N']=len(fingerprints)
            item['measured_KV_checks_pass']=sum(r['passed'] for r in fingerprints)
        records.append(item)
    for prefix in ('ouro_original_prefix8_r0','ouro_down_prefix8_r0','ouro_original_qualification200_seed3001',
                   'huginn_original_prefix93_cap72_r0','ouro_old_canonical_seed7001_N400'):
        names=[f'{prefix}_PD_G1_0',f'{prefix}_PD_G1_1',f'{prefix}_B_2']
        if all((ROOT/'raw'/n/'status.json').exists() and json.loads((ROOT/'raw'/n/'status.json').read_text()).get('status')=='COMPLETE' for n in names):
            checks.extend(comparison(names[0],n) for n in names[1:])
    for path in (ROOT/'state').glob('*_qualification.json'):
        state=json.loads(path.read_text())
        for check in state.get('checks',[]):
            if check.get('runs'):
                checks.extend(comparison(check['runs'][0],name) for name in check['runs'][1:])
    h400=['huginn_down_norm_qual_N400_closed_loop_PD_G1_0','huginn_down_norm_qual_N400_closed_loop_PD_G1_1']
    if all((ROOT/'raw'/n/'status.json').exists() and json.loads((ROOT/'raw'/n/'status.json').read_text()).get('status')=='COMPLETE' for n in h400):
        checks.append(comparison(*h400))
    references=[]
    for model in ('ouro','huginn'):
        left=rows(ROOT/'raw'/f'{model}_new_native_HF_triage8_Native_0')
        right=rows(PRIOR/'raw'/f'{model}_reference_HF')
        ids=sorted(left.keys()&right.keys())
        references.append(dict(model=model,N=len(ids),input_exact=sum(left[i]['input_ids']==right[i]['input_ids'] for i in ids),
            output_exact=sum(left[i]['output_ids']==right[i]['output_ids'] for i in ids),
            fresh_native_vs='Frozen official HF eight-prompt trace',
            scope='Numerical triage only. Not benchmark answer accuracy or proof of original-model equivalence.'))
    candidate=ROOT/'raw/huginn_adapter_native_HF_triage8_Native_0'
    if (candidate/'requests.jsonl').exists():
        left=rows(candidate);right=rows(PRIOR/'raw/huginn_reference_HF');ids=sorted(left.keys()&right.keys())
        references.append(dict(model='huginn',variant='down_norm_adapter',N=len(ids),
            input_exact=sum(left[i]['input_ids']==right[i]['input_ids'] for i in ids),
            output_exact=sum(left[i]['output_ids']==right[i]['output_ids'] for i in ids),
            scope='Separate eight-prompt numerical triage. Not benchmark answer accuracy.'))
    write(ROOT/'configs/results_index.json',dict(runs=records,parity=checks,official_HF_triage=references))
    with (ROOT/'correctness_ablation.csv').open('w',newline='') as stream:
        names=list(dict.fromkeys(k for row in checks for k in row))
        if names:
            writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in checks)
    print(dict(runs=len(records),complete=sum(r['status']=='COMPLETE' for r in records),
               failed=sum(r['status']=='FAILED' for r in records),references=references,
               work_failures=[r['run'] for r in records if r.get('full_depth_work_pass') is False]))


if __name__=='__main__':main()
