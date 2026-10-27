"""Score full validation trajectories. Dense emulation does not reduce GPU work."""
import json
import statistics
from .common import ROOT,write,reference_helpers
from .audit import wilson
from .postprocess import paired_bootstrap
from .refinement_analysis import csvout

def main():
    truth=json.loads((ROOT/'manifests/validation_truth.json').read_text())
    parse=reference_helpers().parse_gsm8k
    reference={}
    for group in ('A','B'):
        for path in (ROOT/'ep4'/f'{group}_pilot_validation_trace').glob('gsm*/bypass.result.rank0.json'):
            record=json.loads(path.read_text())
            reference[record['sample_id']]=record
    def score(record):
        answer=truth[record['sample_id']].split('####')[-1].strip().replace(',','')
        parsed,rule=parse(record['postprocessed_generation'])
        return parsed==answer,parsed,rule
    assert len(reference)==16
    baseline={key:score(record)[0] for key,record in reference.items()}
    summary=[]
    samples=[]
    for folder in sorted((ROOT/'ep4').glob('*_semantic_P*')):
        if 'semantic_P1_' in folder.name and not folder.name.endswith('_uniform_v2'):
            continue  # Preserve the superseded structured-hash diagnostic, not a uniform baseline.
        if not (folder/'status.json').exists():
            continue
        status=json.loads((folder/'status.json').read_text())['status']
        data=[json.loads(p.read_text()) for p in sorted(folder.glob('gsm*/bypass.result.rank0.json'))]
        if not data:
            continue
        left=[]
        right=[]
        bad=[]
        for record in data:
            key=record['sample_id']
            old=reference[key]
            correct,parsed,rule=score(record)
            left.append(baseline[key])
            right.append(correct)
            if baseline[key] and not correct:
                bad.append(key)
            samples.append(dict(run_id=folder.name,sample_id=key,correct=correct,
                baseline_correct=baseline[key],parsed_answer=parsed,parser_rule=rule,
                output_agreement=record['output_ids']==old['output_ids'],
                first_divergence=next((i for i,(a,b) in enumerate(zip(old['output_ids'],record['output_ids'])) if a!=b),None),
                nfe=record['nfe'],baseline_nfe=old['nfe'],output_tokens=record['generation_tokens'],
                termination=record['termination_reason'],emulation=record['emulation'],
                evidence='MEASURED_FREE_RUNNING_DENSE_SEMANTIC_INTERVENTION_NOT_ACCELERATION'))
        n=len(data)
        k=sum(right)
        summary.append(dict(run_id=folder.name,policy=data[0]['emulation']['kind'],
            requested_fraction=data[0]['emulation']['requested_fraction'],status=status,correct=k,n=n,
            accuracy_percent=100*k/n,wilson_95_percent=[100*x for x in wilson(k,n)],
            baseline_correct=sum(left),accuracy_difference_pp=100*(k-sum(left))/n,
            paired_bootstrap_95_pp=paired_bootstrap(left,right),
            output_agreement=sum(r['output_ids']==reference[r['sample_id']]['output_ids'] for r in data),
            nfe_ratio=statistics.mean(r['nfe']/reference[r['sample_id']]['nfe'] for r in data),
            semantic_edge_fraction=sum(r['emulation']['semantically_zeroed_edges'] for r in data)/sum(r['emulation']['full_routed_edges_executed'] for r in data),
            omitted_weight_mass_fraction=statistics.mean(r['emulation']['omitted_original_weight_mass_fraction'] for r in data),
            actual_dispatch_edge_reduction=0.,actual_expert_kernel_work_reduction=0.,
            caps=sum(r['termination_reason']=='GEN_LENGTH_CAP' for r in data),
            baseline_correct_to_wrong=bad,
            evidence='MEASURED_FREE_RUNNING_DENSE_SEMANTIC_INTERVENTION_NOT_ACCELERATION'))
    out=ROOT/'pilot'
    write(out/'sensitivity_summary.json',dict(baseline_correct=sum(baseline.values()),baseline_n=16,
        conditions=summary,limitations=['Validation screen only. No test-label tuning.',
            'All expert kernels and original dispatch IDs still execute.',
            'N=16 cannot establish a 0.5 percentage-point quality margin.']))
    csvout(out/'sensitivity_summary.csv',summary)
    with (out/'sensitivity_samples.jsonl').open('w') as stream:
        for row in samples:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(dict(baseline_correct=sum(baseline.values()),baseline_n=16,conditions=summary),indent=2))

if __name__=='__main__':
    main()
