"""Build trace-backed tables. Never promote diagnostic timing to clean gain."""
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,write
from .cost_predictor import csv_file,path_features
from .metrics import summarize,prefill_exposures,paired_block_interval

def numerical():
    rows=[]
    for folder in sorted((ROOT/'raw').glob('cb_p1_numerical_*')):
        if not (folder/'completed.json').exists():
            continue
        for file in folder.glob('P*.json'):
            data=json.loads(file.read_text());comparison=data.get('comparison',{})
            observations=data['service_observations']
            rows.append(dict(run_id=folder.name,mode=data['mode'],P=data['P'],
                full_R4_work=data['exact_prompt_work']==data['expected_prompt_work'],
                output_depth4=all(d==4 for d in data['depths']),
                tokens_exact=comparison.get('tokens_exact',True),hidden_rel_L2=comparison.get('hidden',{}).get('relative_L2',0.),
                logits_rel_L2=comparison.get('logits',{}).get('relative_L2',0.),KV_rel_L2=comparison.get('KV_max_relative_L2',0.),
                tolerance_pass=data.get('numerical_tolerance_pass',True),
                observations=len(observations),nonnegative_service=all(q['service_ms']>=0 for q in observations),
                alignment_bound_ms=max(q['alignment_bound_ms'] for q in observations),
                ready_at_issue_pct=100*np.mean([all(q['ready_before_issue'].values()) for q in observations]),
                evidence='PRETRAINED_ISOLATED_GPU_NUMERICAL_NOT_TASK_ACCURACY'))
    csv_file(ROOT/'numerical_probe_summary.csv',rows)
    write(ROOT/'numerical_probe_gate.json',dict(cases=len(rows),
        integrity_pass=bool(rows) and all(r['full_R4_work'] and r['output_depth4'] and r['nonnegative_service'] for r in rows),
        tolerance_failures=[r for r in rows if not r['tolerance_pass']],
        token_mismatches=[r for r in rows if not r['tokens_exact']],scope='Isolated probes, not universal batched parity or benchmark quality.'))
    print('Numerical cases',len(rows),'tolerance failures',sum(not r['tolerance_pass'] for r in rows))

def serving():
    rows=[];exposures=[];fairness_raw=defaultdict(dict)
    for pattern in ('cb_p1_fairness_*','cb_p3_*','cb_p5_*'):
        for folder in sorted((ROOT/'raw').glob(pattern)):
            if not (folder/'completed.json').exists():
                continue
            for file in folder.glob('*_T*.json'):
                data=json.loads(file.read_text())
                if 'requests' not in data:
                    continue
                meta=dict(run_id=folder.name,policy=data.get('label',data['policy']),C=data['C'],
                    fairness=data['fairness'],P=data['P'],N=data['configured_N'],trial=data['trial'],
                    GPU=data['physical_GPU'],lane=data['lane'],budget_ms=data.get('configured_budget_ms'))
                rows.append(dict(meta,**summarize(data)))
                if folder.name.startswith('cb_p1_fairness'):
                    fairness_raw[(folder.name,data['policy'],data['C'],data['trial'])][data['fairness']]=data
                for exposure in prefill_exposures(data):
                    exposures.append(dict(meta,**exposure))
    csv_file(ROOT/'serving_cohort_metrics.csv',rows);csv_file(ROOT/'output_prefill_exposures.csv.gz',exposures)
    comparisons=[]
    cells=defaultdict(dict)
    for row in rows:
        if row['run_id'].startswith('cb_p1_fairness'):
            cells[(row['run_id'],row['trial'],row['policy'],row['C'])][row['fairness']]=row
    for policy in sorted({k[2] for k in cells}):
        for P in (2033,8177):
            blocks=defaultdict(list)
            for key,value in cells.items():
                if key[2]==policy and set(value)=={'stock','completion'} and value['stock']['P']==P:
                    blocks[key[0]].append(value)
            for metric in ('TTFT_P95_ms','ITL_pooled_P95_ms','ITL_pooled_P99_ms','drained_output_tokens_s'):
                pairs=[(np.mean([v['stock'][metric] for v in values]),np.mean([v['completion'][metric] for v in values])) for values in blocks.values()]
                ci=paired_block_interval(pairs)
                if ci:
                    comparisons.append(dict(policy=policy,P=P,metric=metric,**ci,
                        contrast='completion minus stock fairness; no optimizer',evidence='EXPLORATORY_SHARED_HOST_PAIRED_BLOCKS'))
    csv_file(ROOT/'fairness_paired_effects.csv',comparisons)
    parity=[]
    for (run,policy,C,trial),pair in fairness_raw.items():
        if set(pair)!={'stock','completion'}:
            continue
        a={r['id']:r for r in pair['stock']['requests']};b={r['id']:r for r in pair['completion']['requests']}
        def executed(data,rid):
            return sum(s['tokens']*s['G'] for q in data['service_observations'] for s in q['spans'] if s['id']==rid and s['phase']=='recurrent')
        for rid,request in a.items():
            other=b[rid]
            parity.append(dict(run_id=run,policy=policy,C=C,trial=trial,P=pair['stock']['P'],id=rid,
                tokens_exact=request['output_ids']==other['output_ids'],length_exact=len(request['output_ids'])==len(other['output_ids']),
                stock_output_tokens=len(request['output_ids']),fair_output_tokens=len(other['output_ids']),
                stock_executed_decode_loop_work=executed(pair['stock'],rid),
                fair_executed_decode_loop_work=executed(pair['completion'],rid),
                stock_finish=request['finish_reason'],fair_finish=other['finish_reason'],
                scope='Same prompts and fixed C/G. Natural outputs can change with scheduling geometry.'))
    csv_file(ROOT/'fairness_output_parity.csv',parity)
    print('Serving cohorts',len(rows),'client gaps',len(exposures))

def main():
    numerical();serving()

if __name__=='__main__':
    main()
