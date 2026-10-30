"""Build reports from preserved measurements and publish only owned paths."""
import csv
import datetime
import gzip
import json
import math
import os
import random
import shutil
import subprocess
from .common import *
from .analyze import csv_write

def read_csv(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []

def number(value):
    try:return float(value)
    except (TypeError,ValueError):return None

def fmt(value,digits=2):
    n=number(value)
    return f'{n:.{digits}f}' if n is not None else 'NOT_MEASURED'

def table(rows,columns):
    text='| '+' | '.join(label for label,key in columns)+' |\n'
    text+='| '+' | '.join('---' for _ in columns)+' |\n'
    for row in rows:text+='| '+' | '.join(str(row.get(key,'NOT_MEASURED')).replace('|','/') for label,key in columns)+' |\n'
    return text

def patch_report(name,content):
    path=REPO/'poc_flashvep/reports'/name
    executable=os.environ.get('LTP_APPLY_PATCH') or shutil.which('apply_patch')
    if not executable:raise RuntimeError('apply_patch is unavailable. Reports were not written.')
    if path.exists():
        old=path.read_text()
        patch='*** Begin Patch\n*** Update File: '+str(path.relative_to(REPO))+'\n@@\n'
        patch+=''.join('-'+line+'\n' for line in old.splitlines())
        patch+=''.join('+'+line+'\n' for line in content.splitlines())+'*** End Patch\n'
    else:patch='*** Begin Patch\n*** Add File: '+str(path.relative_to(REPO))+'\n'+''.join('+'+line+'\n' for line in content.splitlines())+'*** End Patch\n'
    subprocess.run([executable],input=patch,text=True,cwd=REPO,check=True)

def paired(rows):
    result=[]
    for model in ('ouro','huginn'):
        for arm in ('PD_G1','DP','Loop_Static','Token_Dynamic'):
            values=[]
            for start in (1,2,3):
                find=lambda method:next((r for r in rows if r['model']==model and r['workload']=='W1' and r['arm']==method
                    and r['phase']=='repeat' and int(r['start'])==start and r['status']=='COMPLETE'),None)
                a=find('Loop_Dynamic');b=find(arm)
                if a and b and number(a.get('SLO_goodput_per_s')) is not None and number(b.get('SLO_goodput_per_s')) is not None:
                    ga=float(a['SLO_goodput_per_s']);gb=float(b['SLO_goodput_per_s'])
                    values.append((ga-gb,100*(ga/gb-1) if gb else None))
            row=dict(model=model,workload='W1',rate_multiplier=1.5,candidate='Loop_Dynamic',baseline=arm,
                paired_independent_starts=len(values),mean_goodput_difference=sum(v[0] for v in values)/len(values) if values else 'NOT_AVAILABLE',
                relative_goodput_gain_percent='NOT_AVAILABLE',CI95_low_percent='NOT_AVAILABLE',CI95_high_percent='NOT_AVAILABLE',
                scope='Paired start bootstrap. Fixed prompt set. Not population/domain generalization or proof of stable capacity.')
            if values and all(v[1] is not None for v in values):
                gains=[v[1] for v in values];row['relative_goodput_gain_percent']=sum(gains)/len(gains)
                if len(values)>=3:
                    rng=random.Random(2026103500);samples=sorted(sum(rng.choices(gains,k=len(gains)))/len(gains) for _ in range(10000))
                    row.update(CI95_low_percent=samples[249],CI95_high_percent=samples[9749])
            result.append(row)
    csv_write(ROOT/'paired_start_comparisons.csv',result)
    return result

def figures(rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        count=0
        for model in ('ouro','huginn'):
            if not any(r['model']==model and r['phase']=='primary' and r['status']=='COMPLETE' for r in rows):continue
            fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
            for w,ax in zip(('W1','W2','W3','W4'),axes.flat):
                data=[r for r in rows if r['model']==model and r['workload']==w and r['phase']=='primary' and r['status']=='COMPLETE']
                data.sort(key=lambda r:ARMS.index(r['arm']))
                ax.bar([r['arm'] for r in data],[float(r.get('SLO_attainment',0))*100 for r in data])
                ax.axhline(90,color='black',linestyle='--');ax.set_ylim(0,105)
                ax.set_ylabel('Joint SLO attainment (%)');ax.set_title(w+' / one primary start')
                ax.tick_params(axis='x',rotation=60)
            fig.suptitle(model+' / actual controlled generation / finite cohorts, not stable capacity')
            fig.savefig(ROOT/'figures'/f'{model}_primary_slo_attainment.png',dpi=160)
            fig.savefig(ROOT/'figures'/f'{model}_primary_slo_attainment.svg');plt.close(fig)
            fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
            for arm in ARMS:
                points={}
                for r in rows:
                    if r['model']==model and r['workload']=='W1' and r['phase'] in ('primary','repeat') and r['arm']==arm and r['status']=='COMPLETE':
                        points.setdefault(float(r['rate']),[]).append(float(r['SLO_attainment'])*100)
                if points:ax.plot(sorted(points),[sum(points[k])/len(points[k]) for k in sorted(points)],marker='o',label=arm)
            ax.axhline(90,color='black',linestyle='--');ax.set_ylim(0,105);ax.set_xlabel('Offered requests/s')
            ax.set_ylabel('Joint SLO attainment (%)');ax.legend(fontsize=8)
            ax.set_title(model+' W1 / measured rate points only / finite controlled cohorts')
            fig.savefig(ROOT/'figures'/f'{model}_w1_measured_rate_points.png',dpi=160)
            fig.savefig(ROOT/'figures'/f'{model}_w1_measured_rate_points.svg');plt.close(fig)
            count+=4
        write(ROOT/'state/figures.json',dict(status='CREATED' if count else 'NOT_AVAILABLE',figures=count,time=time.time()))
    except Exception as error:write(ROOT/'state/figures.json',dict(status='FAILED',error=repr(error)))

def reports():
    rows=read_csv('per_run_metrics.csv');matrix=read_csv('experiment_matrix.csv');pairs=paired(rows);figures(rows)
    comparisons=[]
    for model in ('ouro','huginn'):
        for w in ('W1','W2','W3','W4'):
            find=lambda arm:next((r for r in rows if r['model']==model and r['workload']==w and r['phase']=='primary' and r['arm']==arm and r['status']=='COMPLETE'),None)
            candidate=find('Loop_Dynamic')
            for arm in ARMS:
                if arm=='Loop_Dynamic':continue
                baseline=find(arm)
                result=dict(model=model,workload=w,candidate='Loop_Dynamic',baseline=arm,
                    p95_E2E_speedup='NOT_AVAILABLE',SLO_goodput_ratio='NOT_AVAILABLE',attainment_difference_pp='NOT_AVAILABLE',
                    scope='ONE_START_EXPLORATORY_NO_QUALIFYING_CI')
                if candidate and baseline:
                    e=float(candidate['e2e_ms_p95']);a=number(candidate.get('SLO_goodput_per_s'));b=number(baseline.get('SLO_goodput_per_s'))
                    result['p95_E2E_speedup']=float(baseline['e2e_ms_p95'])/e if e else 'NOT_AVAILABLE'
                    if a is not None and b:result['SLO_goodput_ratio']=a/b
                    result['attainment_difference_pp']=100*(float(candidate['SLO_attainment'])-float(baseline['SLO_attainment']))
                comparisons.append(result)
    csv_write(ROOT/'primary_dynamic_comparisons.csv',comparisons)
    verdicts={}
    for model in ('ouro','huginn'):
        cells=[r for r in matrix if r['model']==model]
        failed=any(r['status'].startswith('BLOCKED_CORRECTNESS') for r in cells)
        complete=sum(r['status']=='COMPLETE' for r in cells)
        p=[r for r in pairs if r['model']==model and r['baseline'] in ('DP','Loop_Static','Token_Dynamic')]
        negative=all(r['paired_independent_starts']>=3 and number(r['CI95_high_percent']) is not None and float(r['CI95_high_percent'])<=0 for r in p)
        budget_incomplete=remaining()<3600 or any('TIME_BUDGET' in r['status'] for r in cells)
        verdicts[model]='BLOCKED_CORRECTNESS' if failed else 'NOT_RUN_TIME_BUDGET' if complete<28 and budget_incomplete else 'NO_GO_PERFORMANCE' if complete==28 and negative else 'CONDITIONAL'
    completed=sum(r['status']=='COMPLETE' for r in matrix)
    wall=(time.time()-START_EPOCH)/3600
    summary=f'''# Two-GPU Dynamic Loop-versus-Token PD: Final Summary

Campaign start: {START_UTC}.
Report generation elapsed time: {wall:.3f} hours.
This timestamp precedes the final Git push.
Primary cells accepted as complete: {completed}/56.
Primary cells incomplete or rejected: {56-completed}/56.
Branch: `flashvep/adaptive-loop-token-pd-2gpu-w1-w4`.

## Decisions

| Model | Verdict |
|---|---|
| Ouro R4 | {verdicts['ouro']} |
| Huginn R32 | {verdicts['huginn']} |

CONDITIONAL means the evidence does not establish a stable-capacity advantage.
NO_GO_PERFORMANCE applies only to the implemented policy and completed conditions.
It does not prove that every dynamic partitioning policy must fail.
BLOCKED_CORRECTNESS prevents a valid complete seven-arm comparison for the affected model.
Unfinished configurations remain separate from measured negative results.

## What was implemented

The runtime uses actual two-GPU P/D ownership changes.
Token migration transfers full-depth prefix KV before D executes the suffix.
Loop migration transfers valid depth-specific KV and complete prompt carrier rows.
Huginn Token migration also transfers unused canonical initialization states.
The online methods select provisional boundaries from observed runtime state.
They recheck D admission at the export frontier and can finish Prefill on P.
All subsequent Decode tokens execute full fixed depth on D.
DP and DP_Chunked use two independent colocated replicas.
Shared-GEMM Prefill/Decode Mixed co-batching is off in every main arm.

## Correctness and quality limits

Both models passed the initial seven-arm 32-request migration qualification.
There were 192/192 non-reference output and Prompt-state comparisons per model.
The fixture contains repeated prompt content, not 192 distinct benchmark questions.
Zero/full endpoints and complete long-prompt Loop/Token migration also passed finite checks.
Huginn's additional normalized Token grid has separate qualification evidence.
The main serving harness rejects unexplained controlled-output mismatches immediately.
Rejected cells do not become valid performance results.
Original-HF equivalence and GSM8K/MATH accuracy are NOT_ESTABLISHED in this campaign.

## Evidence and primary-metric limits

Main runs generate real greedy outputs with declared controlled targets.
They use ignore_eos and are not Natural-EOS results or frozen-token replay.
W1 targets 384/448/512 tokens in 25/50/25 percent proportions.
Huginn's prompt plus output target never exceeds 4,096 tokens.
The primary SLO combines TTFT and request-average TPOT.
It is not a request-level maximum-ITL SLO.
The SLO thresholds were frozen from disjoint PD_G1 calibration before candidate serving.
All failures and missing requests remain in offered-request denominators.

Each primary point has one 32-request or 24-request start unless the frozen reduced matrix applies.
Secondary W1 trials use the predeclared 1.5-rate point and independent server starts.
Base-rate and 1.5-rate trials are not pooled for confidence intervals.
The exact completed start counts appear in the paired comparison table.
Finite-cohort goodput includes final drain time.
Maximum indefinitely sustainable serving capacity is NOT_ESTABLISHED.
A passing finite rate grid point is not proof of stationary queue stability.

## Transport and novelty controls

PD_G1 uses native completed-token-chunk streaming.
Loop and Token methods use acknowledged snapshot handoff.
Separate full-cut W1 snapshot controls isolate that protocol change where completed.
Protocol-only gains are not Loop-partitioning gains.
The two axes use matched normalized candidate fractions before token-chunk alignment.
The online rule is conservative and is not an exhaustive optimizer.
It does not dynamically select the D-local Cut0 endpoint.
Generic logical-layer equivalence remains an inherited novelty limitation.
Layer Split and 4/8GPU pooled serving are outside this campaign.
Loop-versus-Token performance alone cannot prove Loop-exclusive novelty.

## Timing and resource limits

Clean runs use one physical GPU pair at a time.
All methods in one workload/model use the same pair.
Other managed GPU experiments and burners remain absent from that clean lane.
This is not a claim of complete CPU-host exclusivity.
The resource manager restores approved burners after measurement.
The report preserves all five failed development namespaces and rejected prototypes.

`e2e_component_shares_aggregate.csv` contains exclusive host-observed request-time percentages.
Its denominator is summed request E2E, not the cohort critical path.
Those percentages are not exclusive GPU attention, GEMM, or DMA shares.
NIXL cumulative completion observations are not pure DMA time.
Overlapping device durations are never added to claim an E2E speedup.
Optional profiler results remain separate from clean latency.

## Artifact index

- [Main matrix](adaptive_loop_token_pd_2gpu_w1_w4_matrix.md)
- [Correctness and protocol](adaptive_loop_token_pd_2gpu_correctness_and_protocol.md)
- [Online policy and SLO](adaptive_loop_token_pd_2gpu_online_policy_and_slo.md)
- [All measured run statistics](adaptive_loop_token_pd_2gpu/per_run_metrics.csv)
- [Mandatory executed and missing cells](adaptive_loop_token_pd_2gpu/experiment_matrix.csv)
- [Paired start uncertainty](adaptive_loop_token_pd_2gpu/paired_start_comparisons.csv)
- [Host E2E percentages](adaptive_loop_token_pd_2gpu/e2e_component_shares_aggregate.csv)
- [Communication payloads](adaptive_loop_token_pd_2gpu/communication_payloads.csv)
- [Actual online cuts](adaptive_loop_token_pd_2gpu/online_cut_outcomes.csv)
- [Frozen SLO calibration](adaptive_loop_token_pd_2gpu/slo_calibration.json)
- [Complete aggregated requests, lossless gzip](adaptive_loop_token_pd_2gpu/per_request.jsonl.gz)

Raw requests preserve input IDs, output IDs, exit depths, stop reasons, and token timestamps.
Every job preserves configuration, source archives, GPU mapping, and complete logs.
The uncompressed aggregate remains local. Its complete gzip archive is published instead of an oversized Git object.
Each individual run also retains its plain request JSONL file.
`commands.jsonl` contains the actual GPU launch commands.
`controller_commands.jsonl` contains the phase commands.
Model and source pins are in `configs/initial_contract.json` and `configs/frozen_execution_hashes.json`.
The final Git SHA is the verified branch tip, not an assumed clean-worktree source identity.
'''
    selection_path=ROOT/'static_selection.json'
    if selection_path.exists():
        choices=json.loads(selection_path.read_text())['selected'];selected=[]
        for model in ('ouro','huginn'):
            for w in ('W1','W2','W3','W4'):
                loop=choices.get(f'{model}_{w}_loop',{});token=choices.get(f'{model}_{w}_token',{})
                selected.append(dict(model=model,workload=w,loop_cut=loop.get('cut','NOT_MEASURED'),
                    token_fraction=token.get('prefix_fraction','NOT_MEASURED')))
        summary+='\n## Calibration-selected partitions\n\n'
        summary+=table(selected,[('Model','model'),('Workload','workload'),('Loop cut','loop_cut'),('Token P-prefix fraction','token_fraction')])
        summary+='\nThese are limited calibration choices, not proven global optima for the main workloads.\n'
        summary+='Changing workload also changes prompt and output lengths. It does not isolate queue state alone.\n'
    summary+='\n## Answers to the serving questions\n\n'
    summary+='Q1: The table above records workload-dependent calibration choices. The full main cut map is not measured.\n\n'
    summary+='Q2 and Q3: The following ratios compare actual online Loop decisions with selected Static and Token baselines.\n'
    selected=[r for r in comparisons if r['baseline'] in ('Loop_Static','Token_Dynamic')]
    summary+=table([{k:fmt(v,4) if isinstance(v,float) else v for k,v in r.items()} for r in selected],
        [('Model','model'),('Workload','workload'),('Baseline','baseline'),('p95 E2E speedup','p95_E2E_speedup'),('SLO-goodput ratio','SLO_goodput_ratio')])
    summary+='\nThese primary ratios have one independent start. The equal-rate repeated table provides separate paired uncertainty.\n'
    summary+='A ratio above one does not by itself establish maximum stable capacity.\n\n'
    summary+='Q4: Loop and Token arms use the same configured token chunk size. Token Split is not forced to use smaller chunks.\n'
    summary+='Exclusive clean GEMM efficiency attribution is NOT_MEASURED. Optional profiles are separate diagnostic evidence.\n\n'
    summary+='Q5: communication_payloads.csv records actual state payloads and NIXL completion counts.\n'
    summary+='It includes the additional Huginn suffix initialization state. Payload reduction is not assumed to be an E2E speedup.\n\n'
    summary+='Q6: online_cut_outcomes.csv separates initial Full-PD choices, accepted continuations, and source-side fallbacks.\n'
    summary+='If every request uses Full-PD, any native-streaming difference is not evidence of adaptive partitioning.\n\n'
    credits=[]
    for model in ('ouro','huginn'):
        for w in ('W1','W2','W3','W4'):
            folder=ROOT/'raw'/f'primary_{model}_{w.lower()}_loop_dynamic_start0'
            req_path=folder/'requests.jsonl';plan_path=folder/'online_decisions.jsonl'
            if not req_path.exists() or not plan_path.exists():continue
            requests=[json.loads(s) for s in req_path.read_text().splitlines() if s]
            requests={f'r{r["run"]}_q{r["sample_id"]}':r for r in requests if not r['warmup']}
            plans=[json.loads(s) for s in plan_path.read_text().splitlines() if s]
            plans=[r for r in plans if r['event']=='INITIAL_PLAN' and r['request_id'] in requests]
            blocked=0
            for r in plans:
                req=requests[r['request_id']];required=math.ceil((req['input_tokens']+req['output_tokens']-1)/16)*MODEL_PINS[model][2]
                blocked+=r['state']['d_free_blocks']<required
            if plans:credits.append(dict(model=model,workload=w,N=len(plans),
                mean_active_decode_at_decision=fmt(sum(r['state']['active_decode'] for r in plans)/len(plans)),
                insufficient_D_credit_decisions=blocked,
                insufficient_D_credit_percent=fmt(100*blocked/len(plans))))
    summary+='Q7: The table below checks observed Decode occupancy and full-request KV reservation credit.\n'
    summary+='The credit calculation uses the runtime block formula and the declared controlled output budget.\n'
    summary+='It is structural accounting from observed reservations, not GPU kernel-time attribution.\n\n'
    summary+=table(credits,[('Model','model'),('Workload','workload'),('Completed decision N','N'),
        ('Mean observed active Decode','mean_active_decode_at_decision'),('Insufficient D credit N','insufficient_D_credit_decisions'),
        ('Insufficient D credit %','insufficient_D_credit_percent')])
    summary+='\nEvery cut still reserves D KV for full-depth Decode. Moving Prefill work does not remove this reservation requirement.\n'
    summary+='Ouro and Huginn also differ in width, recurrent structure, boundary state, and supported async execution.\n'
    summary+='This experiment does not isolate one of these differences as the sole measured cause.\n'
    patch_report('adaptive_loop_token_pd_2gpu_final_summary.md',summary)
    body='# Two-GPU W1–W4: Measured Seven-Arm Matrix\n\n'
    body+='Only COMPLETE cells passed the recorded output and work checks. Other status values remain explicit.\n\n'
    for model in ('ouro','huginn'):
        for w in ('W1','W2','W3','W4'):
            body+=f'## {model} / {w}\n\n'
            shown=[]
            for arm in ARMS:
                r=next((r for r in rows if r['model']==model and r['workload']==w and r['arm']==arm and r['phase']=='primary'),None)
                entry=dict(arm=arm,status=next(c['status'] for c in matrix if c['model']==model and c['workload']==w and c['arm']==arm))
                if r:
                    entry.update(GPUs=r['GPU_ids'],N=r['completed_N']+'/'+r['expected_N'],rate=fmt(r.get('rate'),4),
                        attainment=fmt(100*float(r['SLO_attainment'])) if r.get('SLO_attainment') else 'NOT_MEASURED',
                        goodput=fmt(r.get('SLO_goodput_per_s'),5))
                    for key in ('ttft_ms_p95','tpot_ms_p95','ITL_ms_p95','ITL_ms_p99','e2e_ms_p95'):entry[key]=fmt(r.get(key))
                shown.append(entry)
            body+=table(shown,[('Arm','arm'),('Status','status'),('Physical GPUs','GPUs'),('Completed/offered','N'),
                ('Offered req/s','rate'),('Joint SLO %','attainment'),('Goodput req/s','goodput'),
                ('TTFT p95 ms','ttft_ms_p95'),('TPOT p95 ms','tpot_ms_p95'),('ITL p95 ms','ITL_ms_p95'),
                ('ITL p99 ms','ITL_ms_p99'),('E2E p95 ms','e2e_ms_p95')])+'\n'
    body+='## Full metric and uncertainty tables\n\n'
    body+='The CSV includes request-latency mean/p50/p95/p99, throughput, memory, communication, failures, and depth accounting.\n'
    body+='ITL quantiles use observed single-token delivery intervals. Multi-token events remain counted separately.\n'
    body+='One primary start does not provide a qualifying independent-start confidence interval.\n'
    body+='The paired table separates each repeated 1.5-rate W1 comparison.\n\n'
    body+=table([{k:fmt(v) if isinstance(v,float) else v for k,v in r.items()} for r in pairs],
        [('Model','model'),('Baseline','baseline'),('Paired starts','paired_independent_starts'),
         ('Relative goodput gain %','relative_goodput_gain_percent'),('95% low','CI95_low_percent'),('95% high','CI95_high_percent')])
    patch_report('adaptive_loop_token_pd_2gpu_w1_w4_matrix.md',body)
    correctness_path=REPO/'poc_flashvep/reports/adaptive_loop_token_pd_2gpu_correctness_and_protocol.md'
    correctness=correctness_path.read_text().split('\n## Final controlled-serving status')[0]
    correctness=correctness.replace('Status: FINITE_MIGRATION_GATES_PASS. Serving qualification remains in progress.',
        'Status: FINITE_MIGRATION_GATES_PASS. Final controlled-serving results follow below.')
    correctness=correctness.replace('New 384–512-token generation parity remains PENDING until those cells execute.',
        'Controlled-generation parity is summarized below for the cells that actually completed.')
    correctness+='\n## Final controlled-serving status\n\n'
    quality=[]
    for model in ('ouro','huginn'):
        candidates=[r for r in rows if r['model']==model and r['phase']=='primary' and r['arm']!='PD_G1']
        accepted=[r for r in candidates if r['status']=='COMPLETE']
        quality.append(dict(model=model,accepted_candidate_cells=len(accepted),
            matched_output_requests=sum(int(r['completed_N']) for r in accepted),
            rejected_or_incomplete_candidate_cells=24-len(accepted),verdict=verdicts[model]))
    correctness+=table(quality,[('Model','model'),('Accepted candidate cells','accepted_candidate_cells'),
        ('Matched controlled outputs','matched_output_requests'),('Incomplete/rejected candidate cells','rejected_or_incomplete_candidate_cells'),('Model verdict','verdict')])
    correctness+='\nEach accepted candidate sequence matches the same-start PD_G1 reference exactly.\n'
    correctness+='These counts are request instances, not benchmark accuracy questions.\n'
    correctness+='Clean serving does not repeat the heavy per-plane byte hashes on every request.\n'
    correctness+='The finite byte/state gates, actual transfer completions, output checks, and work counters remain separate evidence.\n'
    correctness+='Original-HF fidelity and downstream accuracy remain NOT_ESTABLISHED.\n'
    patch_report(correctness_path.name,correctness)
    policy_path=REPO/'poc_flashvep/reports/adaptive_loop_token_pd_2gpu_online_policy_and_slo.md'
    policy=policy_path.read_text().split('\n## Final measured outcomes')[0]
    policy=policy.replace('Status: IN_PROGRESS. Measured serving comparisons are PENDING.',
        'Status: measured outcomes are attached below. Unmeasured cells remain explicit.')
    policy=policy.replace('Final online decisions, gain tables, rate sweeps, and model verdicts are PENDING.',
        'The following section summarizes the preserved online choices and measured comparisons.')
    policy+='\n## Final measured outcomes\n\n'
    outcomes=read_csv('online_cut_outcomes.csv');choices=[]
    for model in ('ouro','huginn'):
        for w in ('W1','W2','W3','W4'):
            for arm in ('Loop_Dynamic','Token_Dynamic'):
                selected=[r for r in outcomes if r['model']==model and r['workload']==w and r['arm']==arm and r['name'].startswith('primary_')]
                if not selected:continue
                histogram={}
                for r in selected:
                    key=r['actual_cut'] if arm=='Loop_Dynamic' else r['actual_prefix_k']
                    histogram[key]=histogram.get(key,0)+1
                choices.append(dict(model=model,workload=w,arm=arm,N=len(selected),
                    D_continuations=sum(float(r['actual_D_continuation_fraction'])>0 for r in selected),
                    source_fallbacks=sum(r['full_PD_fallback']=='True' for r in selected),
                    actual_boundary_histogram=json.dumps(histogram,sort_keys=True)))
    policy+=table(choices,[('Model','model'),('Workload','workload'),('Arm','arm'),('Completed N','N'),
        ('D continuations','D_continuations'),('P-full fallbacks','source_fallbacks'),('Actual boundary/count','actual_boundary_histogram')])
    policy+='\nToken-prefix values can change only because prompt length and legal alignment change.\n'
    policy+='Boundary variation alone is not evidence of adaptive benefit.\n'
    policy+='The main comparison CSV separates SLO-goodput ratios from p95 E2E speedups.\n'
    policy+='The paired-start CSV contains only equal-rate repeated comparisons.\n'
    policy+='Stable capacity remains NOT_ESTABLISHED for this finite experiment.\n'
    patch_report(policy_path.name,policy)
    write(ROOT/'final_decisions.json',dict(verdicts=verdicts,accepted_primary_cells=completed,required_primary_cells=56,
        report_elapsed_hours=wall,measured_stable_capacity='NOT_ESTABLISHED',time=time.time()))
    return verdicts

def publish():
    from .gate import lines
    for path in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(path.read_text())
        if cfg.get('phase') not in ('primary','repeat','protocol_control') or cfg.get('arm')=='PD_G1':continue
        folder=ROOT/'raw'/cfg['name'];status=folder/'status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':continue
        reference=ROOT/'raw'/cfg.get('reference_run',f'primary_{cfg["model"]}_{cfg["workload"].lower()}_pd_g1_start0')
        if not (reference/'requests.jsonl').exists():continue
        a={r['sample_id']:r for r in lines(reference/'requests.jsonl') if not r['warmup']}
        b={r['sample_id']:r for r in lines(folder/'requests.jsonl') if not r['warmup']}
        checks={i:i in a and all(a[i][k]==r[k] for k in ('input_ids','output_ids','exit_depths','finish_reason')) for i,r in b.items()}
        write(ROOT/'state'/f'primary_parity_{cfg["name"]}.json',dict(passed=len(a)==len(b)==cfg['count'] and all(checks.values()),checks=checks,scope='CPU comparison of preserved complete outputs. No GPU rerun.'))
    subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.analyze'],cwd=REPO,check=True)
    reports()
    source=ROOT/'per_request.jsonl'
    if source.exists():
        with source.open('rb') as inp,gzip.open(ROOT/'per_request.jsonl.gz','wb') as out:shutil.copyfileobj(inp,out)
        write(ROOT/'aggregated_request_archive.json',dict(uncompressed_name=source.name,
            uncompressed_SHA256=digest(source),uncompressed_bytes=source.stat().st_size,
            published_archive='per_request.jsonl.gz',scope='Lossless gzip archive of the complete aggregate. Per-run JSONL is also published.'))
    subprocess.run([str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.index_guard'],cwd=REPO,check=True)
    from .index_guard import OWNED
    paths=[]
    candidates=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--',*OWNED],cwd=REPO).split(b'\0')
    for value in candidates:
        if not value:continue
        name=value.decode()
        if '__pycache__' in name or name.endswith('.pyc') or name.endswith('/burner.log') or name.endswith('/logs/controller.log') or name.endswith('/logs/auto_publication.log') or name.endswith('/per_request.jsonl'):continue
        if (REPO/name).is_file() and (REPO/name).stat().st_size>95*1024**2:raise RuntimeError('Oversized artifact requires explicit lossless packaging: '+name)
        if (REPO/name).is_file():paths.append(name)
    paths=sorted(set(paths))
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!='flashvep/adaptive-loop-token-pd-2gpu-w1-w4':raise RuntimeError('Unexpected branch. No commit or push was attempted.')
    subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
    subprocess.run(['git','-c','commit.gpgsign=false','commit','--only','-m','Publish measured two-GPU loop/token PD serving evidence','--',*paths],cwd=REPO,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    subprocess.run(['git','push','-u','origin',branch],cwd=REPO,check=True,timeout=120)
    remote=subprocess.check_output(['git','ls-remote','--heads','origin',branch],cwd=REPO,text=True,timeout=30).split()[0]
    if remote!=sha:raise RuntimeError('Remote SHA differs from the publication commit')
    write(ROOT/'publication_verification.json',dict(published_data_commit=sha,verified_remote_sha=remote,
        branch=branch,time=time.time(),elapsed_hours_before_confirmation_commit=(time.time()-START_EPOCH)/3600))
    path=str((ROOT/'publication_verification.json').relative_to(REPO))
    subprocess.run(['git','add','--',path],cwd=REPO,check=True)
    subprocess.run(['git','-c','commit.gpgsign=false','commit','--only','-m','Record verified PD serving data publication','--',path],cwd=REPO,check=True)
    subprocess.run(['git','push','origin',branch],cwd=REPO,check=True,timeout=120)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','--heads','origin',branch],cwd=REPO,text=True,timeout=30).split()[0]
    if remote!=sha:raise RuntimeError('Final metadata commit was not confirmed remotely')
    print('FINAL_REMOTE_SHA_VERIFIED',sha,flush=True)

if __name__=='__main__':publish()
