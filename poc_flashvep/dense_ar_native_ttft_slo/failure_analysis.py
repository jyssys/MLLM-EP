"""Attribute frozen metric failures without claiming runtime causal speedup."""
import collections
import csv
import itertools
import json
import time
import numpy as np
from .common import ROOT,REPO,initialize,preserve,write,digest
from .reproduction import reconstruct
from .registry import analysis_runs

CATEGORIES=('JOINT_PASS','TTFT_ONLY_FAIL','ITL_ONLY_FAIL','TTFT_ITL_FAIL',
    'TPOT_ONLY_FAIL','TTFT_TPOT_FAIL','ITL_TPOT_FAIL','ALL_FAIL')
GATES=('TTFT','ITL','TPOT')


def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list,tuple)) else value for key,value in row.items()})


def failure_mask(row):
    return sum((not row[gate+'_PASS'])<<index for index,gate in enumerate(GATES))


def classify(rows,seconds):
    assert rows and seconds>0
    n=len(rows);tokens=sum(row['output_tokens'] for row in rows);categories=[];oracles=[];shapley=[]
    for mask,label in enumerate(CATEGORIES):
        part=[row for row in rows if failure_mask(row)==mask]
        value=sum(row['output_tokens'] for row in part)
        categories.append(dict(category=label,mask=mask,request_count=len(part),request_fraction=len(part)/n,
            output_tokens=value,output_token_fraction=value/tokens,output_tokens_per_second=value/seconds,
            output_tokens_mean=float(np.mean([row['output_tokens'] for row in part])) if part else None,
            output_tokens_p95=float(np.percentile([row['output_tokens'] for row in part],95)) if part else None))
    assert sum(row['request_count'] for row in categories)==n
    assert sum(row['output_tokens'] for row in categories)==tokens
    for removed in range(8):
        good=[row for row in rows if failure_mask(row)&(~removed)==0]
        oracles.append(dict(removed_mask=removed,removed_gates=[gate for i,gate in enumerate(GATES) if removed&(1<<i)],
            request_count=len(good),attainment=len(good)/n,qualified_output_tokens=sum(row['output_tokens'] for row in good),
            SLO_goodput=sum(row['output_tokens'] for row in good)/seconds,
            evidence='OFFLINE_METRIC_GATE_COUNTERFACTUAL'))
    for i,gate in enumerate(GATES):
        request_loss=token_loss=0.
        for row in rows:
            mask=failure_mask(row);count=mask.bit_count()
            if mask&(1<<i):request_loss+=1/count;token_loss+=row['output_tokens']/count
        shapley.append(dict(gate=gate,attributed_failed_requests=request_loss,attributed_lost_output_tokens=token_loss,
            attributed_goodput_loss=token_loss/seconds,evidence='OFFLINE_METRIC_FAILURE_SHAPLEY',
            scope='Each failed request divides its metric loss equally across its failed gates. This is the exact three-gate Shapley value,not a runtime causal contribution.'))
    assert abs(sum(row['attributed_failed_requests'] for row in shapley)-(n-oracles[0]['request_count']))<1e-7
    assert abs(sum(row['attributed_lost_output_tokens'] for row in shapley)-(tokens-oracles[0]['qualified_output_tokens']))<1e-7
    return categories,oracles,shapley


def input_bucket(n):
    return next((label for label,a,b in (('3K-6K',3000,6000),('6K-10K',6000,10000),
        ('10K-20K',10000,20000),('20K-36K',20000,36001)) if a<=n<b),'OUTSIDE')


def output_bucket(n):
    return '1-10' if n<=10 else '11-50' if n<=50 else '51-100' if n<=100 else '101-200' if n<=200 else '201+'


def analyze():
    initialize();preserve();starts=[];requests=[];category_rows=[];oracle_rows=[];shapley_rows=[];groups=[];interval_rows=[];sources={}
    for item in analysis_runs():
        folder=item['root']/'raw'/item['name'];cfg,metric,rows=reconstruct(folder)
        meta={key:value for key,value in item.items() if key!='root'}
        runtime={}
        for path in folder.glob('actual_runtime_*.json'):
            raw=json.loads(path.read_text());runtime[raw.get('role',path.stem)]=dict(
                GPU_UUID=raw.get('uuid'),B=raw.get('scheduler_B'),S=raw.get('scheduler_S'),
                compile_mode=raw.get('compile_mode'),cudagraph_mode=raw.get('cudagraph_mode'),
                D_running_admission_cap=raw.get('D_running_admission_cap'),KV_bytes=raw.get('KV_bytes'))
        receipt=json.loads((folder/'dispatcher_result.json').read_text())
        meta.update(seed=cfg['seed'],GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'],
            B=cfg['token_budget'],S=cfg['max_num_seqs'],source_credit=cfg.get('continuous_source_credit_limit',0),
            D_cap=cfg.get('D_running_admission_cap'),KV_block_tokens=cfg.get('KV_block_tokens'),
            source_SHA256=receipt['source_SHA256'],actual_runtime=runtime,
            correctness_only=cfg.get('correctness_only',False),axis_kind=cfg.get('axis_kind'),axis_cut=cfg.get('axis_cut'),
            source_credit_release_on_retirement=cfg.get('source_credit_release_on_retirement',False),
            control_flags={key:cfg[key] for key in cfg if any(word in key.lower() for word in ('defer','compile','hidden','fusion','fuse','admission','credit'))})
        starts.append(dict(meta,**metric))
        for row in rows:
            part=dict(meta,**row,input_bucket=input_bucket(row['input_tokens']),output_bucket=output_bucket(row['output_tokens']),
                category=CATEGORIES[failure_mask(row)],failure_mask=failure_mask(row))
            for index,value in enumerate(part.pop('ITL_intervals_ms')):
                interval_rows.append(dict(family=meta['family'],label=meta['label'],start=meta['start'],name=meta['name'],
                    sample_id=row['sample_id'],interval_index=index,ITL_ms=value))
            requests.append(part)
        categories,oracles,shapley=classify(rows,metric['measurement_seconds'])
        category_rows.extend(dict(meta,**row) for row in categories)
        oracle_rows.extend(dict(meta,**row,attainment_delta=row['attainment']-oracles[0]['attainment'],
            goodput_delta=row['SLO_goodput']-oracles[0]['SLO_goodput']) for row in oracles)
        shapley_rows.extend(dict(meta,**row) for row in shapley)
        for dimension in ('task','input_bucket','output_bucket'):
            values=collections.defaultdict(list)
            for row in rows:
                key=row['task'] if dimension=='task' else input_bucket(row['input_tokens']) if dimension=='input_bucket' else output_bucket(row['output_tokens'])
                values[key].append(row)
            for key,part in sorted(values.items()):
                groups.append(dict(meta,dimension=dimension,bucket=key,request_count=len(part),
                    output_tokens=sum(row['output_tokens'] for row in part),
                    joint_requests=sum(row['JOINT_PASS'] for row in part),
                    joint_output_tokens=sum(row['output_tokens'] for row in part if row['JOINT_PASS']),
                    TTFT_fail_requests=sum(not row['TTFT_PASS'] for row in part),
                    TTFT_fail_tokens=sum(row['output_tokens'] for row in part if not row['TTFT_PASS']),
                    SLO_goodput_contribution=sum(row['output_tokens'] for row in part if row['JOINT_PASS'])/metric['measurement_seconds'],
                    scope='Descriptive stratum in the mixed workload. Token contribution uses the global cohort window.'))
        for filename in ('configuration.json','requests.jsonl','window.json','dispatcher_result.json'):
            path=folder/filename;sources[str(path.relative_to(REPO))]=digest(path)
    summary=[];by_family=collections.defaultdict(list)
    for row in starts:by_family[row['family'],row['label']].append(row)
    numeric=('TTFT_mean_ms','TTFT_p50_ms','TTFT_p90_ms','TTFT_p95_ms','TTFT_p99_ms','mean_TBT_ms','ITL_p95_ms','ITL_p99_ms','max_ITL_ms',
        'TPOT_mean_ms','TPOT_p95_ms','output_tokens','measurement_seconds','generated_output_tokens_per_second',
        'completed_requests_per_second','SLO_qualified_output_tokens_per_second','TTFT_SLO_attainment',
        'every_ITL_SLO_attainment','TPOT_SLO_attainment','Joint_SLO_attainment')
    for (family,label),part in sorted(by_family.items()):
        subset=[row for row in requests if row['family']==family and row['label']==label]
        durations=sum(row['measurement_seconds'] for row in part)
        summary.append(dict(family=family,label=label,independent_starts=len(part),
            start_mean={key:float(np.mean([row[key] for row in part])) for key in numeric if all(row[key] is not None for row in part)},
            pooled_descriptive=dict(requests=len(subset),TTFT_p95_ms=float(np.percentile([row['TTFT_ms'] for row in subset],95)),
                joint_attainment=float(np.mean([row['JOINT_PASS'] for row in subset])),
                output_tokens_per_sum_of_cohort_windows=sum(row['output_tokens'] for row in subset)/durations,
                goodput_per_sum_of_cohort_windows=sum(row['output_tokens'] for row in subset if row['JOINT_PASS'])/durations),
            scope='Start-mean metrics and descriptive pooled values have separate denominators. No pooled request confidence interval.'))
    value=dict(time=time.time(),passed=True,starts=starts,summary=summary,categories=category_rows,gate_oracles=oracle_rows,
        shapley=shapley_rows,strata=groups,source_files_SHA256=sources,
        evidence='RAW_RECALCULATION_WITH_SEPARATE_START_FAMILIES',
        scope='All eight metric failure combinations. Counterfactuals remove metric gates only and do not change queue,GPU or measured cohort duration. '
            'Different workload,numerical/cache and start families stay separate. Native-relative runtime causes need independent interventions.')
    write(ROOT/'analysis/request_failure_attribution.json',value)
    for filename,rows in (('metrics_per_start',starts),('metrics_start_mean_and_pooled',summary),('request_slo_failures',requests),
            ('client_ITL_intervals',interval_rows),('slo_eight_categories',category_rows),('slo_gate_counterfactuals',oracle_rows),
            ('metric_failure_shapley',shapley_rows),('task_input_output_strata',groups)):
        save_csv(ROOT/'figures'/(filename+'.csv'),rows)
    compact_fields=('family','mode','label','start','name','sample_id','prompt_id','task','input_tokens','output_tokens',
        'input_bucket','output_bucket','TTFT_ms','TPOT_ms','maximum_ITL_ms','TTFT_PASS','ITL_PASS','TPOT_PASS',
        'JOINT_PASS','category','failure_mask','finish_reason','cap_hit','seed','GPU_pair','dataset_SHA256','source_SHA256','B','S','source_credit','D_cap')
    save_csv(ROOT/'figures/main400_request_slo_failures.csv',[
        {key:row[key] for key in compact_fields} for row in requests if row['family'] in
        ('Historical_Main400_Natural','Current_Main400_Natural')])
    print('REQUEST_FAILURE_ATTRIBUTION_PASS',len(starts),'starts',len(requests),'requests',len(interval_rows),'ITLs',flush=True)
    preserve();return value


if __name__=='__main__':analyze()
