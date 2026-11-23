"""Recalculate published host stages without editing historical evidence."""
import json
import time
import numpy as np
from .common import ROOT,PRIOR,write,digest,bind,preserve
from .critical_analysis import records,csv_save


def analyze():
    bind();preserve()
    from ..dense_ar_longbench400_pd import service_accounting
    from ..dense_ar_output_aware_v2.statistics import ratio,paired_difference
    old_root,old_write=service_accounting.ROOT,service_accounting.write
    starts=[];requests=[];sources={};configs={};outputs={}
    try:
        service_accounting.ROOT=PRIOR
        service_accounting.write=lambda path,value:write(ROOT/'analysis'/('historical_'+path.name),value)
        for start in range(3):
            for label in ('Layer','StockToken','Hybrid'):
                name=f'V2_C3_Heldout192Replay_q2.0_s{start}_{label}';folder=PRIOR/'raw'/name
                cfg=json.loads((folder/'configuration.json').read_text());configs[start,label]=cfg
                raw=records(folder/'requests.jsonl');outputs[start,label]=raw
                assert len(raw)==192 and cfg['qps']==2 and cfg['ignore_eos']
                assert len({row['prompt_id'] for row in raw})==192
                assert all(abs(row['TTFT_ms']-1000*(row['first_token']-row['arrival']))<1e-3 for row in raw)
                path=service_accounting.analyze(name)
                starts.append(dict(start=start,label=label,name=name,QPS=cfg['qps'],seed=cfg['seed'],GPU_pair=cfg['gpus'],
                    dataset_SHA256=cfg['dataset_SHA256'],source_archive_SHA256=digest(folder/'source.zip'),
                    output_tokens=sum(row['output_tokens'] for row in raw),TTFT_p95_ms=float(np.percentile([row['TTFT_ms'] for row in raw],95)),
                    mean_host_path_ms=path['means']))
                requests.extend(dict(start=start,label=label,**row) for row in path['rows'])
                for filename in ('configuration.json','requests.jsonl','request_submissions.jsonl','admissions.jsonl',
                    'handoffs.jsonl','actual_queries_prefill.json','actual_queries_decode.json'):
                    sources[str((folder/filename).relative_to(ROOT.parent))]=digest(folder/filename)
    finally:
        service_accounting.ROOT,service_accounting.write=old_root,old_write
    comparisons=[]
    for label in ('Layer','Hybrid'):
        paired=[]
        for start in range(3):
            a,b=(configs[start,key] for key in (label,'StockToken'))
            for key in ('dataset_SHA256','qps','seed','gpus','arrival_offsets','output_cap','ignore_eos'):
                assert a[key]==b[key],(start,label,key)
            def work(rows):return {row['prompt_id']:(row['input_ids_SHA256'],row['output_tokens']) for row in rows}
            assert work(outputs[start,label])==work(outputs[start,'StockToken'])
            paired.append(tuple(next(row for row in starts if row['start']==start and row['label']==key) for key in (label,'StockToken')))
        P_keys=('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms','P_prepare_to_ready_ms')
        post_keys=('Ready_to_D_submit_ms','D_submit_to_accept_ms','D_accept_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms')
        comparisons.append(dict(left=label,right='StockToken',independent_starts=3,
            TTFT_p95_ratio=ratio([a['TTFT_p95_ms'] for a,b in paired],[b['TTFT_p95_ms'] for a,b in paired]),
            P_path_delta_ms=paired_difference([sum(a['mean_host_path_ms'][key] for key in P_keys) for a,b in paired],
                [sum(b['mean_host_path_ms'][key] for key in P_keys) for a,b in paired]),
            post_P_path_delta_ms=paired_difference([sum(a['mean_host_path_ms'][key] for key in post_keys) for a,b in paired],
                [sum(b['mean_host_path_ms'][key] for key in post_keys) for a,b in paired]),
            mean_host_delta_ms={key:paired_difference([a['mean_host_path_ms'][key] for a,b in paired],
                [b['mean_host_path_ms'][key] for a,b in paired]) for key in paired[0][0]['mean_host_path_ms']}))
    published=PRIOR/'analysis/c3_heldout_replay_v2_analysis.json';source=json.loads(published.read_text())
    for row in comparisons:
        original=next(value for value in source['comparisons'] if value['QPS']==2 and value['left']==row['left'] and value['right']=='StockToken')
        assert abs(row['TTFT_p95_ratio']['estimate']-original['ratios']['TTFT_p95_ms']['estimate'])<1e-9
    bridges=[]
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    plan=ROOT/'jobs/baseline_reproduction.json';bridge_plan=json.loads(plan.read_text())
    for item in bridge_plan['arms']:
        label,name=item['label'],item['name'];folder=ROOT/'raw'/name
        if not (folder/'dispatcher_result.json').exists():continue
        assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
        cfg=json.loads((folder/'configuration.json').read_text());metric=summarize(name)
        observed=service_accounting.analyze(name)
        bridges.append(dict(label=label,name=name,independent_starts=1,QPS=cfg['qps'],seed=cfg['seed'],GPU_pair=cfg['gpus'],
            source_archive_SHA256=digest(folder/'source.zip'),dataset_SHA256=cfg['dataset_SHA256'],
            TTFT_p95_ms=metric['TTFT_p95_ms'],ITL_p99_ms=metric['ITL_p99_ms'],TPOT_p95_ms=metric['TPOT_p95_ms'],
            mean_host_path_ms=observed['means'],actual_geometry=metric['actual_geometry'],
            evidence='DIAGNOSTIC_N1_REPRODUCTION_BRIDGE',
            scope='New N1 timing with inherited baseline lineage. Do not count it as a new N3 reproduction or merge it with the primary confirmation.'))
    result=dict(time=time.time(),passed=True,starts=starts,comparisons=comparisons,new_R0_bridges=bridges,source_files_SHA256=sources,
        published_analysis_SHA256=digest(published),
        evidence='HISTORICAL_RAW_RECALCULATION',scope='Read the original nine published runs. Recompute host boundaries and client p95. '
            'Require identical request hashes,arrival offsets and output work within each paired start. '
            'Write only new-namespace derived files. This is not nine new GPU runs or a new independent confirmation. '
            'The new R0 bridge is N1. The new same-pair CPC and primary confirmation are separate N3 families. '
            'Historical host differences are not an exclusive GPU/host causal decomposition.')
    write(ROOT/'analysis/historical_stage_recalculation.json',result)
    csv_save(ROOT/'figures/historical_stage_per_start.csv',starts)
    csv_save(ROOT/'figures/historical_stage_per_request.csv',requests)
    csv_save(ROOT/'figures/historical_stage_paired_comparisons.csv',comparisons)
    csv_save(ROOT/'figures/new_R0_reproduction_bridges.csv',bridges)
    preserve();print('HISTORICAL_N3_RAW_RECALCULATION_PASS',flush=True);return result


if __name__=='__main__':analyze()
