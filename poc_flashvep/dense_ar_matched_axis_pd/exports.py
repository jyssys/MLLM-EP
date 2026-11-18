"""Export observed requests and the full execution matrix without inference."""
import csv
import hashlib
import json
import statistics
from .common import ROOT, digest, write
from .host_paths import records


def main():
    data=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    inventory=json.loads((ROOT/'analysis/experiment_inventory.json').read_text())
    fields=('name','stage','method','layers','output_cap','offered_QPS','arrival_seed',
        'admission_cap','sample_id','prompt_id','input_ids_SHA256','input_tokens',
        'output_tokens','TTFT_ms','TPOT_ms','E2E_ms','first_ITL_ms','steady_ITL_mean_ms',
        'maximum_ITL_ms','worker_maximum_ITL_ms','arrival','first_token','finish',
        'P_host_submit','source_ready_notice','D_host_submit','D_cohort',
        'D_cohort_size','P_prompt_queries','D_prompt_queries','P_layers','D_prompt_layers',
        'P_GEMM_rows','D_GEMM_rows','token_suffix','source_fragments',
        'Balanced_TTFT_pass','Balanced_Decode_pass','Balanced_Joint_pass',
        'Common_Balanced_TTFT_pass','Common_Balanced_Decode_pass','Common_Balanced_Joint_pass',
        'config_SHA256','evidence')
    count=0
    with (ROOT/'analysis/per_request_metrics_and_work.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for start in data['starts']:
            name=start['name'];folder=ROOT/'raw'/name
            cfg_path=ROOT/'configs'/(name+'.json')
            cfg=json.loads(cfg_path.read_text())
            observed=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
            work={r['request_id']:r for r in observed.get('per_request_P_query_coverage',{}).get('rows',[])}
            source={r['request_id']:r['actual_send_start'] for r in records(folder/'actual_submissions.jsonl')}
            ready={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
            dest={r['request_id']:r['actual_send_start'] for r in records(folder/'D_actual_submissions.jsonl')}
            cohorts={r['request_id']:r for r in records(folder/'D_host_admission.jsonl')}
            for row in records(folder/'requests.jsonl'):
                rid='q'+str(row['sample_id']);geometry=work.get(rid,{})
                events=row['token_events']
                intervals=[1000*(b['time']-a['time'])/(b['count']-a['count'])
                    for a,b in zip(events,events[1:])]
                worker=[1000*(b['worker_time']-a['worker_time'])/(b['count']-a['count'])
                    for a,b in zip(events,events[1:]) if 'worker_time' in a and 'worker_time' in b]
                assert len(row['output_ids'])==row['output_tokens']==cfg['request_caps'][row['sample_id']]
                values=dict(name=name,stage=start['stage'],method=start['method'],layers=start['layers'],
                    output_cap=cfg['cap'],offered_QPS=cfg['arrival_rate'],arrival_seed=cfg['arrival_seed'],
                    admission_cap=cfg['D_admission_batch'],sample_id=row['sample_id'],prompt_id=row['prompt_id'],
                    input_ids_SHA256=hashlib.sha256(json.dumps(row['input_ids'],separators=(',',':')).encode()).hexdigest(),
                    input_tokens=row['input_tokens'],output_tokens=row['output_tokens'],TTFT_ms=row['ttft_ms'],
                    TPOT_ms=row['tpot_ms'],E2E_ms=row['e2e_ms'],first_ITL_ms=intervals[0] if intervals else None,
                    steady_ITL_mean_ms=statistics.mean(intervals[1:]) if len(intervals)>1 else None,
                    maximum_ITL_ms=max(intervals,default=None),worker_maximum_ITL_ms=max(worker,default=None),
                    arrival=row['arrival'],first_token=row['first_token'],finish=row['finish'],
                    P_host_submit=source.get(rid),source_ready_notice=ready.get(rid),D_host_submit=dest.get(rid),
                    D_cohort=cohorts.get(rid,{}).get('cohort'),
                    D_cohort_size=cohorts.get(rid,{}).get('cohort_size'),
                    config_SHA256=digest(cfg_path),evidence=start['evidence'])
                if geometry:
                    values.update(P_prompt_queries=geometry['P_prompt_queries'],D_prompt_queries=geometry['D_prompt_queries'],
                        P_layers=geometry['P_transformer_layers'],D_prompt_layers=geometry['D_prompt_transformer_layers'],
                        token_suffix=geometry['suffix_tokens'],source_fragments=json.dumps(geometry['source_fragments'],separators=(',',':')),
                        P_GEMM_rows=geometry['P_prompt_queries']*geometry['P_transformer_layers'],
                        D_GEMM_rows=geometry['D_prompt_queries']*geometry['D_prompt_transformer_layers'])
                for label,regime in (('Balanced','Balanced'),('Common_Balanced','Common_Balanced_OutputSweep')):
                    limits=observed['SLO_regimes'][regime]['thresholds']
                    ttft=row['ttft_ms']<=limits['TTFT_ms']
                    decode=row['tpot_ms']<=limits['TPOT_ms'] and max(intervals,default=0)<=limits['ITL_ms']
                    values.update({label+'_TTFT_pass':ttft,label+'_Decode_pass':decode,label+'_Joint_pass':ttft and decode})
                writer.writerow(values);count+=1
    matrix_fields=('name','status','method','layers','cap','QPS','count','phase',
        'execution_contract_version','superseded_before_clean_performance')
    with (ROOT/'analysis/full_execution_matrix.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=matrix_fields)
        writer.writeheader()
        for row in inventory['runs']:
            writer.writerow({key:row[key] for key in matrix_fields})
    outputs=['per_request_metrics_and_work.csv','full_execution_matrix.csv','per_start_metrics.csv']
    write(ROOT/'analysis/tabular_export_manifest.json',dict(request_rows=count,starts=len(data['starts']),
        files=[dict(path='analysis/'+name,SHA256=digest(ROOT/'analysis'/name)) for name in outputs],
        scope='Client-visible consecutive-token intervals include the first interval. Worker notices remain separate. Blank Aggregated work cells are not zero. Source query and layer products are structural accounting. Raw token IDs and per-token timestamps remain in each immutable requests.jsonl.'))
    print('EXPORTS',count,len(data['starts']),flush=True)


if __name__=='__main__':
    main()
