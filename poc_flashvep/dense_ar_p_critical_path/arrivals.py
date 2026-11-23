"""Audit the actual arrival trace without attributing queue delay to bursts."""
import bisect
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .critical_analysis import csv_save


def preceding_counts(offsets,width):
    """Count arrivals in the half-open interval (t-width,t],including this arrival."""
    assert width>0 and all(a<=b for a,b in zip(offsets,offsets[1:]))
    return [i+1-bisect.bisect_right(offsets,t-width,0,i+1) for i,t in enumerate(offsets)]


def analyze():
    source=ROOT/'analysis/c3_credit_confirmation_analysis.json'
    data=json.loads(source.read_text());summaries=[];requests=[];pairs=[]
    for mode in dict.fromkeys(row['mode'] for row in data['starts']):
        for start in sorted({row['start'] for row in data['starts'] if row['mode']==mode}):
            arms=[row for row in data['starts'] if row['mode']==mode and row['start']==start]
            if not arms:continue
            configs=[json.loads((ROOT/'configs'/(row['name']+'.json')).read_text()) for row in arms]
            offsets=configs[0]['arrival_offsets'];count=len(offsets)
            assert count>1 and all(cfg['arrival_offsets']==offsets and cfg['seed']==configs[0]['seed']
                and cfg['qps']==configs[0]['qps'] and cfg['dataset_SHA256']==configs[0]['dataset_SHA256'] for cfg in configs)
            intervals=np.diff(offsets);span=offsets[-1]-offsets[0]
            windows={width:preceding_counts(offsets,width) for width in (.1,.5,1.,2.)}
            summaries.append(dict(mode=mode,start=start,seed=configs[0]['seed'],offered_QPS=configs[0]['qps'],requests=count,
                scheduled_span_seconds=span,realized_interarrival_rate=(count-1)/span,
                mean_interarrival_ms=1000*float(np.mean(intervals)),minimum_interarrival_ms=1000*float(np.min(intervals)),
                interarrival_p10_p50_p90_ms=(1000*np.percentile(intervals,[10,50,90])).tolist(),
                interarrival_coefficient_of_variation=float(np.std(intervals)/np.mean(intervals)),
                maximum_arrivals_in_preceding_window={str(width):max(values) for width,values in windows.items()},
                identical_scheduled_trace_across_arms=True,arms=[row['label'] for row in arms]))
            by_label={}
            for arm in arms:
                path=ROOT/'analysis'/(arm['name']+'_service_accounting.json')
                observed=json.loads(path.read_text());assert len(observed['rows'])==count
                values={row['request_id']:row for row in observed['rows']};by_label[arm['label']]=values
                for i,offset in enumerate(offsets):
                    row=values['q'+str(i)]
                    requests.append(dict(mode=mode,start=start,label=arm['label'],name=arm['name'],request_id='q'+str(i),
                        prompt_id=row['prompt_id'],scheduled_offset_seconds=offset,input_tokens=row['input_tokens'],
                        output_tokens=row['output_tokens'],TTFT_ms=row['TTFT_ms'],
                        Arrival_to_P_submit_ms=row['Arrival_to_P_submit_ms'],
                        Arrival_to_P_ready_ms=sum(row[key] for key in ('Arrival_to_P_submit_ms','P_submit_to_accept_ms',
                            'P_accept_to_prepare_ms','P_prepare_to_ready_ms')),
                        **{'arrivals_in_preceding_'+str(width)+'s':values[i] for width,values in windows.items()}))
            if 'OriginalLayer' in by_label and 'StockToken' in by_label:
                for i in range(count):
                    a,b=(by_label[label]['q'+str(i)] for label in ('OriginalLayer','StockToken'))
                    assert a['prompt_id']==b['prompt_id']
                    pairs.append(dict(mode=mode,start=start,request_id='q'+str(i),prompt_id=a['prompt_id'],
                        scheduled_offset_seconds=offsets[i],arrivals_in_preceding_1s=windows[1.][i],
                        Layer_minus_Token_Arrival_to_P_submit_ms=a['Arrival_to_P_submit_ms']-b['Arrival_to_P_submit_ms'],
                        Layer_minus_Token_TTFT_ms=a['TTFT_ms']-b['TTFT_ms'],
                        identical_request_output_work=a['output_tokens']==b['output_tokens']))
    result=dict(time=time.time(),source_SHA256=digest(source),traces=summaries,
        evidence='TRACE_ANALYSIS',scope='Offered QPS is a mean arrival rate,not simultaneous batch size. '
            'Realized rate uses (N-1)/(last-first scheduled arrival). Window counts include the current arrival. '
            'Identical scheduled offsets do not guarantee identical coordinator enqueue delays. '
            'Per-request queue/burst associations are descriptive. There is no intervention on burstiness or proof of a causal percentage. '
            'Natural output work can differ. A QPS2 finite cohort does not establish sustainable capacity.')
    write(ROOT/'analysis/arrival_trace_audit.json',result)
    csv_save(ROOT/'figures/arrival_trace_summary.csv',summaries)
    csv_save(ROOT/'figures/arrival_trace_per_request.csv',requests)
    csv_save(ROOT/'figures/arrival_trace_paired_queue_delta.csv',pairs)
    print('ARRIVAL_TRACE_AUDIT',len(summaries),'traces',len(pairs),'paired requests',flush=True)
    return result


if __name__=='__main__':analyze()
