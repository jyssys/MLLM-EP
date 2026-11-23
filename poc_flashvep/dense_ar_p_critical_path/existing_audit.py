"""Reconstruct old credit and transport boundaries before choosing a fix."""
import collections
import json
import time
import numpy as np
from .common import ROOT,PRIOR,initialize,write,digest


def run():
    initialize();rows=[];summary=[]
    for label in ('Layer','StockToken','Hybrid'):
        for start in range(3):
            name=f'V2_C3_Heldout192Replay_q2.0_s{start}_{label}';folder=PRIOR/'raw'/name
            cfg=json.loads((folder/'configuration.json').read_text())
            requests=list(map(json.loads,(folder/'requests.jsonl').read_text().splitlines()))
            sends={}
            for event in json.loads((folder/'NIXL_completion_prefill.json').read_text())['rows']:
                for rid in event['finished_sending']:sends.setdefault(rid,event['time'])
            ack={}
            for path in (folder/'hidden_transport').glob('*.ACK.json'):
                value=json.loads(path.read_text());ack[value['request_id']]=value['time']
            by_external={rid.split('-')[0]:rid for rid in sends}
            for request in requests:
                key='q'+str(request['sample_id']);rid=by_external.get(key)
                kv=sends.get(rid);hidden=ack.get(rid)
                available=max(kv,hidden) if kv is not None and hidden is not None else None
                rows.append(dict(label=label,start=start,request_id=key,prompt_id=request['prompt_id'],
                    input_tokens=request['input_tokens'],P_source_credit=cfg.get('continuous_source_credit_limit',0),
                    layer_cut=cfg.get('axis_cut'),KV_sending_notification_time=kv,hidden_READ_ACK_time=hidden,
                    first_client_time=request['first_token'],
                    joint_notification_to_first_client_ms=1000*(request['first_token']-available) if available is not None else None,
                    scope='Historical notification timestamps. The NIXL observer logs before scheduler retirement. '
                        'This is an opportunity proxy,not proof that a source slot was already reclaimed or a measured speedup.'))
            selected=[r['joint_notification_to_first_client_ms'] for r in rows if r['label']==label and r['start']==start and r['joint_notification_to_first_client_ms'] is not None]
            summary.append(dict(label=label,start=start,count=len(selected),
                mean_ms=float(np.mean(selected)) if selected else None,p95_ms=float(np.percentile(selected,95)) if selected else None,
                positive_count=sum(v>0 for v in selected),max_ms=max(selected,default=None)))
    structural=[]
    for label,m,fraction in (('Layer2',2,0),('StockToken8',0,8/36),('Hybrid2_f0125',2,.125)):
        p=(1-fraction)*(36-m);d=36-p
        attention_p=(36-m)*(1-fraction)**2
        structural.append(dict(label=label,P_GEMM_rows_per_L=p,D_GEMM_rows_per_L=d,
            P_attention_pair_coefficient_long_L_approx=attention_p,
            scope='STRUCTURAL_ACCOUNTING. Omits integer alignment and finite-L linear terms. Not GPU time.'))
    result=dict(time=time.time(),evidence='TRACE_ANALYSIS',requests=rows,summary=summary,structural=structural,
        published_host_path_SHA256=digest(PRIOR/'analysis/c3_heldout_replay_v2_host_path_v2.json'),
        scope='Reuse immutable published traces. Existing best Layer2,StockToken8 and Hybrid2/f1/8 are not iso-work. '
            'Keep best-vs-best serving results and matched-work mechanism experiments separate.')
    write(ROOT/'analysis/existing_credit_and_work_audit.json',result)
    from ..dense_ar_output_aware_v2.c0_analysis import csv_save
    csv_save(ROOT/'figures/existing_credit_completion_proxy.csv',rows)
    print('HISTORICAL_CREDIT_NOTIFICATION_PROXY',summary,flush=True)
    print('BEST_CONFIGURATIONS_ARE_NOT_ISOWORK',structural,flush=True)
    regression=json.loads((PRIOR/'analysis/c3_heldout_replay_v2_analysis.json').read_text())
    ratios=[r['ratios']['TTFT_p95_ms'] for r in regression['comparisons'] if r['QPS']==2 and r['right']=='StockToken' and r['left'] in ('Layer','Hybrid')]
    assert len(ratios)==2 and all(r['CI95'][0]>1 for r in ratios)
    print('PUBLISHED_STRONG_C3_REGRESSION_REPRODUCED_FROM_RAW_START_TABLES',ratios,flush=True)


if __name__=='__main__':run()
