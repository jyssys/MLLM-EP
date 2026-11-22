"""Keep additive host paths and source-window occupancy separate from GPU cost."""
import argparse
import collections
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .statistics import paired_difference
from .c0_analysis import csv_save

STAGES=('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms','P_prepare_to_ready_ms',
    'Ready_to_D_submit_ms','D_submit_to_accept_ms','D_accept_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms')

def credit(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    limit=cfg.get('source_credit_limit',4) if cfg.get('controlled_pipeline') else cfg.get('continuous_source_credit_limit',0)
    path=folder/'credit_events.jsonl';events=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if not limit:return dict(limit=0,status='UNBOUNDED_COORDINATOR_WINDOW',occupancy_time_weighted_mean=None,
        full_fraction=None,scope='This does not remove worker KV,hidden-pool or scheduler admission limits.')
    window=json.loads((folder/'window.json').read_text());left=window['start'];right=window['end']
    pending=set();acquired={};durations=[];full=0.;area=0.;cursor=left
    for event in sorted(events,key=lambda r:r['time']):
        point=event['time'];assert left<=point<=right
        area+=len(pending)*(point-cursor)
        if len(pending)>=limit:full+=point-cursor
        assert event['owned_before']==len(pending)
        rid=event['request_id']
        if event['event']=='ACQUIRE_P_DISPATCH':
            assert rid not in pending;pending.add(rid);acquired[rid]=point
        else:
            assert event['event']=='RELEASE_FIRST_CLIENT_TOKEN' and rid in pending
            pending.remove(rid);durations.append(1000*(point-acquired[rid]))
        assert len(pending)<=limit;cursor=point
    area+=len(pending)*(right-cursor)
    if len(pending)>=limit:full+=right-cursor
    assert not pending and len(acquired)==cfg['count']
    result=dict(limit=limit,status='BOUNDED_COORDINATOR_WINDOW',occupancy_time_weighted_mean=area/(right-left),
        full_fraction=full/(right-left),acquire_to_first_client_mean_ms=float(np.mean(durations)),
        scope='Credit is acquired at coordinator P dispatch and returned at the first client token. '
            'It is not GPU batch size,KV credit or a hidden-buffer slot. Full occupancy alone does not identify the cause of a queue.')
    path=folder/'P_wait_segments.jsonl'
    if path.exists():
        totals=collections.Counter();request_ms=collections.Counter()
        for segment in map(json.loads,path.read_text().splitlines()):
            active=segment['P_active'] is not None;blocked=segment['source_owned']>=limit
            state=f'P_outstanding_{active}_credit_full_{blocked}'
            amount=1000*(segment['end']-segment['start'])
            totals[state]+=amount*len(segment['pending'])
            for index in segment['pending']:request_ms['q'+str(index)]+=amount
        result.update(observed_pending_request_ms_by_joint_state=dict(totals),per_request_observed_pending_ms=dict(request_ms),
            pending_scope='Coordinator pending intervals. Joint states can overlap mechanisms. No P outstanding is not a physical GPU-idle measurement.')
    return result

def analyze(plan_name):
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.service_accounting import analyze as service
    plan=json.loads((ROOT/'jobs'/plan_name).read_text());starts=[];grouped=collections.defaultdict(list);cache={}
    for block in plan['blocks']:
        arms=block.get('arms') or {k:block[k] for k in ('Native','Layer','Token')}
        for label,name in arms.items():
            if label in ('Agg','Deflection'):continue
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
            if name not in cache:
                path=ROOT/'analysis'/(name+'_service_accounting.json')
                cache[name]=(json.loads(path.read_text()) if path.exists() else service(name),credit(name))
            value,window=cache[name]
            quality=None
            if 'c3_' in plan_name:
                from .c3_analysis import actual_quality
                quality=actual_quality(name)['passed']
            assert abs(sum(value['means'][s] for s in STAGES)-value['means']['TTFT_ms'])<1e-3
            row=dict(name=name,label=label,QPS=block.get('qps'),split=block.get('split'),start=block['start'],
                input_bucket=block.get('input_bucket'),output=block.get('output'),cut=block.get('cut'),
                evidence='TRACE_ANALYSIS' if quality is not False else 'DIAGNOSTIC_QUALITY_FAILURE',
                quality_passed=quality,means=value['means'],source_window=window)
            starts.append(row);grouped[(block.get('split'),block.get('qps'),block.get('input_bucket'),block.get('output'),block.get('cut'),block['start'])].append(row)
    comparisons=collections.defaultdict(list)
    for key,rows in grouped.items():
        arms={r['label']:r for r in rows}
        token='StockToken' if 'StockToken' in arms else 'Token'
        for left,right in (('Layer',token),('Hybrid',token),('Hybrid','Layer')):
            if left not in arms or right not in arms:continue
            comparisons[key[:-1]+(left,right)].append((arms[left],arms[right]))
    values=[]
    for key,rows in comparisons.items():
        split,qps,input_bucket,output,cut,left,right=key
        differences={field:paired_difference([a['means'][field] for a,b in rows],[b['means'][field] for a,b in rows])
            for field in (*STAGES,'TTFT_ms','D_decode_residency_ms','D_submit_to_finish_ms')}
        assert abs(sum(differences[s]['estimate'] for s in STAGES)-differences['TTFT_ms']['estimate'])<1e-3
        values.append(dict(split=split,QPS=qps,input_bucket=input_bucket,output=output,cut=cut,left=left,right=right,
            paired_starts=len(rows),mean_host_delta_ms=differences,evidence='TRACE_ANALYSIS'))
    stem=plan_name.removesuffix('.json')+'_host_path_v2'
    result=dict(time=time.time(),plan_SHA256=digest(ROOT/'jobs'/plan_name),starts=starts,comparisons=values,
        scope='Exact additive mean client-TTFT host boundaries. This does not decompose TTFT p95. '
            'Prepare is a host boundary,not a GPU kernel start. C3 service enclosures can include worker queues and interleaved requests. '
            'Source-window occupancy is separate from GPU batch,KV credit and hidden-pool state. '
            'A stage delta locates latency. It does not isolate a GPU operator or prove one admission mechanism. '
            'A C0 Native run can appear under several cuts. Reused rows are not additional independent server starts.')
    write(ROOT/'analysis'/f'{stem}.json',result)
    csv_save(ROOT/'figures'/f'{stem}_starts.csv',starts);csv_save(ROOT/'figures'/f'{stem}_comparisons.csv',values)
    if 'c3_' in plan_name:figures(result,stem)
    print('ADDITIVE_HOST_AND_CREDIT_TABLES',len(starts),len(values),flush=True)

def figures(result,stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=(('P queue/host before prepare',STAGES[:3]),('P prepare→ready host enclosure',STAGES[3:4]),
        ('Ready→D submit',STAGES[4:5]),('D admission/remote-KV host wait',STAGES[5:7]),
        ('D prepare→first worker host enclosure',STAGES[7:8]),('Worker→client',STAGES[8:9]))
    for split in sorted({r['split'] for r in result['starts']}):
        for qps in sorted({r['QPS'] for r in result['starts'] if r['split']==split}):
            rows=[r for r in result['starts'] if r['split']==split and r['QPS']==qps and r['label'] in ('NativeStrong','Layer','StockToken','Hybrid')]
            labels=[x for x in ('NativeStrong','Layer','StockToken','Hybrid') if any(r['label']==x for r in rows)]
            qualified={label:all(r['quality_passed'] for r in rows if r['label']==label) for label in labels}
            fig,ax=plt.subplots(figsize=(10,5.5));bottom=np.zeros(len(labels));table=[]
            for group,fields in groups:
                values=[]
                for label in labels:
                    starts=[sum(r['means'][f] for f in fields) for r in rows if r['label']==label]
                    value=float(np.mean(starts));values.append(value)
                    table.append(dict(split=split,QPS=qps,label=label,host_stage=group,mean_ms=value,independent_starts=len(starts),quality_passed=qualified[label],
                        scope='Additive mean client TTFT. Not TTFT p95 or exclusive GPU compute.'))
                bars=ax.bar(labels,values,bottom=bottom,label=group)
                for bar,label in zip(bars,labels):
                    if not qualified[label]:bar.set_hatch('//')
                bottom+=values
            ax.set(ylabel='Mean client TTFT host path (ms)',title=f'C3 {split} | QPS{qps} | additive host boundaries')
            ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
            fig.text(.02,.015,'TRACE_ANALYSIS. Hatching: quality failure. Input preparation is a host boundary. Enclosures can include scheduling,transfer and other requests.\n'
                'These stages sum to mean TTFT. They do not decompose TTFT p95 or identify operator-level causes.',fontsize=8)
            fig.tight_layout(rect=(0,.08,1,1));target=ROOT/'figures'/f'{stem}_{split}_q{qps}'
            for ext in ('png','pdf'):fig.savefig(str(target)+'.'+ext,dpi=170)
            csv_save(ROOT/'figures'/f'{stem}_{split}_q{qps}.csv',table);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='c3_main_serving_v2.json');args=p.parse_args();analyze(args.plan)
