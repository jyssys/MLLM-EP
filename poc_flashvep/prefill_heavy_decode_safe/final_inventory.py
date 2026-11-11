"""Collect completed cells without promoting diagnostic traces to clean results."""
import csv,json,time
from .common import *
from .analyze import native,csv_write
from .host_screen import screen


def observer_results():
    rows=[]
    for p in sorted((ROOT/'analysis').glob('*results.json')):
        value=json.loads(p.read_text());groups=value if isinstance(value,list) else value.get('observers',[])
        for r in groups:
            if not isinstance(r,dict) or not r.get('trace'):continue
            name=r.get('name') or r.get('names',[None,None])[1]
            aa=value.get('Native_AA') if isinstance(value,dict) else r.get('Native_AA')
            rows.append(dict(name=name,source=str(p.relative_to(ROOT)),qualified=bool(r.get('qualified',False) and aa and aa.get('passed')),
                status=r.get('status'),trace=r['trace'],Native_AA=value.get('Native_AA') if isinstance(value,dict) else r.get('Native_AA'),
                whole_run_screens=r.get('whole_run_screens'),local_screens=r.get('local_screens')))
        if isinstance(value,dict) and 'trace' in value and 'qualified' in value and value.get('plan'):
            plan=json.loads(Path(value['plan']).read_text());name=plan.get('capture') or plan['jobs'][0]
            rows.append(dict(name=name,source=str(p.relative_to(ROOT)),qualified=value['qualified'],
                status=value['status'],trace=value['trace'],Native_AA=value.get('Native_AA'),
                whole_run_screens=value.get('whole_run_screens'),local_screens=value.get('local_screens'),
                scope='Conservative graph-envelope bound, not an exact D kernel union.'))
    return rows


def main():
    initialize();assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'Finish GPU timing before bulk analysis'
    runs=[];serving=[];qual=[];costs=[];errors=[]
    for p in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(p.read_text())
        if not isinstance(cfg,dict) or cfg.get('name')!=p.stem or p.stem.startswith('priority_'):continue
        name=p.stem;folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        result=json.loads(receipt.read_text()) if receipt.exists() else {}
        state='COMPLETE' if result.get('complete') else result.get('status','NOT_RUN')
        if state=='RUNNING':state='INTERRUPTED_NO_COMPLETION'
        if folder.exists() and not receipt.exists():state='NO_COMPLETED_DISPATCH_RECEIPT'
        row=dict(name=name,model=cfg['model'],length=cfg['length'],cap=cfg['cap'],count=cfg.get('count'),
            QPS=cfg.get('arrival_rate'),P_slots=cfg.get('P_slots'),D_slots=cfg.get('D_slots'),
            P_blocks=cfg.get('kv_blocks'),D_blocks=cfg.get('D_kv_blocks'),gpus=cfg.get('gpus'),
            arrival_seed=cfg.get('arrival_seed'),phase=cfg.get('phase'),status=state,
            config_SHA256=digest(p),instrumented=bool(cfg.get('discovery_activity') or cfg.get('discovery_event_observer') or cfg.get('observe_device_execution')),
            distinct_prompt_ids=len(set(cfg.get('prompt_ids',[]))),
            selection_reason=cfg.get('selection_reason',cfg.get('observation_selection_reason')))
        row['evidence_origin']='REUSED_PREDECESSOR_REFERENCE' if result.get('time',START)>=0 and result.get('time',START)<START else 'CURRENT_CAMPAIGN'
        runs.append(row)
        if state!='COMPLETE' or row['evidence_origin']!='CURRENT_CAMPAIGN':continue
        proof=folder/'qualification_with_frozen_reference.json'
        if cfg.get('state_probe') and proof.exists():qual.append(dict(row,qualification=json.loads(proof.read_text())))
        if cfg.get('calibration_serial') and cfg.get('priority_serial_comparison'):
            from .serial_analysis import analyze as serial
            try:
                value=serial(name)
                if isinstance(value,tuple):costs.append(value[0])
            except Exception as e:errors.append(dict(name=name,operation='serial_analysis',error=repr(e)))
        if cfg.get('count',0)>=64 and not cfg.get('state_probe') and not cfg.get('calibration_serial') and (folder/'requests.jsonl').exists():
            try:
                metrics=native(name)
                host=screen(name,output_path=ROOT/'analysis'/('final_host_'+name+'.json'))
                serving.append(dict(configuration=row,metrics=metrics,host=host))
            except Exception as e:errors.append(dict(name=name,operation='serving_analysis',error=repr(e)))
    observations=observer_results()
    write(ROOT/'analysis/final_campaign_inventory.json',dict(time=time.time(),runs=runs,serving=serving,
        state_qualifications=qual,serial_costs=costs,observers=observations,analysis_errors=errors,
        scope='Completed correctness, serial enclosures, marker-free Serving, and instrumented diagnostics remain separate. NOT_RUN arms are not zero-valued measurements.'))
    csv_write(ROOT/'analysis/campaign_run_status.csv',runs)
    fields=('name','model','length','cap','count','QPS','P_slots','D_slots','arrival_seed','status','instrumented','distinct_prompt_ids')
    metrics=('TTFT_p50_ms','TTFT_p95_ms','TTFT_p99_ms','TPOT_p50_ms','TPOT_p95_ms','TPOT_p99_ms',
        'ITL_p50_ms','ITL_p95_ms','ITL_p99_ms','maximum_ITL_ms','E2E_p50_ms','E2E_p95_ms','E2E_p99_ms',
        'completed_QPS','input_tokens_s','output_tokens_s','SLO_goodput','TTFT_attainment','TPOT_attainment','ITL_limit_misses')
    csv_write(ROOT/'analysis/serving_results.csv',[
        dict({k:r['configuration'].get(k) for k in fields},**{k:r['metrics'].get(k) for k in metrics},
            Service_TTFT='NOT_MEASURED',exact_request_queue_latency='NOT_MEASURED',
            audited_P_queue_growth=r['host']['queue']['half_growth'],audited_P_queue_max=r['host']['queue']['maximum'],
            P_service_proximity=r['host']['P_service_fraction'],
            D_credit_refusal_fraction=r['host']['D_credit_refusal_snapshot_fraction']) for r in serving])
    print(json.dumps(dict(runs=len(runs),serving_cells=len(serving),state_cells=len(qual),serial_cells=len(costs),analysis_errors=errors)))


if __name__=='__main__':main()
