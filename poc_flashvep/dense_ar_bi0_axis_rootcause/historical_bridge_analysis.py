"""Audit archived winner traces and current-runtime bridge boundaries."""
import collections
import hashlib
import json
import subprocess
import zipfile
import numpy as np
from .common import ROOT,REPO,write,digest
from .historical_bridge import HISTORICAL,SEEDS
from .statistics import interval
from .causal_replay import extract
from .queue_accounting import lines


def overlap(a,b,c,d):return max(0.,min(b,d)-max(a,c))


def external_id(value):return value.split('-')[0] if isinstance(value,str) else None


def archived(axis,seed):
    cut=0 if axis=='Native' else 8
    name=f'heldout_confirmation_{axis}_Last{cut}_N72_O16_q6_s{seed}_B1_W32_v1'
    folder=HISTORICAL/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    host=json.loads((HISTORICAL/'analysis'/(name+'_host_paths.json')).read_text())
    slot=json.loads((HISTORICAL/'analysis'/(name+'_slot_decomposition.json')).read_text())
    audit=json.loads((HISTORICAL/'analysis'/(name+'_service_queue_audit.json')).read_text())
    feedback=json.loads((HISTORICAL/'analysis'/(name+'_reservation_feedback.json')).read_text())
    command=json.loads((folder/'command.json').read_text());provenance=json.loads((folder/'input_provenance.json').read_text())
    assert digest(folder/'source.zip')==command['source_archive_SHA256']
    assert digest(folder/'configuration.json')==provenance['config_SHA256']
    assert digest(HISTORICAL/'data'/cfg['dataset'])==provenance['dataset_SHA256']
    assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    assert json.loads((folder/'worker_exits.json').read_text())['passed']
    assert cfg['ar_batch_invariant'] and cfg['D_admission_batch']==1 and cfg['source_outstanding_limit']==32
    assert host['cohort_sizes']=={'1':72}
    requests={r['request_id']:r for r in host['request_paths']}
    raw={f'q{r["sample_id"]}':r for r in lines(folder/'requests.jsonl')}
    previous=[];parts=[]
    for cohort in host['cohorts']:
        assert len(cohort['requests'])==1
        rid=cohort['requests'][0];a,b=requests[rid]['boundaries_monotonic'][2:4]
        prepare=sum(1000*overlap(a,b,*p['prepare']) for p in previous)
        decode=sum(1000*overlap(a,b,*p['decode']) for p in previous)
        delivery=sum(1000*overlap(a,b,*p['delivery']) for p in previous)
        full=1000*(b-a);residual=full-prepare-decode-delivery
        assert residual>=-1e-5
        parts.append(dict(request_id=rid,ready_to_D_submit_ms=full,
            prior_D_preparation_guard_overlap_ms=prepare,prior_D_decode_residency_guard_overlap_ms=decode,
            prior_delivery_guard_overlap_ms=delivery,other_host_wait_ms=max(0.,residual)))
        first=raw[rid]['token_events'][0]['worker_time'];last=raw[rid]['token_events'][-1]['worker_time']
        previous.append(dict(prepare=(b,first),decode=(first,last),delivery=(last,cohort['last_client_notice'])))
    runtime={}
    for role in ('prefill','decode'):
        value=json.loads((folder/f'effective_runtime_{role}.json').read_text())
        runtime[role]={k:v for k,v in value.items() if k!='KV_tensors'}
        runtime[role]['first_KV_tensor']=value['KV_tensors'][0]
    # Compare archived executable files to the published commit, not the current dirty index.
    selected=('poc_flashvep/dense_ar_matched_axis_pd/serving.py',
        'poc_flashvep/dense_ar_matched_axis_pd/site_bootstrap.py',
        'poc_flashvep/dense_ar_slo_layer_pd/native.py',
        'poc_flashvep/dense_ar_slo_layer_pd/dynamic_native.py',
        'poc_flashvep/dense_ar_layer_token_pd/token_axis.py')
    hashes=[]
    with zipfile.ZipFile(folder/'source.zip') as bundle:
        for path in selected:
            frozen=bundle.read(path)
            published=subprocess.check_output(['git','show','4595080c:'+path],cwd=REPO)
            hashes.append(dict(path=path,frozen_SHA256=hashlib.sha256(frozen).hexdigest(),
                published_SHA256=hashlib.sha256(published).hexdigest(),equals_published=frozen==published))
    assert all(r['equals_published'] for r in hashes)
    return dict(name=name,axis=axis,seed=seed,N=72,config=cfg,runtime=runtime,
        provenance=provenance,source_archive_SHA256=digest(folder/'source.zip'),executable_source=hashes,
        mean_stages_ms={k:v['mean'] for k,v in host['summaries'].items()},
        mean_D_host_parts_ms=slot['mean_D_per_request_ms'],
        mean_guard_wait_parts_ms={k:float(np.mean([r[k] for r in parts])) for k in parts[0] if k!='request_id'},
        request_guard_parts=parts,reservation_rejections=audit['reservation_rejections'],
        credit_retries=audit['common_credit_retries'],reservation_feedback=feedback,
        metrics=audit['metric_summary'],SLO=audit['common_Balanced'],
        evidence='HISTORICAL_MEASURED_ONLINE_REUSED',physical_GPU_service='NOT_MEASURED')


def current(row):
    value=extract(row['name']);folder=ROOT/'raw'/row['name']
    handoff={r['request_id']:r for r in lines(folder/'handoffs.jsonl')}
    submissions={(r['role'],r['request_id']):r['time'] for r in lines(folder/'request_submissions.jsonl')}
    offered={r['request_id']:r for r in lines(folder/'offered_requests.jsonl')}
    requests=sorted(lines(folder/'requests.jsonl'),key=lambda r:r['sample_id'])
    admission={(r['role'],r['request_id']):r['worker_accepted'] for r in lines(folder/'admissions.jsonl')}
    query=json.loads((folder/'actual_queries_decode.json').read_text())['rows']
    transport=json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
    notifications=json.loads((folder/'NIXL_completion_decode.json').read_text())['rows']
    transfer_rows=[]
    prior_finish=None;stages=[]
    for request in requests:
        rid=f'q{request["sample_id"]}';ready=handoff[rid]['received'];submit=submissions['decode',rid]
        prepare=min(q['time'] for q in query if q['request_id']==rid)
        first_worker=request['token_events'][0]['worker_time'];first=request['first_token']
        guard=max(0.,min(submit,prior_finish)-ready) if prior_finish is not None else 0.
        stages.append(dict(request_id=rid,
            arrival_to_P_submit_ms=1000*(submissions['prefill',rid]-offered[rid]['arrival']),
            P_submit_to_parent_ready_ms=1000*(ready-submissions['prefill',rid]),
            parent_ready_to_D_submit_ms=1000*(submit-ready),
            prior_D_whole_request_guard_overlap_ms=1000*guard,
            eligible_D_host_delay_ms=1000*(submit-max(ready,prior_finish or ready)),
            D_submit_to_accept_ms=1000*(admission['decode',rid]-submit),
            D_accept_to_first_prepare_ms=1000*(prepare-admission['decode',rid]),
            D_first_prepare_to_first_worker_ms=1000*(first_worker-prepare),
            first_worker_to_client_ms=1000*(first-first_worker),TTFT_ms=request['TTFT_ms']))
        related=[r for r in transport if external_id(r.get('request_id'))==rid]
        reads=[r for r in related if r.get('kind')=='read_blocks_host']
        submits=[r for r in related if r.get('kind')=='submit_transfer_host']
        received=[r['time'] for r in notifications if any(external_id(x)==rid for x in r['finished_recving'])]
        transfer_rows.append(dict(request_id=rid,
            D_submit_to_first_KV_read_host_ms=1000*(min(r['start'] for r in reads)-submit) if reads else None,
            KV_read_API_enclosure_ms=sum(1000*(r['end']-r['start']) for r in reads),
            KV_submit_API_enclosure_ms=sum(1000*(r['end']-r['start']) for r in submits),
            last_KV_submit_return_to_scheduler_receive_notice_ms=1000*(min(received)-max(r['end'] for r in submits)) if received and submits else None,
            scheduler_receive_notice_to_first_D_prepare_ms=1000*(prepare-min(received)) if received else None,
            read_calls=len(reads),submit_calls=len(submits),
            scope='Nested host API enclosures and scheduler notifications. These fields are not additive exclusive components or pure DMA duration.'))
        prior_finish=request['finish']
    result=dict(row,canonical=value,
        mean_stages_ms={k:float(np.mean([r[k] for r in stages])) for k in stages[0] if k!='request_id'},
        request_stages=stages,
        transfer_host_boundaries=transfer_rows,
        uninstrumented=['Per-request exact DMA start/end','Exclusive destination GEMM and attention kernel time'],
        transfer_boundary='Current official pull starts after D submission. Ready-to-D is not the complete KV/hidden transfer interval.')
    return result


def main():
    old=[archived(axis,seed) for seed in SEEDS for axis in ('Native','Layer','Token')]
    groups={a:[r for r in old if r['axis']==a] for a in ('Native','Layer','Token')}
    fields=list(old[0]['mean_stages_ms'])
    difference={k:float(np.mean([r['mean_stages_ms'][k] for r in groups['Layer']])-
        np.mean([r['mean_stages_ms'][k] for r in groups['Token']])) for k in fields}
    mean_delta=float(np.mean([r['metrics']['mean_TTFT_ms'] for r in groups['Layer']])-
        np.mean([r['metrics']['mean_TTFT_ms'] for r in groups['Token']]))
    assert abs(sum(difference.values())-mean_delta)<1e-3
    guard={axis:{k:float(np.mean([r['mean_guard_wait_parts_ms'][k] for r in group]))
        for k in group[0]['mean_guard_wait_parts_ms']} for axis,group in groups.items()}
    historical=dict(starts=old,mean_stage_difference_Layer_minus_Token_ms=difference,
        mean_TTFT_difference_ms=mean_delta,guard_wait_by_axis_ms=guard,
        TTFT_p95_ratio=interval([r['metrics']['TTFT_p95_ms'] for r in groups['Layer']],
                               [r['metrics']['TTFT_p95_ms'] for r in groups['Token']]),
        scope='All three published paired starts. Guard-overlap accounting is not a guard-removal intervention or an exclusive GPU-time result.')
    historical['Native_relative']={axis:{field:interval(
        [r['metrics'][field] for r in groups[axis]],[r['metrics'][field] for r in groups['Native']])
        for field in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')} for axis in ('Layer','Token')}
    historical['P_D_host_balance_by_axis']={axis:dict(
        P_submit_to_ready_ms=float(np.mean([r['mean_stages_ms']['P_submit_to_source_notification'] for r in values])),
        D_whole_request_guard_mean_ms=float(np.mean([sum(r['mean_D_host_parts_ms'].values()) for r in values])))
        for axis,values in groups.items()}
    write(ROOT/'analysis/historical_winner_source_guard_audit_v1.json',historical)
    new=ROOT/'analysis/historical_runtime_bridge_v1.json'
    if new.exists():
        data=json.loads(new.read_text());rows=[current(r) for r in data['results'] if r.get('name') and r['status'] in ('MEASURED_ONLINE','DIAGNOSTIC')]
        comparisons=[]
        for BI in (False,True):
            pair={axis:{r['seed']:r for r in rows if r['axis']==axis and r['BI']==BI} for axis in ('Layer','Token')}
            seeds=sorted(set(pair['Layer'])&set(pair['Token']))
            if not seeds:continue
            a=[pair['Layer'][s]['metrics']['TTFT_p95_ms'] for s in seeds];b=[pair['Token'][s]['metrics']['TTFT_p95_ms'] for s in seeds]
            comparisons.append(dict(BI=BI,seeds=seeds,TTFT_p95_ratio=interval(a,b) if len(seeds)>=3 else dict(ratio=float(np.mean(a)/np.mean(b)),CI95=None,starts=len(seeds)),
                mean_stage_differences_ms={k:float(np.mean([pair['Layer'][s]['mean_stages_ms'][k]-pair['Token'][s]['mean_stages_ms'][k] for s in seeds]))
                    for k in pair['Layer'][seeds[0]]['mean_stages_ms']}))
            native={r['seed']:r for r in rows if r['axis']=='Native' and r['BI']==BI}
            if len(seeds)>=3 and all(s in native for s in seeds):
                comparisons[-1]['Native_relative']={axis:{field:interval(
                    [pair[axis][s]['metrics'][field] for s in seeds],[native[s]['metrics'][field] for s in seeds])
                    for field in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')}
                    for axis in ('Layer','Token')}
        write(ROOT/'analysis/historical_current_ready_D_comparison_v1.json',dict(rows=rows,comparisons=comparisons,
            evidence='TRACE_ANALYSIS',scope='Per-family same-request boundaries. Historical/current boundaries and return policies differ. Never subtract different cuts as a causal effect.'))
    print(json.dumps(dict(historical_mean_delta_ms=mean_delta,historical_stages=difference,guard=guard),indent=2))


if __name__=='__main__':main()
