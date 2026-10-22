"""Aggregate observed stage work and explicit PP tensor copies. No latency model."""
import numpy as np
from .contract import *
from .effective_sources import subset_views
from .serving_analysis import table


def balance(values):
    mean=float(np.mean(values))
    return dict(CV=float(np.std(values)/mean) if mean else None,
        max_over_mean=max(values)/mean if mean else None)


def main():
    rank_rows=[];summaries=[]
    for arm,view in subset_views().items():
        if arm not in ARMS:continue
        n=len(ARMS[arm]['GPUs']);sources=[];requests=0;cohorts=0
        for group in view['cohorts']:
            folder=ROOT/'raw'/group['source_run']
            files=[folder/f"cohort_{group['cohort']:03d}_rank{rank}.json" for rank in range(n)]
            if not all(f.exists() for f in files):continue
            sources.extend(read(f) for f in files);requests+=len(group['ids']);cohorts+=1
        if not sources:continue
        rows=[]
        for rank in range(n):
            local=[r for r in sources if r['rank']==rank]
            counted=all('counts' in r for r in local)
            def count(key):return sum(r.get('counts',{}).get(key,0) for r in local)
            def bytes_(key):return sum(r.get('bytes',{}).get(key,0) for r in local)
            row=dict(run=view['name'],arm=arm,rank=rank,physical_GPU=rank+4,N=requests,completed_cohorts=cohorts,
                physical_layer_calls=count('physical_layer_invocations') if counted else None,
                token_layer_work=count('token_layer_work') if counted else None,
                R1_decode_visits=count('R1_decode_visits') if counted else None,
                R2_decode_visits=count('R2_decode_visits') if counted else None,
                R3_decode_visits=count('R3_decode_visits') if counted else None,
                R4_decode_visits=count('R4_decode_visits') if counted else None,
                explicit_PP_hidden_remote_bytes=bytes_('peer_prefill')+bytes_('peer_decode') if counted else None,
                explicit_PP_token_remote_bytes=bytes_('peer_token') if counted else None,
                explicit_PP_exit_KV_remote_bytes=bytes_('exit_KV') if counted else None,
                explicit_PP_exit_KV_remote_calls=count('exit_KV_copies') if counted else None,
                queued_local_repair_messages_received=count('local_exit_fills') if counted else None,
                rank_host_service_span_sum_s=sum(r['end']-r['origin'] if 'end' in r else r['makespan_s'] for r in local),
                evidence='LIVE_MEASURED stage-body and explicit copy-call counters. Tensor bytes are logical payload, not wire bytes.',
                timing_limit='Host spans include queue waits and shutdown messages. Work counts are not GPU-time shares. TP collective bytes are separate.')
            row['explicit_PP_remote_tensor_bytes']=sum(row[k] for k in ['explicit_PP_hidden_remote_bytes','explicit_PP_token_remote_bytes','explicit_PP_exit_KV_remote_bytes']) if counted else None
            rows.append(row);rank_rows.append(row)
        counted=all(r['physical_layer_calls'] is not None for r in rows)
        layer_balance=balance([r['physical_layer_calls'] for r in rows]) if counted else {}
        token_balance=balance([r['token_layer_work'] for r in rows]) if counted else {}
        summaries.append(dict(run=view['name'],arm=arm,N=requests,completed_cohorts=cohorts,
            all_400_questions_complete=requests==400,
            layer_call_CV=layer_balance.get('CV'),layer_call_max_over_mean=layer_balance.get('max_over_mean'),
            token_layer_work_CV=token_balance.get('CV'),token_layer_work_max_over_mean=token_balance.get('max_over_mean'),
            explicit_PP_hidden_remote_bytes=sum(r['explicit_PP_hidden_remote_bytes'] for r in rows) if counted else None,
            explicit_PP_exit_KV_remote_bytes=sum(r['explicit_PP_exit_KV_remote_bytes'] for r in rows) if counted else None,
            explicit_PP_exit_KV_remote_calls=sum(r['explicit_PP_exit_KV_remote_calls'] for r in rows) if counted else None,
            explicit_PP_remote_tensor_bytes=sum(r['explicit_PP_remote_tensor_bytes'] for r in rows) if counted else None,
            scope='Observed counts over complete canonical-input cohorts. No additive latency, wire-bandwidth, or E2E-causality inference.'))
    table(ROOT/'placement_work_by_rank.csv',rank_rows)
    table(ROOT/'placement_accounting_summary.csv',summaries)
    write(ROOT/'state/placement_accounting.json',dict(rows=summaries,
        exclusion='Custom TP logical row-reduction accounting is in comm_profile.csv. Missing TP stage counters are not zero communication.',
        local_repair_limit='Queued local-repair messages exclude the terminal owner immediate local fill. They are not the total number of all local copy kernels.'))
    print('PLACEMENT_ACCOUNTING',len(summaries),'RANK_ROWS',len(rank_rows),flush=True)


if __name__=='__main__':main()
