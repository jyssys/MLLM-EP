"""Reconstruct primary client metrics directly from immutable event records."""
import json
import math
import time
import numpy as np
from .common import ROOT,write,digest


def one(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    cached=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
    rows=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    window=json.loads((folder/'window.json').read_text());seconds=window['end']-window['start']
    assert len(rows)==cfg['count'] and len({r['sample_id'] for r in rows})==cfg['count']
    assert seconds>0 and window['start']<=min(r['arrival'] for r in rows)
    assert window['end']>=max(r['finish'] for r in rows)
    ttft=[];tpot=[];end_to_end=[];itl=[];good=[];decode=[];notification=[]
    for row in rows:
        n=len(row['output_ids']);assert n==row['output_tokens'] and n>=1
        events=row['token_events'];assert events[0]['count']>=1 and events[-1]['count']==n
        first=events[0]['time'];finish=row['finish']
        assert first==row['first_token'] and first<=events[-1]['time']<=finish
        slots=[]
        previous_time=first;previous_count=0
        for event in events:
            count=event['count'];assert count>previous_count and event['time']>=previous_time
            if previous_count:
                slots.append(1000*(event['time']-previous_time))
                slots.extend([0.]*(count-previous_count-1))
            else:slots.extend([0.]*(count-1))
            previous_time=event['time'];previous_count=count
        assert len(slots)==n-1
        a=1000*(first-row['arrival']);b=1000*(finish-first)/(n-1) if n>1 else None
        e=1000*(finish-row['arrival'])
        assert math.isclose(a,row['TTFT_ms'],abs_tol=1e-6)
        assert (b is None and row['TPOT_ms'] is None) or math.isclose(b,row['TPOT_ms'],abs_tol=1e-6)
        assert math.isclose(e,row['E2E_ms'],abs_tol=1e-6)
        ttft.append(a);end_to_end.append(e);itl.extend(slots)
        if b is not None:tpot.append(b)
        passed_decode=all(v<=34 for v in slots) and (b is None or b<=33)
        decode.append(passed_decode);good.append(passed_decode and a<=5000)
        notification.append(1000*(finish-events[-1]['time']))
        if cfg.get('ignore_eos'):
            lengths=cfg.get('replay_output_lengths')
            expected=lengths[row['sample_id']] if lengths is not None else cfg['output_cap']
            assert n==expected,(name,row['sample_id'],n,expected)
    pct=lambda values,q:float(np.percentile(values,q)) if values else None
    tokens=sum(r['output_tokens'] for r in rows)
    reconstructed=dict(n=len(rows),TTFT_p95_ms=pct(ttft,95),TTFT_mean_ms=float(np.mean(ttft)),
        TTFT_p50_ms=pct(ttft,50),TTFT_p99_ms=pct(ttft,99),ITL_p99_ms=pct(itl,99),
        ITL_p95_ms=pct(itl,95),mean_TBT_ms=float(np.mean(itl)) if itl else None,
        max_ITL_ms=max(itl,default=0),ITL_interval_count=len(itl),TPOT_p95_ms=pct(tpot,95),
        TPOT_p50_ms=pct(tpot,50),TPOT_p99_ms=pct(tpot,99),E2E_p95_ms=pct(end_to_end,95),
        output_tokens=tokens,generated_output_tokens_per_second=tokens/seconds,
        completed_requests_per_second=len(rows)/seconds,Joint_SLO_attainment=float(np.mean(good)),
        decode_SLO_attainment=float(np.mean(decode)),
        SLO_qualified_output_tokens_per_second=sum(r['output_tokens'] for r,p in zip(rows,good) if p)/seconds,
        measurement_seconds=seconds)
    for key,value in reconstructed.items():
        observed=cached[key]
        assert (value is None and observed is None) or math.isclose(value,observed,rel_tol=1e-11,abs_tol=1e-6),(name,key,value,observed)
    output=[r['output_tokens'] for r in rows]
    return dict(name=name,passed=True,fields_verified=list(reconstructed),
        request_count=len(rows),actual_output_tokens=tokens,
        output_length=dict(mean=float(np.mean(output)),p50=pct(output,50),p90=pct(output,90),
            p95=pct(output,95),p99=pct(output,99),min=min(output),max=max(output),
            cap_hits=sum(bool(r['cap_hit']) for r in rows)),
        completion_notification_after_last_token_ms=dict(mean=float(np.mean(notification)),
            p99=pct(notification,99),max=max(notification)),
        requests_SHA256=digest(folder/'requests.jsonl'),window_SHA256=digest(folder/'window.json'))


def main():
    names=[];missing_plans=[]
    for manifest in ('final_continuous_execution_plan_v1.json','controlled_continuous_I3_bridge_frozen_v1.json',
            'hybrid_READ_repair_frozen_v1.json','hybrid_READ_repair_layer_control_extension_v1.json',
            'request_registration_deferral_frozen_v1.json','request_registration_q28_extension_frozen_v1.json',
            'request_registration_controlled_bridge_frozen_v1.json','scoped_compile_confirmation_frozen_v1.json',
            'eager_prefill_native_decode_confirmation_frozen_v2.json'):
        path=ROOT/'configs'/manifest
        if path.exists():
            plan=json.loads(path.read_text())
            names.extend(name for cell in plan['cells'] for name in cell['jobs'].values())
        else:missing_plans.append(manifest)
    for path in ('analysis/axis_matrix_heldout_confirmation_r2.json','configs/iso8_hybrid_confirmation_frozen_v1.json'):
        value=json.loads((ROOT/path).read_text())
        for cell in value['cells']:
            blocks=cell.get('results',[cell])
            for block in blocks:
                names.extend(block[a] for a in ('Native','Layer','Token','Hybrid') if a in block)
    failed_other_dispatches=[]
    for receipt in (ROOT/'raw').glob('*/dispatcher_result.json'):
        if json.loads(receipt.read_text())['complete']:
            names.append(receipt.parent.name)
        else:failed_other_dispatches.append(receipt.parent.name)
    stop_file=ROOT/'state/scoped_compile_confirmation_complete_v1.json'
    stopped={r['name']:r for r in json.loads(stop_file.read_text())['results']
        if r['status']=='NOT_RUN' and r.get('reason')} if stop_file.exists() else {}
    stop_file=ROOT/'state/eager_compile_confirmation_v2_complete_v1.json'
    if stop_file.exists():
        stopped.update({r['name']:r for r in json.loads(stop_file.read_text())['results']
            if r['status']=='NOT_RUN' and r.get('reason')})
    results=[];pending=[];failed=[];not_run=[]
    for name in sorted(set(names)):
        receipt=ROOT/'raw'/name/'dispatcher_result.json'
        if not receipt.exists():
            if name in stopped:not_run.append(stopped[name])
            else:pending.append(name)
            continue
        if not json.loads(receipt.read_text())['complete']:failed.append(name);continue
        results.append(one(name))
    value=dict(time=time.time(),passed=not pending and not failed and not missing_plans,verified=len(results),
        missing_plans=missing_plans,
        failed_dispatcher_records_preserved=failed_other_dispatches,
        results=results,pending=pending,failed=failed,evidence='RAW_CLIENT_METRIC_RECONSTRUCTION',
        explicit_NOT_RUN=not_run,
        scope='Reconstruct every complete raw dispatcher record,including motivation,controls,profiling and correctness-only records. '
            'Pending counts refer to the explicitly named core plans. Failed dispatcher records stay separate and have no reconstructed valid latency. '
            'Reconstruct from event counts,delivery times,scheduled arrivals and completion notifications. '
            'Do not call the primary interval or metric implementation. Keep its frozen definitions. '
            'Co-delivered tokens have zero client separation. Request TPOT includes the final completion-notification boundary. '
            'The notification delta is reported separately,not silently removed from TPOT. '
            'A metric reconstruction does not qualify correctness,observer overhead,statistical independence or sustainable capacity.')
    write(ROOT/'analysis/final_independent_client_metric_audit_v1.json',value)
    print('INDEPENDENT_CLIENT_METRICS',len(results),'verified',len(pending),'pending',len(failed),'failed',flush=True)


if __name__=='__main__':
    main()
