"""Reconstruct source eligibility from observed events, without a GPU model."""
import json
import numpy as np
from .common import ROOT,PREVIOUS,write


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def decompose(arrival,previous_ready_received,credit_return,dispatch,prepare):
    noncredit=max(arrival,previous_ready_received)
    eligible=max(noncredit,credit_return)
    values=dict(previous_source_chain_ms=1000*(noncredit-arrival),
        direct_credit_block_ms=1000*(eligible-noncredit),
        eligible_to_host_dispatch_ms=1000*(dispatch-eligible),
        dispatch_to_first_prepare_ms=1000*(prepare-dispatch))
    values['sum_ms']=sum(values.values())
    values['observed_P_queue_ms']=1000*(prepare-arrival)
    assert abs(values['sum_ms']-values['observed_P_queue_ms'])<1e-3
    assert values['eligible_to_host_dispatch_ms']>=-1e-3
    return values


def summarize(values):
    return dict(n=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),
        maximum=float(max(values)),sum=float(sum(values)))


def audit(folder):
    cfg=json.loads((folder/'configuration.json').read_text())
    output=sorted(lines(folder/'requests.jsonl'),key=lambda r:r['sample_id'])
    decisions={r['request_id']:r for r in lines(folder/'axis_decisions.jsonl')}
    handoffs={r['request_id']:r for r in lines(folder/'handoffs.jsonl')}
    queries=json.loads((folder/'actual_queries_prefill.json').read_text())['rows']
    credit=cfg['source_credit_limit'];rows=[]
    for index,row in enumerate(output):
        rid=f'q{row["sample_id"]}'
        first=min(q['time'] for q in queries if q['request_id']==rid)
        previous=handoffs[f'q{output[index-1]["sample_id"]}']['received'] if index else row['arrival']
        returned=output[index-credit]['first_token'] if index>=credit else row['arrival']
        pieces=decompose(row['arrival'],previous,returned,decisions[rid]['time'],first)
        pieces.update(prompt_id=row['prompt_id'],request_id=rid,input_tokens=row['input_tokens'],
            P_prepare_to_ready_host_ms=1000*(handoffs[rid]['source_ready']-first),
            ready_to_parent_host_ms=1000*(handoffs[rid]['received']-handoffs[rid]['source_ready']))
        rows.append(pieces)
    fields=('previous_source_chain_ms','direct_credit_block_ms','eligible_to_host_dispatch_ms',
        'dispatch_to_first_prepare_ms','observed_P_queue_ms','P_prepare_to_ready_host_ms','ready_to_parent_host_ms')
    return dict(name=folder.name,credit=credit,rows=rows,
        summary={key:summarize([r[key] for r in rows]) for key in fields},
        evidence='TRACE_ANALYSIS',
        scope='Exact observed-event accounting under FIFO P1/D1. Previous-source-chain waiting includes accumulated earlier service and delay. '
        'A direct credit-block interval is not the total counterfactual effect of changing credit. Host envelopes are not GPU-exclusive time.')


def main():
    names={axis:f'ControlledDiscovery_{axis}4_N200_q2.35_s2026112061_BI0_v3' for axis in ('Layer','Token')}
    results={axis:audit(PREVIOUS/'raw'/name) for axis,name in names.items()}
    fields=results['Layer']['summary']
    differences={key:results['Layer']['summary'][key]['mean']-results['Token']['summary'][key]['mean'] for key in fields}
    paths={axis:json.loads((PREVIOUS/'raw'/name/'configuration.json').read_text()) for axis,name in names.items()}
    assert paths['Layer']['arrival_offsets']==paths['Token']['arrival_offsets']
    assert [r['prompt_id'] for r in results['Layer']['rows']]==[r['prompt_id'] for r in results['Token']['rows']]
    value=dict(results=results,Layer_minus_Token_mean_ms=differences,same_arrival_trace=True,
        same_request_order=True,evidence='HISTORICAL_TRACE_ANALYSIS',
        scope='This audit explains the location of the reported2711ms P-queue difference. '
        'It does not add independent starts, prove a GPU cause, or mix Last8 serial sums with Last4 online differences.')
    write(ROOT/'analysis/historical_Last4_source_eligibility_accounting.json',value)
    print(json.dumps(dict(differences=differences,means={axis:r['summary'] for axis,r in results.items()}),indent=2))


if __name__=='__main__':main()
