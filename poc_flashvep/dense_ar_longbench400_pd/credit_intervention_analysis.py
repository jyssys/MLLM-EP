"""Separate source admission waits from host prepare and continuation paths."""
import json
import time
import numpy as np
from .common import ROOT,write
from .metrics import summarize
from .service_accounting import analyze,records,compare


def one(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    path=analyze(name)
    events=records(folder/'credit_events.jsonl')
    acquire={r['request_id']:r['time'] for r in events if r['event']=='ACQUIRE_P_DISPATCH'}
    release={r['request_id']:r['time'] for r in events if r['event']=='RELEASE_FIRST_CLIENT_TOKEN'}
    requests={f"q{r['sample_id']}":r for r in records(folder/'requests.jsonl')}
    assert len(acquire)==len(release)==cfg['count']
    assert set(acquire)==set(release)==set(requests)
    rows=[]
    for row in path['rows']:
        rid=row['request_id']
        arrival=requests[rid]['arrival']
        assert arrival<=acquire[rid]<=release[rid]
        rows.append(dict(request_id=rid,prompt_id=row['prompt_id'],
            arrival_to_credit_acquire_ms=1000*(acquire[rid]-arrival),
            credit_holding_ms=1000*(release[rid]-acquire[rid]),
            arrival_to_P_prepare_ms=sum(row[k] for k in ('Arrival_to_P_submit_ms',
                'P_submit_to_accept_ms','P_accept_to_prepare_ms')),
            P_prepare_to_ready_ms=row['P_prepare_to_ready_ms'],
            Ready_to_D_submit_ms=row['Ready_to_D_submit_ms'],
            D_prepare_to_first_worker_ms=row['D_prepare_to_first_worker_ms'],
            D_decode_residency_ms=row['D_decode_residency_ms'],
            actual_output_tokens=row['output_tokens']))
    fields=[k for k in rows[0] if k.endswith('_ms')]
    return dict(name=name,credit=cfg['continuous_source_credit_limit'],
        means={k:float(np.mean([r[k] for r in rows])) for k in fields},rows=rows,
        metrics=summarize(name),source_capsule_SHA256=cfg['source_capsule_SHA256'],
        actual_GPU_pair=cfg['gpus'])


def main():
    families=[]
    for axis in ('Native','Layer','Token','Hybrid'):
        label=axis+('4' if axis=='Hybrid' else '8')
        results=[]
        for credit in (1,4,8,16):
            name=f'CreditTune_Discovery200_q2.8_{label}_C{credit}_Pool16_v1'
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if receipt.exists() and json.loads(receipt.read_text())['complete']:
                results.append(one(name))
        controls=[r for r in results if r['credit']==4]
        if controls:
            base=controls[0]
            ca=json.loads((ROOT/'raw'/base['name']/'configuration.json').read_text())
            for row in results:
                cb=json.loads((ROOT/'raw'/row['name']/'configuration.json').read_text())
                for field in ('source_capsule_SHA256','dataset_SHA256','arrival_offsets','seed','gpus',
                    'axis_cut','token_budget','max_num_seqs','KV_block_tokens','hidden_source_slots',
                    'hidden_destination_slots','d_compute_throttle','defer_D_remote_setup'):
                    assert ca.get(field)==cb.get(field),(axis,row['credit'],field)
                row['mean_delta_from_credit4_ms']={k:row['means'][k]-base['means'][k] for k in base['means']}
                row['paired_request_host_boundaries_from_credit4']=compare(row['name'],base['name'])
        families.append(dict(axis=axis,results=results,complete=len(results)==4))
    value=dict(time=time.time(),families=families,complete=all(f['complete'] for f in families),
        evidence='MEASURED_ONLINE_WITH_TRACE_ANALYSIS',independent_starts_per_credit=1,
        scope='Within each axis,source credit is the intervention. Keep source,pair,trace,hidden pool and D policy fixed. '
            'Acquire occurs at P dispatch. Release occurs at first client output,not final decode completion. '
            'Arrival-to-acquire is source admission wait. It does not identify GPU idle or a full critical-path cause. '
            'Host prepare-to-ready can include packing and other requests between chunks. '
            'Mean stage deltas do not decompose TTFT p95. Natural EOS outputs can differ. '
            'No independent-start CI or cross-axis hardware-invariant causal claim. Credit1 is not the serving default.')
    write(ROOT/'analysis/source_credit_host_intervention_v1.json',value)
    print([(f['axis'],[r['credit'] for r in f['results']]) for f in families],flush=True)


if __name__=='__main__':
    main()
