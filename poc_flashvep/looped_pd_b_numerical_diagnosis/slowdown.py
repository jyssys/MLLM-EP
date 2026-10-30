"""CPU-only decomposition of saved matched-runtime measurements."""
import csv
import json
from statistics import mean
import numpy as np
from .common import ROOT,PRIOR,setup,write

RUNS=[
 ('original','ouro','ouro_W1_PD_G1_N2_rate1_seed2026103001','ouro_W1_B_N2_rate1_seed2026103001'),
 ('original','huginn','huginn_W1_PD_G1_N2_rate0.2_seed2026103001','huginn_W1_B_N2_rate0.2_seed2026103001'),
 ('canonical','ouro','repair_epoch_ouro_W1_PD_G1_N2_rate1_seed2026106001','repair_epoch_ouro_W1_B_N2_rate1_seed2026106001'),
 ('canonical','ouro','repair_epoch_ouro_W1_PD_G1_N2_rate1_seed2026107001','repair_epoch_ouro_W1_B_N2_rate1_seed2026107001'),
]


def summarize(folder):
    rows={r['sample_id']:r for line in (folder/'requests.jsonl').read_text().splitlines()
          if (r:=json.loads(line)) and not r.get('warmup')}
    before=json.loads((folder/'before_measured.json').read_text())
    after=json.loads((folder/'after_measured.json').read_text())
    closed=json.loads((folder/'closed_stats.json').read_text())
    counters={key:sum(r.get(key,0)-before.get(k,{}).get(key,0) for k,r in after.items())
              for key in ('transport_bytes_sent','transport_completed_count','transport_completion_observation_seconds',
                          'issued_loop_rows','issued_submissions')}
    phases={}
    details=[]
    for record in closed['metrics']:
        rid=record['request_id']
        if not rid.startswith('r0_q'): continue
        index=int(rid.split('_q')[1])
        if index not in rows: continue
        times=record['timings']
        needed=('admitted','d_reserved','prefill_started','prefill_compute_completed','d_received','first_token','finished')
        if not all(key in times for key in needed): continue
        row=rows[index]
        item=dict(sample_id=index,input_tokens=row['input_tokens'],output_tokens=row['output_tokens'],
                  TTFT_ms=row['ttft_ms'],E2E_ms=row['e2e_ms'],
                  harness_submission_ms=1000*(record['created_monotonic']-row['arrival']),
                  coordinator_queue_ms=1000*times['admitted'],
                  reservation_ms=1000*(times['prefill_started']-times['admitted']),
                  prefill_with_backpressure_ms=1000*(times['prefill_compute_completed']-times['prefill_started']),
                  exposed_handoff_ms=1000*(times['d_received']-times['prefill_compute_completed']),
                  resume_first_token_ms=1000*(times['first_token']-times['d_received']),
                  remaining_decode_ms=1000*(times['finished']-times['first_token']),
                  harness_first_token_observation_ms=1000*(row['first_token']-record['created_monotonic']-times['first_token']))
        details.append(item)
    for key in details[0] if details else ():
        if key.endswith('_ms'):
            vals=[row[key] for row in details]
            phases[key]=dict(mean=float(np.mean(vals)),p50=float(np.percentile(vals,50)),
                             p95=float(np.percentile(vals,95)),p99=float(np.percentile(vals,99)))
    byworker={role:{key:r.get(key,0)-before.get(name,{}).get(key,0)
                   for key in ('issued_submissions','issued_loop_rows','transport_completed_count')}
              for name,r in after.items() for role in ('prefill' if 'prefill' in name else 'decode',)}
    return rows,dict(N=len(rows),counters=counters,host_phases=phases,by_worker=byworker),details


def main():
    setup()
    results=[]
    all_details=[]
    deltas=[]
    for epoch,model,left,right in RUNS:
        a,sa,da=summarize(PRIOR/'raw'/left)
        b,sb,db=summarize(PRIOR/'raw'/right)
        matched=sorted(a.keys() & b.keys())
        result=dict(epoch=epoch,model=model,control=left,B=right,control_metrics=sa,B_metrics=sb,
                    exact_output=sum(a[i]['output_ids']==b[i]['output_ids'] for i in matched),N=len(matched),
                    input_exact=sum(a[i]['input_ids']==b[i]['input_ids'] for i in matched),
                    fragment_ratio=sb['counters']['transport_completed_count']/sa['counters']['transport_completed_count'],
                    payload_equal=sb['counters']['transport_bytes_sent']==sa['counters']['transport_bytes_sent'],
                    evidence='TRACE_ANALYSIS_OF_PRIOR_GPU_MEASUREMENTS')
        # Percentile differences are paired within a single saved run pair. They do not estimate independent-start uncertainty.
        rng=np.random.default_rng(20261030)
        x=np.array([a[i]['ttft_ms'] for i in matched])
        y=np.array([b[i]['ttft_ms'] for i in matched])
        samples=rng.integers(0,len(x),(2000,len(x)))
        bootstrap_p95_deltas=np.percentile(y[samples],95,axis=1)-np.percentile(x[samples],95,axis=1)
        result['TTFT_p95_delta_B_minus_control_ms']=dict(point=float(np.percentile(y,95)-np.percentile(x,95)),
              paired_request_bootstrap_95=np.percentile(bootstrap_p95_deltas,[2.5,97.5]).tolist(),
              scope='Descriptive within-run request bootstrap. Queue dependence and between-start variation not modeled.')
        results.append(result)
        mapping_a={r['sample_id']:r for r in da};mapping_b={r['sample_id']:r for r in db}
        terms=('harness_submission_ms','coordinator_queue_ms','reservation_ms','prefill_with_backpressure_ms',
               'exposed_handoff_ms','resume_first_token_ms','harness_first_token_observation_ms')
        for term in (*terms,'TTFT_ms'):
            pairs=sorted(mapping_a.keys()&mapping_b.keys())
            delta=[mapping_b[i][term]-mapping_a[i][term] for i in pairs]
            deltas.append(dict(epoch=epoch,model=model,control=left,B=right,N=len(pairs),component=term,
                               control_mean_ms=mean(mapping_a[i][term] for i in pairs),
                               B_mean_ms=mean(mapping_b[i][term] for i in pairs),mean_delta_ms=mean(delta),
                               scope='Mean paired host-phase difference. Not additive p95 attribution or GPU DMA time.'))
        for mapping in (mapping_a,mapping_b):
            if any(abs(sum(r[t] for t in terms)-r['TTFT_ms'])>1e-4 for r in mapping.values()):
                raise AssertionError('TTFT phase decomposition did not reconstruct the measured timestamp')
        for arm,records in (('PD_G1',da),('B',db)):
            all_details.extend(dict(epoch=epoch,model=model,arm=arm,run=left if arm=='PD_G1' else right,**r) for r in records)
    write(ROOT/'configs/saved_slowdown_analysis.json',results)
    with (ROOT/'saved_phase_pairs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_details[0]))
        writer.writeheader();writer.writerows(all_details)
    with (ROOT/'phase_delta_attribution.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(deltas[0]));writer.writeheader();writer.writerows(deltas)
    print(json.dumps([{key:r[key] for key in ('epoch','model','N','exact_output','fragment_ratio','payload_equal','TTFT_p95_delta_B_minus_control_ms')}
                       for r in results],indent=2))


if __name__=='__main__': main()
