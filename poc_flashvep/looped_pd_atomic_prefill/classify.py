"""Apply the frozen bottleneck conditions. Missing evidence never becomes a pass."""
from .common import *
from .analyze import serving,save_csv


def main():
    runs=serving();profile_path=ROOT/'profile_activity.json';profiles=json.loads(profile_path.read_text()) if profile_path.exists() else []
    mapped={r['clean_reference']:r for r in profiles if r.get('clean_reference') and r.get('eligible_profile_transfer_to_clean')}
    selected_path=ROOT/'configs/capacity_references.json';refs=json.loads(selected_path.read_text()) if selected_path.exists() else {}
    output=[]
    for r in runs:
        if r['phase'] not in ('qps_pilot','final_repeat','paper_budget_control') or r['arm'] not in ('PD_G1',):continue
        if not r['eligible_clean']:continue
        cell=refs.get(r['model']+'_L'+str(r['length']))
        if not cell or str(r['output_cap']) not in cell['D']:continue
        muP=max(v['mu_P'] for v in cell['chunks'].values());muD=cell['D'][str(r['output_cap'])]['mu_D']
        observed_batch=r.get('D_recurrent_batch_requests_mean',r.get('D_running_mean'))
        matched_D=None
        if observed_batch is not None:
            samples=json.loads((ROOT/'capacity_summary.json').read_text())
            legal=[s for s in samples if s['model']==r['model'] and s['length']==r['length'] and s['output_cap']==r['output_cap'] and
                s['capacity_mode']=='PREPARED_D' and s['batch']<=observed_batch]
            if legal:matched_D=max(legal,key=lambda s:s['batch']);muD=matched_D['finite_requests_per_s_mean']
        rate=r['realized_QPS'];p_rate=r.get('P_completion_rate_central');slope=r.get('P_queue_slope_per_s');growth=r.get('P_queue_half_growth')
        credit=r.get('D_KV_refusal_sample_fraction');slot=r.get('D_slot_refusal_sample_fraction',0. if r['model']=='qwen8' and r.get('D_scheduler_waiting_max')==0 else None)
        backlog=rate is not None and slope is not None and growth is not None and slope>=.05*rate and growth>=1.
        credit_pass=isinstance(credit,(float,int)) and isinstance(slot,(float,int)) and credit<=.01 and slot<=.01
        service=p_rate is not None and p_rate>=.85*muP
        d_slope=r.get('D_ready_queue_slope_per_s')
        reserve=p_rate is not None and matched_D is not None and muD>=1.2*p_rate and d_slope is not None and d_slope<=.05*rate
        adequate_cohort=r['offered']>=128 and r['arrival_window_s']>=30.
        profile=mapped.get(r['name']);device={d['role']:d for d in profile['devices']} if profile else {}
        p_busy=device.get('prefill',{}).get('kernel_busy_fraction');d_busy=device.get('decode',{}).get('kernel_busy_fraction')
        # A continuously active Decode kernel can still have batching reserve.
        # Kernel activity is not SM capacity. Do not invent a D-idle threshold.
        gpu_pass=p_busy is not None and p_busy>=.8
        maxlag=r.get('load_generator_lag_ms_p99');arrival_ok=maxlag is not None and rate is not None and maxlag/1000<=1/max(rate,1e-12)
        checks=dict(P_backlog_growth=bool(backlog),P_near_strongest_finite_service_rate=bool(service),D_finite_service_reserve=bool(reserve),
            D_credit_clear=bool(credit_pass),D_TPOT_margin_pass=bool(r['D_TPOT_margin_pass']),actual_P_and_D_kernel_condition=bool(gpu_pass),
            final_cohort_size_and_duration=bool(adequate_cohort),open_loop_emission_lag_bounded=bool(arrival_ok))
        if all(checks.values()):category='TRUE_PREFILL_HEAVY'
        elif isinstance(credit,(int,float)) and credit>.01 or isinstance(slot,(int,float)) and slot>.01:category='KV_ADMISSION_BOUND'
        elif not r['D_TPOT_margin_pass'] and (r['request_TPOT_ms_p95']>r.get('TPOT_threshold_ms',float('inf')) or (d_busy is not None and d_busy>=.8)):
            category='DECODE_BOUND'
        else:category='MIXED_OR_INCONCLUSIVE'
        output.append(dict(name=r['name'],model=r['model'],length=r['length'],output_cap=r['output_cap'],chunk=r['chunk'],
            category=category,screen_P_candidate=all(checks[k] for k in ('P_backlog_growth','D_finite_service_reserve','D_credit_clear','D_TPOT_margin_pass')),
            **checks,muP_finite=muP,muD_finite=muD,actual_P_completion_rate=p_rate,P_queue_slope=slope,P_queue_half_growth=growth,
            P_profiled_kernel_busy_fraction=p_busy,D_profiled_kernel_busy_fraction=d_busy,
            D_observed_recurrent_batch_mean=observed_batch,D_reference_matched_batch=matched_D['batch'] if matched_D else None,
            profile_source=profile['name'] if profile else 'NOT_AVAILABLE',
            scope='Finite observed regime only. A profile is transferred to its matched clean run only within the preregistered 5% wall-time limit. Not Dynamic-A benefit.'))
    # Compare the same offered trace. A tiny-C-only effect is not loop headroom.
    byname={r['name']:r for r in runs}
    for r in output:
        a=byname[r['name']];cell=refs[a['model']+'_L'+str(a['length'])];best=cell['best_chunk']
        controls=[b for b in runs if b['model']==a['model'] and b['length']==a['length'] and b['output_cap']==a['output_cap'] and b['phase']==a['phase'] and
            b['arm']==a['arm'] and b['independent_start']==a['independent_start'] and b['target_QPS']==a['target_QPS'] and b['chunk']==best]
        if controls and best!=a['chunk']:
            b=controls[0];r['strong_tuned_control']=b['name'];r['tuned_to_current_TTFT_p95_ratio']=b['TTFT_ms_p95']/a['TTFT_ms_p95']
            if r['screen_P_candidate'] and (b.get('P_queue_half_growth') or 0)<1 and b['TTFT_ms_p95']<=.8*a['TTFT_ms_p95']:
                r['category']='GENERIC_TUNING_ARTIFACT'
    save_csv('bottleneck_classification.csv',output);write(ROOT/'bottleneck_classification.json',output)
    print('CLASSIFIED',len(output));return output


if __name__=='__main__':main()
