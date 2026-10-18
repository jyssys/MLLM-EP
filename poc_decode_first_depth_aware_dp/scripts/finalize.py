"""Record actual stage completion. A missing latency is not a zero latency."""
import csv
import datetime
from poc_decode_first_depth_aware_dp.contract import *


def main():
    output=ROOT/'artifacts/final_decision.json'
    assert not output.exists()
    analysis=ROOT/'artifacts/quality_pilot_analysis_v2/quality_summary.json'
    quality=read(analysis)
    assert all(r['n']==32 for r in quality['rows'])
    assert read(ROOT/'artifacts/quality_extension_controller.json')['all_success']
    severe=any(r['severe_pilot_failure'] for r in quality['rows'])
    verdict='QUALITY_FAIL' if severe else 'INCONCLUSIVE'
    reason=('The registered Stage-0 severe pilot loss condition holds at the tested operating point.' if severe else
        'The bounded pilot does not qualify quality preservation. Independent fixed-R4 reference parity also remains partial.')
    stages=[dict(stage='0_execution',status='MEASURED',result='Online gate and physically omitted recurrent calls'),
        dict(stage='0_reference',status='PARTIAL',result='Numerical limits pass; exact natural prefix differs'),
        dict(stage='0_quality',status='MEASURED',result=verdict),
        dict(stage='1_serving_depth_cohort',status='NOT_RUN',result='Blocked by Stage 0; quality traces are diagnostic only'),
        dict(stage='2_CDB_cost_model',status='NOT_RUN',result='No live cost-surface or DP4 simulator validation'),
        dict(stage='2_live_DP4',status='NOT_RUN',result='Four quality workers are not placement baselines'),
        dict(stage='3_depth_oracle',status='NOT_RUN',result='No qualified depth-only or length-only headroom'),
        dict(stage='4_causal_predictor',status='NOT_RUN',result='G3 not reached; no predictor or RL'),
        dict(stage='5_prefill_audit',status='MEASURED',result='PREFILL_FULL_DEPTH in the execution probe')]
    write(output,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verdict=verdict,
        reason=reason,stages=stages,quality_summary_sha256=sha(analysis),
        gate_q=.5,min_depth=2,max_depth=4,KV='last_exited',
        checkpoint=MODEL_ID,revision=REVISION,scope='Tested CDB configuration only. No general DP scheduling impossibility claim.',
        full_GSM8K_1319_run=False,full_MATH500_500_run=False,
        no_threshold_tuning=True,no_RL=True,no_new_serving_engine=True,
        DP_scheduling_gain=None,oracle_incremental_depth_gain=None,
        next_action='Resolve the fixed-R4 reference qualification and establish a quality-approved adaptive point before any DP oracle.'))
    with (ROOT/'artifacts/stage_status.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['stage','status','result'],lineterminator='\n')
        writer.writeheader();writer.writerows(stages)
    # Retain the requested schemas without manufacturing measurements.
    for name,fields in [
        ('per_loop_costs.csv',['stage','status','GPU_service_ms']),
        ('live_dp4_metrics.csv',['C','arrivals','policy','P95_E2E_ms','P95_ITL_ms','goodput','status']),
        ('oracle_schedule_results.csv',['policy','incremental_depth_gain_percent','status'])]:
        with (ROOT/'artifacts'/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
            writer.writerow(dict(status='NOT_RUN_STAGE0_BLOCKED'))
    print('FINAL_DECISION',verdict)


if __name__=='__main__':main()
