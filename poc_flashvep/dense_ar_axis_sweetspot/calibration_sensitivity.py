"""Report cost-rank uncertainty without changing any frozen predictor."""
import collections
import json
import statistics
import time
from .common import ROOT,digest,write
from .resampling import mean_interval


def main():
    receipt=ROOT/'state/service_model_v3_frozen.json'
    if not receipt.exists():return
    grouped=collections.defaultdict(dict);sources=[]
    for row in json.loads(receipt.read_text())['training']:
        name=row['name'];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        feedback=ROOT/'analysis'/(name+'_reservation_feedback.json')
        observed=json.loads(feedback.read_text());cost=row['cost_seconds']
        parts=observed['mean_backlogged_nonwindow_gap_parts_ms']
        values=dict(P_service_ms=1000*cost['P'],
            uncensored_backlog_gap_ms=observed['mean_backlogged_nonwindow_P_gap_ms'],
            ACK_part_ms=parts['reservation_RTT_ms'],
            D_preparation_ms=1000*cost['preparation'],
            D_cohort_residence_per_request_ms=1000*cost['decode']/4)
        values['effective_P_demand_ms']=values['P_service_ms']+values['uncensored_backlog_gap_ms']
        values['effective_D_demand_ms']=values['D_preparation_ms']+values['D_cohort_residence_per_request_ms']+1000*cost['D_gap']/4
        grouped[cfg['arrival_seed']][cfg['evaluation_arm']]=values
        sources.append(dict(name=name,feedback_SHA256=digest(feedback),values=values))
    assert len(grouped)==3 and all(set(r)=={'Layer','Token'} for r in grouped.values())
    difference={}
    for field in next(iter(grouped.values()))['Layer']:
        vals=[r['Layer'][field]-r['Token'][field] for _,r in sorted(grouped.items())]
        difference[field]=dict(Layer_minus_Token_per_start_ms=vals,mean_ms=statistics.mean(vals),
            paired_start_CI95_ms=mean_interval(vals))
    write(ROOT/'analysis/new_calibration_cost_rank_uncertainty.json',dict(time=time.time(),
        calibration_starts_per_axis=3,seed_pairs=sorted(grouped),differences=difference,sources=sources,
        evidence='TRACE_ANALYSIS',
        scope='Only the six fresh calibration traces. This diagnostic does not refit any cost or rule. Host-demand uncertainty is not exclusive-kernel variance, WCET, or a held-out accuracy result. Separate medians in the frozen Lookup can differ from these paired mean differences.'))
    print('CALIBRATION_COST_RANK_UNCERTAINTY',difference['effective_P_demand_ms'],flush=True)


if __name__=='__main__':
    main()
