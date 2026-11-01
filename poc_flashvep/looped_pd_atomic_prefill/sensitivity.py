"""Rescore frozen evidence without changing primary SLOs or selection rules."""
from .common import *
from .analyze import serving,save_csv,wilson
from .evidence import rows
from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals


def main():
    output=[]
    for run in serving():
        if run['phase'] not in ('qps_pilot','final_repeat') or run['TTFT_threshold_ms'] is None:continue
        values=rows(run['name'])
        for multiplier in (1.5,2.,4.):
            ttft=run['TTFT_threshold_ms']*multiplier/2;tpot=run['TPOT_threshold_ms']*multiplier/2
            for max_itl in (None,25.,50.,100.,200.,500.):
                passed=0;bundles=0
                for r in values:
                    intervals,coalesced=token_intervals(r);bundles+=coalesced
                    passed+=r['ttft_ms']<=ttft and r['tpot_ms']<=tpot and (max_itl is None or bool(intervals) and max(intervals)<=max_itl)
                lo,hi=wilson(passed,run['offered'])
                output.append(dict(name=run['name'],model=run['model'],length=run['length'],output_cap=run['output_cap'],
                    arm=run['arm'],chunk=run['chunk'],transport_control=run['transport_control'],target_QPS=run['target_QPS'],
                    calibration_multiplier=multiplier,TTFT_limit_ms=ttft,request_TPOT_limit_ms=tpot,request_max_ITL_limit_ms=max_itl,
                    passed=passed,offered=run['offered'],attainment=passed/run['offered'],Wilson95_low=lo,Wilson95_high=hi,
                    finite_SLO_goodput=passed/run['elapsed_s'],coalesced_delivery_events=bundles,
                    scope='Post-hoc sensitivity only. Frozen primary SLOs and finalist decisions do not change. Coalesced delivery intervals cannot prove every device token deadline.'))
    save_csv('slo_sensitivity.csv',output);print('SENSITIVITY_ROWS',len(output));return output


if __name__=='__main__':main()
