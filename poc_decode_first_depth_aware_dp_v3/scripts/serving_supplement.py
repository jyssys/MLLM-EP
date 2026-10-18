"""Additional host-visible serving metrics from preserved live outputs."""
import csv
import numpy as np
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    result=[]
    for run,plan in [('live_clean_v1','live_clean_plan.json'),
        ('hindsight_confirmation_v1','hindsight_confirmation_frozen.json'),
        ('hindsight_confirmation_c64_v1','hindsight_confirmation_c64_frozen.json')]:
        folder=ROOT/'outputs'/run
        for case in read(ROOT/'artifacts'/plan)['cases']:
            ranks=[read(folder/f'rank_{i}'/(case['name']+'.json')) for i in range(4)]
            outputs=[o for rank in ranks for o in rank['outputs']]
            decode=[1000*(o['finish_s']-o['first_token_s']) for o in outputs]
            tpot=[1000*(o['finish_s']-o['first_token_s'])/(o['generated_tokens']-1)
                for o in outputs if o['generated_tokens']>1]
            itl=[1000*(b-a) for o in outputs for a,b in zip(o['token_host_times_s'],o['token_host_times_s'][1:])]
            result.append(dict(run=run,case=case['name'],C=case['concurrency'],arrival=case['arrival'],
                policy=case.get('arm',case['policy']),requests=len(outputs),
                mean_request_TPOT_ms=float(np.mean(tpot)),decode_latency_P50_ms=float(np.quantile(decode,.5)),
                decode_latency_P95_ms=float(np.quantile(decode,.95)),ITL_P50_ms=float(np.quantile(itl,.5)),
                ITL_P99_ms=float(np.quantile(itl,.99)),
                peak_rank_memory_allocated_bytes=max(r['memory_allocated'] for r in ranks),
                peak_rank_max_memory_allocated_bytes=max(r['peak_memory'] for r in ranks),
                label='MEASURED_HOST_VISIBLE_LIVE_SERVING',
                queue_delay='UNRESOLVED: worker admission is stored, but exact first-schedule timestamp is not. TTFT includes that wait.'))
    write(ROOT/'artifacts/supplementary_serving_metrics.json',result)
    with (ROOT/'artifacts/supplementary_serving_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)

if __name__=='__main__':main()
