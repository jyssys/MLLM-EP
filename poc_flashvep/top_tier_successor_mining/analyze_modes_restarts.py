"""Keep independent engine restarts distinct from within-engine repetitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
    out=a.result/'analysis/modes_b16_independent_20260908';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for run in [1,2,3]:
        name=f'ep_clean_renderer_gqa_b16_run{run}_modes_cached_20260908'
        raw=a.result/'online'/name
        completed=json.loads((raw/'completed.json').read_text())
        assert completed['exit_codes']==[0,0] and not (raw/'INVALIDATED.json').exists()
        for x in json.loads((a.result/'analysis'/name/'summary.json').read_text()):
            if x['policy']=='vanilla':continue
            rows.append(dict(run=run,policy=x['policy'],observations=x['observations'],
                within_engine_repetitions=x['cohort_repetitions'],
                quality_delta_pp=x['paired_quality_delta_pp'],
                paired_e2e_reduction_percent=x['paired_e2e_reduction_median_percent'],
                within_engine_ci_low=x['paired_e2e_reduction_95ci_percent'][0],
                within_engine_ci_high=x['paired_e2e_reduction_95ci_percent'][1],
                aggregate_request_reduction_percent=x['aggregate_request_latency_reduction_percent']))
    df=pd.DataFrame(rows);df.to_csv(out/'runs.csv',index=False)
    results=[]
    for policy,g in df.groupby('policy'):
        values=g.paired_e2e_reduction_percent.to_numpy()
        results.append(dict(policy=policy,independent_engines=len(g),
            paired_run_results_percent=values.tolist(),median_across_engines_percent=float(np.median(values)),
            observed_engine_range_percent=[float(values.min()),float(values.max())],
            runs_ge5percent=int((values>=5).sum()),
            scope='Same128GQA questions; independent engines,not independent datasets.',
            warning='Only3 engines; within-engine bootstrap CI is not cross-engine confidence.'))
    (out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
