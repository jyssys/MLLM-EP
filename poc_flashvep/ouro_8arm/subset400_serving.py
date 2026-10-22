"""Aggregate fixed 400-question views without recounting checkpointed cohorts."""
import argparse
import numpy as np
from .contract import *
from .effective_sources import subset_views
from .serving_analysis import table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extension-only',action='store_true');args=parser.parse_args()
    extension_indices={c['index'] for c in read(ROOT/'evaluation_subset400.json')['cohorts'] if c['source']=='extension'}
    output=[]
    for arm,view in subset_views().items():
        values=[];timing=[];resident=[];prep=[]
        for cohort in view['cohorts']:
            if args.extension_only and cohort['cohort'] not in extension_indices:continue
            values.extend(read(ROOT/file) for file in cohort['question_files'])
            record=read(ROOT/cohort['timing_file'])
            timing.append(record['makespan_s'] if 'makespan_s' in record else record['end']-record['origin'])
            if record.get('resident_model_cohort_wall_s') is not None:
                resident.append(record['resident_model_cohort_wall_s']);prep.append(record['admission_to_service_s'])
        if not values:continue
        duration=sum(timing)
        def mean(key):
            data=[r[key] for r in values if r.get(key) is not None]
            return float(np.mean(data)) if data else None
        target=286 if args.extension_only else 400
        output.append(dict(run=view['name'].replace('quality400_','extension286_') if args.extension_only else view['name'],
            arm=values[0]['arm'],N=len(values),complete=len(values)==target,
            completed_cohorts=len(timing),cohort_makespan_sum_s=duration,mean_E2E_s=mean('E2E_s'),
            P50_E2E_s=float(np.quantile([r['E2E_s'] for r in values],.50)),
            P90_E2E_s=float(np.quantile([r['E2E_s'] for r in values],.90)),
            P95_E2E_s=float(np.quantile([r['E2E_s'] for r in values],.95)),
            P99_E2E_s=float(np.quantile([r['E2E_s'] for r in values],.99)),
            mean_TTFT_s=mean('TTFT_s'),mean_TPOT_s=mean('TPOT_s'),output_tokens=sum(r['output_tokens'] for r in values),
            output_tokens_per_s=sum(r['output_tokens'] for r in values)/duration,requests_per_s=len(values)/duration,
            GPU_count=len(values[0]['GPUs']),
            output_tokens_per_s_per_GPU=sum(r['output_tokens'] for r in values)/duration/len(values[0]['GPUs']),
            requests_per_s_per_GPU=len(values)/duration/len(values[0]['GPUs']),
            preparation_inclusive_cohorts=len(resident),preparation_inclusive_cohort_sum_s=sum(resident) if len(resident)==len(timing) else None,
            preparation_to_service_sum_s=sum(prep) if len(prep)==len(timing) else None,
            lane=values[0]['lane'],status=values[0]['status'],
            warning='Service clock excludes per-cohort graph preparation. Preparation-inclusive values require complete instrumentation coverage. Static FIFO and TP wave schedules differ. Use matched-natural and identical-work replay for controls.'))
    table(ROOT/('extension286_serving.csv' if args.extension_only else 'subset400_serving.csv'),output)
    print('EXTENSION286_TASK_VIEWS' if args.extension_only else 'SUBSET400_TASK_VIEWS',[(r['run'],r['N']) for r in output],flush=True)


if __name__=='__main__':main()
