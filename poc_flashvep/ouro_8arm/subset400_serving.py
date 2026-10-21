"""Aggregate fixed 400-question views without recounting checkpointed cohorts."""
import numpy as np
from .contract import *
from .effective_sources import subset_views
from .serving_analysis import table


def main():
    output=[]
    for arm,view in subset_views().items():
        values=[];timing=[];resident=[];prep=[]
        for cohort in view['cohorts']:
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
        output.append(dict(run=view['name'],arm=values[0]['arm'],N=len(values),complete=not view['missing_cohorts'],
            completed_cohorts=len(timing),cohort_makespan_sum_s=duration,mean_E2E_s=mean('E2E_s'),
            mean_TTFT_s=mean('TTFT_s'),mean_TPOT_s=mean('TPOT_s'),output_tokens=sum(r['output_tokens'] for r in values),
            output_tokens_per_s=sum(r['output_tokens'] for r in values)/duration,requests_per_s=len(values)/duration,
            preparation_inclusive_cohorts=len(resident),preparation_inclusive_cohort_sum_s=sum(resident) if len(resident)==len(timing) else None,
            preparation_to_service_sum_s=sum(prep) if len(prep)==len(timing) else None,
            lane=values[0]['lane'],status=values[0]['status'],
            warning='Service clock excludes per-cohort graph preparation. Preparation-inclusive values require complete instrumentation coverage. Static FIFO and TP wave schedules differ. Use matched-natural and identical-work replay for controls.'))
    table(ROOT/'subset400_serving.csv',output)
    print('SUBSET400_TASK_VIEWS',[(r['run'],r['N']) for r in output],flush=True)


if __name__=='__main__':main()
