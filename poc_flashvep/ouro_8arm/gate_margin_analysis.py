"""Inspect recorded gate margins. It does not rerun a model or infer missing CDFs."""
from .contract import *
from .behavior_analysis import first_difference
from .serving_analysis import table


def main():
    reference=ROOT/'raw/CDB_SD_pilot_v1';candidate=ROOT/'raw/quality_SD_pilot_v1'
    cohorts={};rows=[]
    for file in sorted(reference.glob('*.json')):
        a=read(file)
        if not isinstance(a,dict) or 'output_ids' not in a:continue
        other=candidate/file.name
        if not other.exists():continue
        b=read(other);assert a['prompt_ids']==b['prompt_ids']
        step=first_difference(a['depths'],b['depths'])
        token_step=first_difference(a['output_ids'],b['output_ids'])
        if step is None or step>=min(len(a['depths']),len(b['depths'])):continue
        index=b['cohort_index'];path=candidate/f'cohort_{index:03d}_rank0.json'
        if not path.exists():continue
        if index not in cohorts:cohorts[index]=read(path)['gates']
        reference_depth=a['depths'][step];candidate_depth=b['depths'][step]
        common_depth=min(reference_depth,candidate_depth)
        custom=next((v for v in cohorts[index][a['id']] if v['step']==step and v['loop']==common_depth),None)
        event=next((v for v in a['depth_events'] if v['position']==len(a['prompt_ids'])+step-1),None)
        # The reference records only its exit CDF. Earlier-loop CDFs are unavailable.
        ref_CDF=event['cdf'] if event and event['depth']==common_depth else None
        custom_CDF=custom['cdf'] if custom else None
        rows.append(dict(id=a['id'],dataset=a['dataset'],first_depth_difference_index=step,
            first_token_difference_index=token_step,
            matched_prior_token_history=token_step is None or step<=token_step,
            reference_depth=reference_depth,candidate_depth=candidate_depth,comparison_loop=common_depth,
            reference_CDF=ref_CDF,candidate_CDF=custom_CDF,
            reference_margin_vs_q=ref_CDF-.5 if ref_CDF is not None else None,
            candidate_margin_vs_q=custom_CDF-.5 if custom_CDF is not None else None,
            evidence='LIVE_MEASURED recorded gate values. Same token history does not imply equal hidden or KV tensors.',
            missing_scope='The official reference saves only the final exit CDF. No earlier missing value is estimated.'))
    table(ROOT/'gate_margin_first_drift.csv',rows)
    available=[r for r in rows if r['matched_prior_token_history'] and r['reference_CDF'] is not None and r['candidate_CDF'] is not None]
    write(ROOT/'state/gate_margin_analysis.json',dict(first_depth_drift_requests=len(rows),
        matched_prior_token_history=sum(r['matched_prior_token_history'] for r in rows),
        both_CDF_values_recorded_at_same_loop=len(available),
        opposite_threshold_sides=sum((r['reference_margin_vs_q']>=0)!=(r['candidate_margin_vs_q']>=0) for r in available),
        limit='A threshold crossing is a gate observation, not an isolated proof of which upstream kernel caused the difference.'))
    print('RECORDED_FIRST_GATE_DRIFTS',len(rows),'SAME_HISTORY_BOTH_CDF',len(available),flush=True)


if __name__=='__main__':main()
