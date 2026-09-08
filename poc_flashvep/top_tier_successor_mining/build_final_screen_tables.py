"""Materialize final paired screening tables; require every planned clean run.

Keep each policy paired with its own engine's vanilla. Do not compare absolute
latency from the corrected MoDES restart to SERE's separate restart.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
    out=a.result/'analysis/final_screen_20260908';out.mkdir(parents=True,exist_ok=True)
    rows=[];all_count=0;proof=[]
    for dataset in ['chartqa','gqa']:
        for batch in [1,4,16]:
            for method,suffix in [('SERE','run0'),('MoDES','run1_modes_cached')]:
                name=f'ep_clean_renderer_{dataset}_b{batch}_{suffix}_20260908'
                raw=a.result/'online'/name
                completed=json.loads((raw/'completed.json').read_text())
                assert completed['exit_codes']==[0,0] and not (raw/'INVALIDATED.json').exists(),name
                summaries=json.loads((a.result/'analysis'/name/'summary.json').read_text())
                all_count+=sum(x['observations'] for x in summaries)
                ranks={}
                for f in raw.glob('runtime_proof*.jsonl'):
                    record=json.loads(f.read_text().splitlines()[0]);ranks[record['ep_rank']]=record
                assert set(ranks)=={0,1,2,3},name
                for rank,r in ranks.items():
                    assert r['prepare_finalize']=='DeepEPHTPrepareAndFinalize'
                    assert (r['dp_size'],r['tp_size'],r['ep_size'])==(2,2,4)
                    assert r['physical_visible_devices']=='4,5,6,7' and r['enable_ep'] and not r['dbo']
                    proof.append(dict(run=name,rank=rank,**{k:v for k,v in r.items() if k!='ep_rank'}))
                baseline=next(x for x in summaries if x['policy']=='vanilla')
                for x in summaries:
                    if x['policy']!='vanilla' and not x['policy'].startswith(method.lower()):continue
                    assert not x['instrumented'] and not x['fixed_output'] and x['observations']==384
                    ci=x['quality_delta_95ci_pp'];timing=x['paired_e2e_reduction_95ci_percent']
                    rows.append(dict(method=method,dataset=dataset,batch_cap_per_dp=batch,run=name,
                        policy=x['policy'],observations=x['observations'],unique_questions=x['unique_requests'],
                        accuracy_percent=x['accuracy_percent'],quality_delta_pp=x['paired_quality_delta_pp'],
                        quality_ci_low_pp=ci[0],quality_ci_high_pp=ci[1],
                        paired_e2e_reduction_percent=x['paired_e2e_reduction_median_percent'],
                        e2e_ci_low_percent=timing[0],e2e_ci_high_percent=timing[1],
                        e2e_ms_p50=x['e2e_ms_p50'],e2e_ms_mean=x['e2e_ms_mean'],
                        e2e_ms_p90=x['e2e_ms_p90'],e2e_ms_p99=x['e2e_ms_p99'],
                        ttft_ms_p50=x['ttft_ms_p50'],tpot_ms_p50=x['tpot_ms_p50'],
                        itl_ms_p99=x['itl_ms_p99'],output_tokens_mean=x['output_tokens_mean'],
                        throughput_tokens_s=x['cohort_throughput_output_tokens_s'],
                        vanilla_accuracy_percent=baseline['accuracy_percent'],
                        vanilla_e2e_ms_p50=baseline['e2e_ms_p50'],
                        timing_ci_unit=x['paired_e2e_ci_unit']))
    df=pd.DataFrame(rows);assert len(df)==42,len(df)
    df.to_csv(out/'quality_efficiency_primary.csv',index=False)
    df.to_csv(Path(__file__).with_name('QUALITY_EFFICIENCY_PARETO.csv'),index=False)
    pd.DataFrame(proof).to_csv(out/'runtime_activation.csv',index=False)
    summary=dict(completed_clean_engine_runs=12,all_clean_request_observations=all_count,
        primary_condition_policy_rows=len(df),primary_request_observations=int(df.observations.sum()),
        unique_evaluation_questions=256,warning='Reused questions across policies/batches/repetitions; not independent20K questions.',
        physical_gpus=[4,5,6,7],deep_ep_verified_ranks=4,
        e2e_definition='HOST_SUBMISSION_TO_ENGINE_CORE_TOKEN_READY; frontend receipt recorded separately',
        confidence='Paired cohort bootstrap conditional on one engine per condition; quality image-cluster bootstrap.',
        original_uncached_modes='Excluded from primary MoDES; retained as port-cost control.',
        kimi='NOT_RUN_NO_MATERIAL_NONTRIVIAL_SURVIVOR_ESTABLISHED')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for method in ['SERE','MoDES']:
        selected=df[(df.method==method)&(df.policy!='vanilla')]
        lines=['| Task | B/DP cap | Policy | Quality Δ pp [95% CI] | Paired request E2E reduction % [95% CI] |',
               '|---|---:|---|---:|---:|']
        for x in selected.itertuples():
            lines.append(f'| {x.dataset} | {x.batch_cap_per_dp} | {x.policy} | {x.quality_delta_pp:.3f} [{x.quality_ci_low_pp:.3f}, {x.quality_ci_high_pp:.3f}] | {x.paired_e2e_reduction_percent:.3f} [{x.e2e_ci_low_percent:.3f}, {x.e2e_ci_high_percent:.3f}] |')
        (out/f'{method.lower()}_table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
