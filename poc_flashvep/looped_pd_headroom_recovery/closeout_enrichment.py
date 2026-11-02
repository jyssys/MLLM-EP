"""Add final counts, uncertainty, and causal limits after terminal GPU evidence."""
import collections
import csv
import statistics
from datetime import datetime, timezone
from .common import *
from .reporting import FINAL, fmt, load, md
from .paired_analysis import interval


def csv_rows(name):
    path=ROOT/name
    if not path.exists():return []
    with path.open() as handle:return list(csv.DictReader(handle))


def main():
    runs=load(ROOT/'serving_runs.json',[])
    index={r['name']:r for r in runs}
    plan=load(ROOT/'state/confirmation_plan.json',{'runs':[],'selection':[]})
    ledger=load(ROOT/'state/confirmation_results.json',[])
    completed={r['name'] for r in ledger if r['passed']}
    by_model=[];paired=[];summary=[];parity=[]
    for model in ('ouro','huginn'):
        for arm in ('PD','DP'):
            names=[r[arm] for r in plan['runs'] if r['model']==model and r.get(arm)]
            rows=[index[n] for n in names if n in completed and n in index]
            summary.append([model,arm,len(rows),sum(r['offered'] for r in rows),sum(r['completed'] for r in rows),
                fmt(statistics.mean(r['target_QPS'] for r in rows)) if rows else None,
                fmt(statistics.mean(r['requests_per_s'] for r in rows)) if rows else None,
                fmt(statistics.mean(r['TTFT_ms_p95'] for r in rows)) if rows else None,
                fmt(statistics.mean(r['request_TPOT_ms_p95'] for r in rows)) if rows else None,
                fmt(statistics.mean(r['token_ITL_ms_p99'] for r in rows)) if rows else None,
                fmt(statistics.mean(r['SLO_goodput'] for r in rows)) if rows else None])
            for metric in ('requests_per_s','TTFT_ms_p95','request_TPOT_ms_p95','token_ITL_ms_p99','SLO_goodput'):
                values=[r[metric] for r in rows]
                lo,hi=interval(values)
                by_model.append(dict(model=model,arm=arm,metric=metric,independent_starts=len(values),
                    mean=statistics.mean(values) if values else None,CI95_low=lo,CI95_high=hi,
                    CI_method='Two-sided Student t, df=2' if len(values)==3 else 'NOT_AVAILABLE_NEEDS_THREE_STARTS',
                    scope='Mean of independent-start metrics. A mean p95 is not a pooled-request p95. These are finite-cohort results.'))
    from .analyze import table
    table('final_start_level_metrics.csv',by_model)
    for r in csv_rows('paired_start_uncertainty.csv'):
        paired.append([r['model'],r['metric'],r['independent_starts'],r['mean'],r['CI95_low'] or 'NOT_AVAILABLE',r['CI95_high'] or 'NOT_AVAILABLE'])
    for item in plan['runs']:
        for arm in ('PD','DP'):
            name=item.get(arm)
            if not name:continue
            folder=ROOT/'raw'/name
            work=load(folder/'confirmation_work.json',{})
            parity.append([item['model'],item['start'],arm,name,name in completed,
                work.get('actual_recurrent_rows',work.get('observed_recurrent_rows','See exact payload')),
                json.dumps(work,separators=(',',':'))])
    admission=[]
    initial=load(ROOT/'state/initial_plan_v3.json',{})
    for item in initial.get('ouro_controls',[]):
        r=index.get(item['pilot'],{})
        if not r:continue
        admission.append([item['label'],r.get('offered'),r.get('completed'),fmt(r.get('requests_per_s')),
            fmt(r.get('TTFT_ms_p95')),fmt(r.get('request_TPOT_ms_p95')),fmt(r.get('token_ITL_ms_p99'))])
    profiles=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir() or not folder.name.startswith(('profile_warm_','long_profile_','profile_warm_ouro_nsys')):continue
        p=load(folder/'profile_analysis.json')
        if not p:continue
        c=load(ROOT/'configs'/(folder.name+'.json'),{})
        g=load(ROOT/'p_d_gate_results.json',[])
        g=next((v for v in g if v['name']==folder.name),{})
        profiles.append([folder.name,c.get('count'),fmt(100*p['matched_wall_difference_fraction']) if p.get('matched_wall_difference_fraction') is not None else None,
            p.get('eligible_for_matched_clean',False),fmt(100*g['P_actual_kernel_busy_central']) if g.get('P_actual_kernel_busy_central') is not None else None,
            g.get('P_compute_gate'),g.get('D_SLO_gate'),g.get('D_credit_gate'),g.get('classification')])
    started=load(ROOT/'state/start.json',{})
    start=started.get('time',started.get('started',started.get('start_time')))
    if not isinstance(start,(int,float)):
        start=load(ROOT/'state/preservation.json',{})['time']
    now=time.time()
    elapsed=dict(start_epoch=start,closeout_epoch=now,elapsed_hours=(now-start)/3600,
        start_KST=datetime.fromtimestamp(start,timezone.utc).astimezone().isoformat(),
        closeout_KST=datetime.fromtimestamp(now,timezone.utc).astimezone().isoformat(),
        scope='Wall time includes implementation, controls, failed diagnostics, profiler export, and independent server starts. It is not summed GPU compute time.')
    write(ROOT/'state/elapsed_time.json',elapsed)
    text='\n## Final independent-start comparison\n\n'
    text+='All latency columns use milliseconds. Each displayed p95 or p99 is the mean of independent-start quantiles.\n'
    text+='The table does not pool requests across starts. The paired uncertainty table follows.\n\n'
    text+=md(['Model','Arm','Starts','Offered','Completed','Target req/s','Completed req/s','Mean TTFT p95','Mean TPOT p95','Mean ITL p99','Mean joint goodput'],summary)+'\n\n'
    text+='DP means two native mixed colocated replicas. PD means native 1P1D.\n'
    text+='Both arms use two GPUs, identical prompts, identical generation seeds, and paired arrival schedules.\n'
    text+='These finite N128 cohorts do not establish infinite-horizon sustainable capacity.\n\n'
    text+=md(['Model','Paired metric','Independent starts','Mean','95% CI lower','95% CI upper'],paired)+'\n\n'
    text+='The interval uses a two-sided Student-t distribution with two degrees of freedom.\n'
    text+='Three starts give wide uncertainty. A negative lower confidence bound for a nonnegative metric is a statistical interval, not negative physical latency.\n'
    text+='`final_start_level_metrics.csv` provides arm-specific start-level intervals.\n\n'
    text+='## Original Ouro 8192/64 admission controls\n\n'
    text+=md(['Supported control','Offered','Completed','Completed req/s','TTFT p95 ms','TPOT p95 ms','ITL p99 ms'],admission)+'\n\n'
    text+='These are single N24 starts with matched original arrival traces. They do not have independent-start confidence intervals.\n'
    text+='The controls change supported resource budgets. They do not reduce recurrent work or release unacknowledged state.\n'
    text+='The improved original control still must satisfy the frozen Decode tail limit. Admission relief alone is not a Prefill-heavy proof.\n\n'
    text+='## Actual selected-point device profiles\n\n'
    text+=md(['Profiled start','N','Wall change %','Perturbation qualified','Central P kernel union %','P compute gate','D SLO gate','D credit gate','Classification'],profiles)+'\n\n'
    text+='Each row uses that profiled start only. A profile must remain within the predeclared 5% wall-perturbation limit.\n'
    text+='Kernel union is recorded device activity. It is not SM occupancy, achieved bandwidth, or exclusive compute ownership.\n'
    text+='A failed perturbation gate leaves the trace diagnostic. It does not become a clean-latency measurement.\n'
    text+='The central window covers 20% to 90% of the actual arrival span. Arrival and drain remain separately visible.\n\n'
    text+='## Accuracy scope and full-work evidence\n\n'
    text+='`correctness_and_parity.md` lists the byte and output gates. Their denominators remain separate.\n'
    text+='`paired_repetitions.csv` reports PD/DP output parity within each independent pair.\n'
    text+='`state/cross_start_output_parity.json` reports PD output parity across independent arrival seeds.\n'
    text+='`full_depth_checks.csv` checks actual recurrent rows, not padded projection rows.\n'
    text+='No GSM8K, MATH-500, or original-HF accuracy result is measured in this campaign.\n\n'
    text+='## Workload and rate-selection limit\n\n'
    text+='The N24 Poisson pilot realized an arrival-interval rate about 24.86% above its nominal rate.\n'
    text+='A declared rule corrected the N128 nominal rates before any N128 measurement.\n'
    text+='Ouro confirmation uses 8192/32 at 2.5 nominal requests/s. Huginn confirmation uses 2560/32 at 1.5 nominal requests/s.\n'
    text+='The old nominal 2.0 and 1.2 pilot rates are not described as identical-rate independent repeats.\n'
    text+='Both paired policies receive the same revised schedule. No SLO threshold changes.\n'
    text+='The primary 6144/64 Ouro and 2560/32 Huginn sweeps remain in the complete configuration matrix.\n'
    text+='The secondary 3072/32 Huginn DP2 C2048 control first failed with a real CUDA OOM.\n'
    text+='One smaller legal KV-pool retry completed its byte/output gate and N24 pilot.\n'
    text+='The original failure and its source/configuration remain preserved.\n\n'
    text+='## Memory denominator correction\n\n'
    text+='NVML reports 85,520,809,984 bytes, or 79.6474609375 GiB, per physical GPU.\n'
    text+='CUDA reports 85,028,372,480 managed bytes, or 79.1888427734375 GiB.\n'
    text+='The 0.4586181640625-GiB difference is outside the CUDA-managed capacity.\n'
    text+='Earlier `percent_HBM` fields divide by CUDA-visible capacity. Their raw records remain unchanged.\n'
    text+='`physical_hbm_accounting.csv` provides both explicit denominators for every recorded owner endpoint.\n'
    text+='Do not add active KV bytes to the pool that already contains them.\n'
    text+='Do not add allocator totals to their contained tensor-storage categories.\n\n'
    text+='## E2E percentages and unmeasured device components\n\n'
    text+='`e2e_component_shares_summary.csv` reports additive host-observed phase percentages.\n'
    text+='Their weighted sum covers request E2E. The CSV does not report exclusive dispatch, GEMM, or DMA percentages.\n'
    text+='NIXL host-observed cumulative durations can overlap compute and other transfers.\n'
    text+='Exclusive dispatch/combine/kernel percentages, SM occupancy, and measured HBM bandwidth remain NOT_MEASURED when no valid device attribution exists.\n\n'
    text+='## Elapsed time and final artifacts\n\n'
    text+=f'This campaign used {elapsed["elapsed_hours"]:.3f} hours through analysis closeout.\n'
    text+='The exact timestamp pair remains in `state/elapsed_time.json`.\n'
    text+='`figure_manifest.json` identifies each plotted file, exact run namespace, and source hash.\n'
    text+='`config_matrix.csv` retains failed, superseded, and unrun conditions.\n'
    text+='`failures_and_unrun.md` explains their limits.\n'
    text+='`state/resource_closeout.json` records the final managed-burner restoration.\n'
    text+='`state/results_publication.json` records the publication branch and verified remote SHA.\n'
    text+='The active staged index and the external qualified runtime remain unchanged.\n'
    FINAL.write_text(FINAL.read_text()+text)
    write(ROOT/'state/closeout_enrichment.json',dict(time=now,independent_completed=len(completed),final=str(FINAL)))
    print(json.dumps(dict(start_level_rows=len(by_model),independent_completed=len(completed),elapsed_hours=elapsed['elapsed_hours'])))


if __name__=='__main__':main()
