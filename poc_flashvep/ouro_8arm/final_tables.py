"""Build reviewed evidence tables. Preserve every raw run and its original flags."""
import csv
import numpy as np
from .contract import *
from .effective_sources import subset_views
from .serving_analysis import table


def csv_rows(name):
    return list(csv.DictReader((ROOT/name).open()))


def markdown(path,columns,rows,preamble):
    text=preamble+'\n\n| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    text+=''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
    (ROOT/path).write_text(text)


def qualification(arm):
    if arm in ['T4','TD','native_TP4']:return 'NUMERICALLY_UNQUALIFIED_EXPLORATORY'
    if arm in ['SD','LD','OD']:return 'CUSTOM_POLICY_VALIDATED_WITH_NATIVE_BACKEND_DRIFT'
    if arm=='native_TP1':return 'PRODUCTION_BACKEND_CONTROL'
    return 'BOUNDED_NUMERICAL_AND_400_OUTPUT_PARITY'


def ratio_interval(x,y,groups):
    x=np.asarray(x);y=np.asarray(y);rng=np.random.default_rng(20261021)
    indices=np.concatenate([rng.choice(group,(5000,len(group)),replace=True) for group in groups],axis=1)
    samples=x[indices].sum(1)/y[indices].sum(1)
    return float(x.sum()/y.sum()),*map(float,np.quantile(samples,[.025,.975]))


def main():
    stage=read(ROOT/'state/continuation_stage.json');assert stage['status'] in ['NEEDS_REVIEW','FINAL_REVIEW_COMPLETE']
    views=subset_views();assert all(not v['missing_cohorts'] for v in views.values())
    quality={(r['run'].removeprefix('quality400_'),r['dataset']):r for r in csv_rows('accuracy_by_arm.csv') if r['run'].startswith('quality400_')}
    paired={(r['run'],r['reference'],r['dataset']):r for r in csv_rows('paired_accuracy_deltas.csv')}
    serving={r['run'].removeprefix('quality400_'):r for r in csv_rows('subset400_serving.csv')}
    replay={r['arm']:r for r in csv_rows('replay_summary.csv') if r['family']=='TRACE_LOCKED_DYNAMIC_TOKENS' and r['C']=='4'}
    results=[];quality_rows=[];matrix=[]
    for arm in list(ARMS)+['native_TP1','native_TP4']:
        mode_single='SD' if arm in ['SD','TD','LD','OD'] else 'S4'
        if arm.startswith('native_'):mode_single='native_TP1'
        for domain in ['gsm8k','math500']:
            row=quality[arm,domain];references=[]
            for reference in ['S4',mode_single]:
                if arm==reference:references.append('0.000 (reference)')
                else:
                    p=paired['quality400_'+arm,'quality400_'+reference,domain]
                    references.append(f"{float(p['delta_accuracy_pp']):+.3f} [{float(p['CI95_low_pp']):.3f}, {float(p['CI95_high_pp']):.3f}]")
            quality_rows.append([arm,domain,row['correct']+'/'+row['N'],f"{float(row['accuracy_percent']):.3f}",*references,
                row['completed_correct']+'/'+row['N'],row['parse_failures'],row['truncated'],qualification(arm)])
        service=serving[arm];same=serving[mode_single]
        speed=float(same['cohort_makespan_sum_s'])/float(service['cohort_makespan_sum_s'])
        e2e=float(same['mean_E2E_s'])/float(service['mean_E2E_s'])
        overall=float(serving['S4']['cohort_makespan_sum_s'])/float(service['cohort_makespan_sum_s'])
        item=dict(arm=arm,GPUs=int(service['GPU_count']),quality_GSM8K_correct=int(quality[arm,'gsm8k']['correct']),
            quality_GSM8K_N=300,quality_MATH_correct=int(quality[arm,'math500']['correct']),quality_MATH_N=100,
            service_cohort_sum_s=float(service['cohort_makespan_sum_s']),task_cohort_speedup_vs_mode_single=speed,
            task_mean_E2E_speedup_vs_mode_single=e2e,task_cohort_overall_gain_vs_S4=overall,
            clean_C4_makespan_s=float(replay[arm]['median_cohort_makespan_s']) if arm in replay else None,
            clean_C4_speedup_vs_mode_single=float(replay[arm]['speedup_cohort_vs_mode_Single']) if arm in replay else None,
            qualification=qualification(arm),scope='MEASURED. Task clocks include observers and can differ in generated work. Clean C4 uses frozen output/depth replay, not task quality.')
        results.append(item)
        if arm in ARMS:
            matrix.append([arm,'1' if arm in ['S4','SD'] else '4',f"{item['quality_GSM8K_correct']}/300",f"{item['quality_MATH_correct']}/100",
                f"{speed:.3f}x",f"{item['clean_C4_speedup_vs_mode_single']:.3f}x",qualification(arm)])
    table(ROOT/'reviewed_eight_arm_results.csv',results)
    markdown('quality_results.md',['Arm','Dataset','Correct/N','Accuracy %','Delta vs S4 pp [95% CI]',
        'Delta vs mode Single pp [95% CI]','Completed-correct/N','Parse failures','Truncated','Reviewed scope'],quality_rows,
        '# Actual free-running quality\n\nEach primary arm used the same 300 GSM8K and 100 MATH questions. The cap was 4096 tokens.\n'
        'Accuracy counts a correct extracted answer even if the generation reaches its cap. Completed-correct excludes those outputs.\n'
        'Intervals use 5000 question-paired bootstrap draws. Identical observed scores do not establish population equivalence.\n'
        'Native controls are separate engine comparisons. Their mode-Single reference is native TP1. Raw source flags remain unchanged.')
    markdown('eight_arm_matrix.md',['Arm','GPUs','GSM8K correct/N','MATH correct/N','Task cohort speedup vs mode Single',
        'Clean C4 replay speedup vs mode Single','Qualification'],matrix,
        '# Eight-arm final evidence matrix\n\nAll eight primary arms completed 400 actual free-running answers.\n'
        'Task ratios use the sum of 101 service-cohort durations. They exclude model load and request-bound graph preparation.\n'
        'Static task scheduling differs between PP FIFO and Single/TP waves. Outputs can differ across backends.\n'
        'Clean C4 uses identical frozen prompt/output IDs and five measured repetitions. Dynamic depth is also trace-locked.\n'
        'A four-GPU ratio against one GPU does not establish four-GPU resource efficiency. Historical TP numerical failure remains FAIL.')
    # Resample complete paired request cohorts. Preserve within-cohort queue dependencies.
    cohorts={}
    for arm,view in views.items():
        times=[]
        for c in view['cohorts']:
            value=read(ROOT/c['timing_file'])
            times.append(value['makespan_s'] if 'makespan_s' in value else value['end']-value['origin'])
        cohorts[arm]=times
    plan=read(ROOT/'evaluation_subset400.json');byid={r['id']:r for r in dataset(False)}
    groups=[np.array([i for i,c in enumerate(plan['cohorts']) if byid[c['ids'][0]]['dataset']==d]) for d in ['gsm8k','math500']]
    ratios=[]
    for arm in views:
        refs=['SD' if arm in ['SD','TD','LD','OD'] else 'S4']
        if arm.startswith('native_'):refs=['native_TP1']
        if arm=='O4':refs.append('L4')
        if arm=='OD':refs.append('LD')
        for reference in refs:
            point,low,high=ratio_interval(cohorts[reference],cohorts[arm],groups)
            ratios.append(dict(arm=arm,reference=reference,N_requests=400,N_cohorts=101,cohort_sum_speedup=point,
                CI95_low=low,CI95_high=high,draws=5000,seed=20261021,
                qualification=qualification(arm),
                scope='Request-cohort paired bootstrap, stratified by dataset. Captures cohort composition uncertainty from one task pass, not repeat-to-repeat timing stability.'))
    table(ROOT/'task_cohort_paired_ratios.csv',ratios)
    performance=[]
    for arm,s in serving.items():
        performance.append([arm,s['GPU_count'],f"{float(s['cohort_makespan_sum_s']):.3f}",f"{float(s['mean_E2E_s']):.3f}",
            f"{1000*float(s['mean_TTFT_s']):.3f}",f"{1000*float(s['mean_TPOT_s']):.3f}",
            f"{float(s['P95_E2E_s']):.3f}",s['output_tokens'],f"{float(s['output_tokens_per_s']):.3f}",
            f"{float(s['output_tokens_per_s_per_GPU']):.3f}"])
    markdown('serving_summary.md',['Arm','GPUs','Cohort sum s','Mean E2E s','Mean TTFT ms','Mean TPOT ms','P95 E2E s',
        'Generated tokens','Output tok/s','Output tok/s/GPU'],performance,
        '# Actual task-serving measurements\n\nThese are actual 400-question task runs. They are not observer-free hardware headlines.\n'
        'Custom T4, TD, and native TP4 remain NUMERICALLY_UNQUALIFIED_EXPLORATORY. These rows do not establish a qualified TP architecture baseline.\n'
        'Cohort sum covers 101 measured service windows. It is not whole-command wall time or model-cold latency.\n'
        'The custom clock excludes request-bound graph preparation. Earlier Static pilots lack full preparation-inclusive coverage.\n'
        'Use extension286_serving.csv for the fully instrumented 286-question preparation control. Native graphs use a different lifecycle.\n'
        'Generated work differs between engines. Native TP1/TP4 use packed FA3, paged KV, and optimized graphs.\n'
        'The custom family uses request-local B1 stages. Custom TP also uses FP32 partial-output reductions and separate projections.\n'
        'Use replay_summary.csv for identical-work five-repeat results and request-level paired intervals.\n'
        'Use task_cohort_paired_ratios.csv for paired cohort-composition intervals. These intervals do not measure repeat-to-repeat noise.')
    memory=[]
    all_memory=csv_rows('memory_by_rank.csv')
    for arm,view in views.items():
        source_cohorts={(c['source_run'],str(c['cohort']).zfill(3)) for c in view['cohorts']}
        source_runs={c['source_run'] for c in view['cohorts']}
        selected=[m for m in all_memory if (m['run'],m['cohort']) in source_cohorts or
                  (arm.startswith('native_') and m['run'] in source_runs and m['cohort']=='native_setup')]
        for rank in range(1 if arm in ['S4','SD','native_TP1'] else 4):
            rr=[r for r in selected if int(r['rank'])==rank]
            def largest(key):
                values=[int(r[key]) for r in rr if r[key]]
                return max(values) if values else None
            memory.append(dict(arm=arm,rank=rank,physical_GPU=rank+4,weight_bytes=largest('weight_bytes'),
                KV_reserved_bytes=largest('KV_reserved_bytes'),service_or_setup_peak_allocated_bytes=largest('peak_bytes'),
                preparation_peak_allocated_bytes=largest('preparation_peak_bytes'),allocator_reserved_bytes=largest('reserved_bytes'),
                scope='Native rows: setup/graph peak only, not task peak. Custom rows: maximum service-period allocation over selected cohorts. No continuous NVML peak or OOM capacity sweep.'))
    table(ROOT/'reviewed_memory.csv',memory)
    profile_rows=[];GPU_rows=[]
    for path in sorted((ROOT/'timelines').glob('*/summary.json')):
        p=read(path);profile_rows.append(dict(label=p['label'],window_ms=p['window_ms'],
            actual_different_request_kernel_overlap_ms=p['different_request_actual_kernel_overlap_ms'],
            simultaneous_tagged_ranks=p['max_simultaneous_tagged_kernel_ranks'],KV_protocol_intervals=p['KV_protocol_interval_count'],
            scope=p['lane']))
        GPU_rows.extend(dict(profile=p['label'],**r,scope=p['lane']) for r in p['ranks'])
    table(ROOT/'profile_summary.csv',profile_rows);table(ROOT/'gpu_activity.csv',GPU_rows)
    stage_rows=[]
    for folder in sorted((ROOT/'raw').glob('profile_*')):
        if not (folder/'complete.json').exists():continue
        for file in sorted(folder.glob('cohort_*_rank*.json')):
            record=read(file);events=record.get('events',[])
            if not events:continue
            for loop in sorted({r['loop'] for r in events}):
                local=[r for r in events if r['loop']==loop]
                # Initial ready_s=0 is a sentinel, not an observed admission time.
                waits=[1000*(r['dispatch_s']-r['ready_s']) for r in local if r['ready_s']>0]
                stage_rows.append(dict(profile=folder.name,rank=record['rank'],physical_GPU=record['rank']+4,loop=loop,
                    stages=len(local),observed_ready_intervals=len(waits),
                    mean_ready_to_dispatch_ms=float(np.mean(waits)) if waits else None,
                    P95_ready_to_dispatch_ms=float(np.quantile(waits,.95)) if waits else None,
                    negative_ready_intervals=sum(v<0 for v in waits),
                    mean_CUDA_copy_interval_ms=float(np.mean([r['copy_ms'] for r in local])),
                    mean_CUDA_compute_interval_ms=float(np.mean([r['compute_ms'] for r in local])),
                    mean_CUDA_head_export_interval_ms=float(np.mean([r['head_export_ms'] for r in local])),
                    scope='HEAVY_TRACE. Host ready-to-dispatch includes IPC/control/queue waiting. Initial zero sentinel excluded. CUDA intervals can include launch gaps and waits. No additive E2E breakdown.'))
    table(ROOT/'stage_timing_summary.csv',stage_rows)
    write(ROOT/'reviewed_results.json',dict(arms=results,quality_denominators=dict(gsm8k=300,math500=100),
        quality_scope='Actual free-running outputs. Same 400 requests per primary arm. Not the full public test suites.',
        qualifiers=['Historical strict TP numerical gate remains FAIL.','Dynamic reference has measured long-run native/custom exit drift.',
            'Task observers are charged. Clean replay uses frozen token IDs and future depth traces.',
            'Kernel service sums are not exclusive E2E component shares.','No optimized PP or optional DP4 capacity baseline was implemented.']))
    print('FINAL_TABLES',len(results),'engines',len(profile_rows),'profiles',flush=True)


if __name__=='__main__':main()
