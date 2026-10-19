"""Generate the numerical report tables from complete, reviewed artifacts."""
import numpy as np
from poc_depth_cdb_factorial.contract import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(str(c) for c in row)+' |' for row in rows])+'\n'

def ratio(r):
    return f"{r['speedup']:.3f}× [{r['CI95'][0]:.3f}, {r['CI95'][1]:.3f}]"

def pp(r):
    return f"{r['delta_pp']:+.3f} [{r['CI95_pp'][0]:+.3f}, {r['CI95_pp'][1]:+.3f}]"

def save(name,body):
    (ROOT/'reports'/name).write_text(body.strip()+'\n')

def main():
    decision=read(ROOT/'artifacts/final_decision.json')
    scaling=read(ROOT/'artifacts/scaling_summary.json')
    quality=read(ROOT/'artifacts/quality_summary.json')
    increments=read(ROOT/'artifacts/factorial_incremental.json')
    raw=read(ROOT/'artifacts/metrics_raw.json')
    assert len(scaling)==25 and all(r['n_paired_repeats']==5 for r in scaling)
    assert len(quality)==10 and all(r['N']==128 for r in quality)
    point={(r['C'],r['arm']):r for r in scaling}
    score={(r['dataset'],r['arm']):r for r in quality}
    qs=[]
    for arm in ARMS:
        for domain in ['gsm8k','math500']:
            r=score[domain,arm]
            qs.append([arm,domain,f"{r['correct']}/{r['N']}",f"{r['accuracy_pct']:.3f}%",r['correct_completed'],pp(r['vs_A0']),
                pp(r['vs_A3']) if arm=='A4' else '—',r['EOS'],r['stop'],r['truncated'],r['parse_fail'],r['total_generated_tokens']])
    qtext='# Five-arm natural-generation quality\n\n'
    qtext+='Every configuration uses the same 128 GSM8K and 128 MATH-500 questions.\n'
    qtext+='The corpus covers 128/1319 and 128/500 questions, respectively.\n'
    qtext+='The pre-registered subset can overlap old pilot questions. It is not an untouched held-out confirmation.\n'
    qtext+='All generations are greedy with native EOS, frozen stop sequences, and an 8192-token cap.\n\n'
    qtext+='GSM8K uses the CDB three-shot chain-of-thought prompt.\n'
    qtext+='MATH-500 uses the frozen zero-shot boxed-answer prompt.\n'
    qtext+='The Base checkpoint does not receive a chat template in this protocol.\n'
    qtext+='The answer parser is `parpp-final-v2`.\n\n'
    qtext+='The intervals describe paired accuracy differences in percentage points.\n'
    qtext+='An interval that includes zero does not establish quality equivalence.\n\n'
    qtext+='These descriptive intervals do not correct for multiple arm/domain comparisons.\n\n'
    qtext+=table(['Arm','Domain','Correct/N','Accuracy','Correct completed','Vs A0, pp [95% CI]','A4 vs A3, pp [95% CI]',
        'EOS','Configured stop','Cap','Parse fail','Output tokens'],qs)
    qtext+='\nA configured stop is a valid completion, not an output truncation.\n'
    qtext+='The native API exposes completion status, not a separate stop-reason enumeration.\n'
    qtext+='The wrapper classifies terminal EOS first, then a reached cap, then an earlier configured stop.\n'
    qtext+='The primary accuracy counts extracted answers that match the gold answer.\n'
    qtext+='`correct_completed` separately excludes cap-truncated outputs in the raw score file.\n\n'
    qtext+='The quality corpus is different from the 256-token bounded speed corpus.\n'
    qtext+='This report does not infer speed-cohort accuracy from this quality table.\n\n'
    qtext+='Quality runs use four C64 question cohorts per arm.\n'
    qtext+='Accuracy under the separate C8/C16/C32 speed operating points is `NOT_MEASURED`.\n'
    qtext+='The study does not assume that batch-dependent accuracy is identical across those operating points.\n\n'
    qtext+='## Generated work in the quality runs\n\n'
    qw=read(ROOT/'artifacts/quality_work_and_service.json')
    qtext+=table(['Arm','Questions','Output tokens','Scheduled loop work','Sum of four separate C64 serving intervals, s'],[
        [r['arm'],r['questions'],r['output_tokens'],r['scheduled_loop_work'],f"{r['sum_separate_C64_serving_s']:.3f}"] for r in qw])
    qtext+='\nThese are actual generated lengths, not fixed token counts or forced continuations.\n'
    qtext+='The summed serving intervals exclude intervening trials of other arms.\n'
    qtext+='They are not a C256 burst makespan or the main table latency denominator.\n\n'
    natural=read(ROOT/'artifacts/natural_quality_timing_diagnostic.json')
    qtext+=table(['Exploratory natural-workload comparison','Ratio of summed serving intervals [95% CI]'],[
        [r['comparison'],ratio(r)] for r in natural])
    qtext+='\nThe bootstrap resamples four matched but different 64-question cohorts, with one timing measurement each.\n'
    qtext+='It is not the primary five-repeat makespan bootstrap.\n'
    qtext+='These positive timing diagnostics do not establish quality-preserving or depth-only scheduling gains.\n'
    qtext+='Generated lengths, answers, and exit traces change between arms.\n'
    qtext+='A strong length-aware natural-workload control is also absent.\n\n'
    qp=read(ROOT/'artifacts/quality_output_parity.json')
    qtext+='## Full-answer output parity\n\n'+table(['Pair','Token-exact requests','Depth-exact requests','Free-running positional token agreement'],[
        [r['left']+' → '+r['right'],f"{r['token_exact']}/{r['N']}",f"{r['depth_exact']}/{r['N']}",
            f"{r['free_running_positional_token_agreement_pct']:.3f}%"] for r in qp])
    qtext+='\nPositional agreement compares two free-running outputs.\n'
    qtext+='It is not teacher-aligned next-token agreement or a logit-equivalence test.\n'
    qtext+='The raw parity file records first divergent tokens and changed output lengths.\n\n'
    qtext+='The short functional probe passed gate, work, and terminal-KV checks.\n'
    qtext+='Those checks do not isolate the cause of every long-answer divergence.\n'
    qtext+='Batch-dependent numerical changes and runtime-state effects remain possible.\n'
    qtext+='This experiment does not prove that BF16 rounding alone caused the differences.\n\n'
    flips=read(ROOT/'artifacts/quality_answer_flips.json')
    qtext+='## Paired answer changes\n\n'
    flip_rows=[]
    for a,b in [('A0','A1'),('A0','A2'),('A0','A3'),('A0','A4'),('A1','A2'),('A3','A4')]:
        for domain in ['gsm8k','math500']:
            group=[r for r in flips if r['left_arm']==a and r['right_arm']==b and r['dataset']==domain]
            flip_rows.append([a+' → '+b,domain,sum(r['direction']=='correct_to_wrong' for r in group),
                sum(r['direction']=='wrong_to_correct' for r in group)])
    qtext+=table(['Pair','Domain','Correct → wrong','Wrong → correct'],flip_rows)
    qtext+='\n`quality_answer_flips.json` records every changed answer, its parsed value, and completion status.\n'
    qtext+='The parity artifact also records first differing token and depth positions.\n'
    qtext+='Common-history greedy comparisons stop at the first differing token.\n'
    qtext+='Later positional comparisons do not share an identical token history.\n\n'
    qtext+='## Inherited quality warning\n\n'
    qtext+='The old 32-question pilot reported adaptive GSM8K 21/32 and MATH-500 19/32.\n'
    qtext+='Its fixed-R4 references were 22/32 and 24/32.\n'
    qtext+='The old fixed-minus-adaptive GSM8K difference was 3.125 pp, with 95% CI [−6.25, 12.5] pp.\n'
    qtext+='The old MATH-500 difference was 15.625 pp, with 95% CI [3.125, 28.125] pp.\n\n'
    qtext+='Those historical values are not the new five-arm scores.\n'
    qtext+='The experiment preserves the adaptive setting despite that warning.\n'
    qtext+='It does not claim fixed-R4 quality equivalence.\n'
    save('quality_five_arms.md',qtext)
    ltext='# Measured five-arm latency and absolute scaling\n\n'
    ltext+='Each table uses one fixed burst cohort.\nC denotes total eligible requests across the system.\n'
    ltext+='A0 uses physical GPU4. The other configurations use physical GPUs 4/5/6/7.\n\n'
    ltext+='The model, online gate, KV policy, local capacity, and request IDs match across configurations.\n'
    ltext+='Times are median cohort makespans over five paired repeats.\n'
    ltext+='Speedup is the matching A0 median makespan divided by the method median makespan.\n'
    ltext+='This is multi-request makespan scaling, not single-request latency speedup.\n\n'
    ltext+='Hindsight uses a precomputed future-informed placement. Its reference-trace acquisition cost is outside the serving makespan.\n'
    ltext+='The 95% intervals resample paired whole-cohort repeat IDs.\n\n'
    ltext+='They are conditional on the frozen request cohorts and do not include new-query population uncertainty.\n\n'
    for C in [4,8,16,32,64]:
        ltext+=f'## C{C}\n\n'
        ltext+=table(['Arm','Makespan, s','Speedup vs A0 [95% CI]','Median TTFT, ms','Mean TPOT, ms','P95 E2E, ms','Peak allocated/GPU, GiB'],[
            [a,f"{point[C,a]['median_makespan_s']:.3f}",ratio(point[C,a]),f"{point[C,a]['median_TTFT_ms']:.2f}",
                f"{point[C,a]['median_TPOT_ms']:.2f}",f"{point[C,a]['median_P95_E2E_ms']:.2f}",
                f"{point[C,a]['median_peak_HBM_GiB']:.3f}"] for a in ARMS])+'\n'
    ltext+='`metrics_raw.csv` contains requests/s, output tokens/s, ITL, goodput, queue time, actual output length, and loop work.\n'
    ltext+='The goodput rule permits max(10 seconds, 200 milliseconds per emitted token).\n'
    ltext+='This diagnostic is not a production latency guarantee.\n\n'
    ltext+='Peak allocated memory comes from PyTorch, not an instantaneous whole-device NVML maximum.\n'
    ltext+='Each memory cell is the median of the per-repeat maximum rank peak.\n'
    ltext+='Each latency cell also uses the median of its per-repeat statistic.\n'
    ltext+='The raw data also retain peak reserved memory and exact KV tensor bytes.\n\n'
    ltext+='Loop-token counts describe scheduled inference work, not extra setup kernels during lazy graph capture.\n'
    ltext+='The separate capture counters retain those setup events.\n\n'
    obs=read(ROOT/'artifacts/observer_parity.json')
    valid=[r['parity_matched_trace_delta_pct'] for r in obs if r['parity_matched_trace_delta_pct'] is not None]
    ltext+='## Observer validity\n\n'
    ltext+=f'{sum(r["observer_validity"]=="OUTPUT_AND_DEPTH_PARITY" for r in obs)}/{len(obs)} trace/clean comparisons preserve complete token and depth traces.\n'
    if valid:ltext+=f'The median observed trace-minus-clean time difference in those comparisons is {float(np.median(valid)):.3f}%.\n'
    ltext+='Each observer comparison has one paired run.\n'
    ltext+='It cannot isolate observer cost from run-to-run variation. The causal overhead estimate remains `PARTIAL`.\n\n'
    ltext+='Output-changing comparisons do not provide a validated observer-overhead estimate.\n'
    ltext+='Trace CUDA-event intervals never replace clean makespan in the primary tables.\n'
    save('latency_five_arms.md',ltext)
    itext='# Refill and hindsight interaction\n\n'
    itext+='The two switches act at different scheduling levels.\n'
    itext+='The native CDB scheduler is `LOCAL_ONLY`.\n'
    itext+='The outer controller assigns a GPU before prefill.\n'
    itext+='No runtime component migrates an active request or its KV.\n'
    itext+='See `source_scope_audit.md` for source locations and queue semantics.\n\n'
    itext+=table(['C','Hindsight, no-refill T(A1)/T(A2)','Hindsight, refill T(A3)/T(A4)',
        'Refill alone T(A1)/T(A3)','Refill under hindsight T(A2)/T(A4)'],[
        [r['C'],ratio(r['no_refill_hindsight']),ratio(r['refill_hindsight']),ratio(r['CDB_without_hindsight']),
            ratio(r['CDB_with_hindsight'])] for r in increments])
    itext+='\nEach cell is a speedup with its paired 95% interval.\nA value below 1.000× means slower execution.\n\n'
    itext+=table(['C','Mean log interaction','95% CI'],[[r['C'],f"{r['interaction_log']['mean']:+.6f}",
        f"[{r['interaction_log']['CI95'][0]:+.6f}, {r['interaction_log']['CI95'][1]:+.6f}]"] for r in increments])
    itext+='\nThe interaction is `log[T(A1)/T(A2)] − log[T(A3)/T(A4)]`.\n'
    itext+='A positive value means greater hindsight benefit without refill.\n'
    itext+='The calculation uses paired raw repeat times, not averages of unrelated ratios.\n\n'
    itext+='## Mechanism and limitations\n\n'
    itext+='The selected cache requires slot-homogeneous recurrent kernels.\n'
    itext+='Refill interleaves token chains but cannot combine arbitrary recurrent depths into one kernel batch.\n'
    itext+='The native coda-priority and front-reentry rules also change local batch sizes.\n'
    itext+='Thus a better depth-work balance need not give a matching GPU-time balance.\n\n'
    itext+='The trace scatter and batch histograms show the observed difference.\n'
    itext+='They do not measure actual expert or weight HBM traffic.\n'
    itext+='The synchronous selected runtime is not proof about the fastest possible CDB implementation.\n'
    itext+='The campaign does not validate the optional asynchronous path.\n\n'
    rank_data=read(ROOT/'artifacts/rank_time_analysis.json')
    itext+='## Actual rank work and time at C64\n\n'
    itext+=table(['Arm','Rank','Reference loop work','Actual loop work','Output tokens','Core stream, ms','All measured stage intervals, ms','Rank completion, s'],[
        [r['arm'],r['rank'],r['predicted_loop_work'],r['actual_loop_work'],r['generated_tokens'],
            f"{r['core_GPU_ms']:.2f}",f"{r['GPU_stream_stage_ms']:.2f}",f"{r['rank_completion_s']:.3f}"]
        for r in rank_data if r['C']==64])
    itext+='\nThese are trace-lane measurements of the observed trace workload.\n'
    itext+='Reference work comes from the earlier frozen requests, not the new counterfactual execution.\n'
    itext+='Thus reference work can differ from realized work after a batching change.\n'
    itext+='Stream-interval sums are not pure kernel time or exclusive cohort critical-path shares.\n\n'
    itext+='## C64 completion-critical rank breakdown\n\n'
    critical=[]
    for arm in ARMS:
        selected=[r for r in rank_data if r['C']==64 and r['arm']==arm]
        if selected:critical.append(max(selected,key=lambda r:r['rank_completion_s']))
    itext+=table(['Arm','Completion-critical rank','Prefill, ms','Core, ms','Exit-KV copy, ms','Prelude, ms','Coda, ms'],[
        [r['arm'],r['rank'],f"{r['prefill_GPU_ms']:.2f}",f"{r['core_GPU_ms']:.2f}",f"{r['exit_KV_copy_GPU_ms']:.2f}",
            f"{r['prelude_GPU_ms']:.2f}",f"{r['coda_GPU_ms']:.2f}"] for r in critical])
    itext+='\nEvery component in one row comes from the same completion-critical rank.\n'
    itext+='The table never adds maxima from different ranks.\n\n'
    itext+='## Comparison with v3\n\n'
    itext+='The historical 12.77% estimate concerned modeled P95 latency.\n'
    itext+='The historical 4.41% concerned mean measured P95 latency, approximately 1.046×.\n'
    itext+='Neither number was the primary cohort-makespan statistic used here.\n'
    itext+='`historical_v3_recomputed.json` recalculates both old statistics from preserved evidence.\n'
    itext+='The named sensitivity reuses both historical C64 placements.\n\n'
    itext+='The runtime keeps the new shared 10,240-token model-length limit, rather than the old 8192 limit.\n'
    itext+='New actual token and depth traces can also differ from the historical frozen reference.\n'
    itext+='These differences prevent a claim of identical experiment reproduction.\n'
    itext+='\n## Matched historical and arrival-order sensitivities\n\n'
    sens=read(ROOT/'artifacts/sensitivity_summary.json')
    itext+=table(['Control','C','No-refill makespan speedup','Refill makespan speedup','No-refill mean-P95 speedup','Refill mean-P95 speedup'],[
        [r['kind'],r['C'],ratio(r['no_refill']['makespan']),ratio(r['refill']['makespan']),
            ratio(r['no_refill']['P95_ratio_of_means']),ratio(r['refill']['P95_ratio_of_means'])] for r in sens])
    itext+='\nThe historical sensitivity reuses both the old generic and old hindsight assignments.\n'
    itext+='The shuffled-order sensitivity uses one seed, not a search for the most favorable order.\n'
    itext+='Its matched 4GPU ratios do not use an unmatched A0 denominator.\n'
    itext+='The mean-P95 ratios have their own paired bootstrap.\n\n'
    itext+='These sensitivity intervals resample repeats only. The old v3 intervals resampled both requests and repeats.\n'
    itext+='They do not transform a makespan confidence interval into a P95 confidence interval.\n'
    save('cdb_interaction_analysis.md',itext)
    ctext='# Concurrency, queues, and memory\n\n'
    ctext+='Each C is a finite simultaneous burst, not C requests per GPU.\n'
    ctext+='Every replica has the same 16-request resident limit and KV pool.\n'
    ctext+='A0 queues requests that exceed its local capacity.\n'
    ctext+='The reserve policy prevents those queued requests from allocating full resident KV immediately.\n\n'
    ctext+=table(['C','Arm','Mean core batch','Refill events','Depth-work CV','Rank-finish CV','Peak allocated/GPU, GiB'],[
        [C,a,f"{point[C,a]['median_core_batch']:.3f}",f"{point[C,a]['median_refill_events']:.0f}",
            f"{point[C,a]['median_depth_CV']:.4f}",f"{point[C,a]['median_finish_CV']:.4f}",
            f"{point[C,a]['median_peak_HBM_GiB']:.3f}"] for C in [4,8,16,32,64] for a in ARMS])
    ctext+='\nAll cache tensors together occupy 24.0234375 GiB per replica.\n'
    ctext+='The nominal request-row pool is 24 GiB. Padding accounts for the remaining 24 MiB.\n\n'
    ctext+='Peak allocated memory excludes some non-PyTorch CUDA context allocations.\n'
    ctext+='Peak reserved memory appears separately in the raw measurements.\n'
    ctext+='The configured launch limit is not the achieved recurrent batch size.\n'
    ctext+='A deficit from 16 slots is not a direct measurement of GPU hardware idle.\n\n'
    parity_data=read(ROOT/'artifacts/output_parity.json')
    diagnostics=[]
    for C in [4,8,16,32,64]:
        for arm in ARMS:
            trials=[r for r in raw if r['kind']=='speed' and not r['trace'] and r['C']==C and r['arm']==arm]
            traces=[r for r in rank_data if r['C']==C and r['arm']==arm]
            critical_rank=max(traces,key=lambda r:r['rank_completion_s'])
            matches=[r['token_exact_pct'] for r in parity_data if r['kind']=='speed' and r['C']==C and r['left']=='A0' and r['right']==arm]
            diagnostics.append([C,arm,int(np.median([r['output_tokens'] for r in trials])),
                int(np.median([r['actual_recurrent_steps'] for r in trials])),
                f"{float(np.median([r['queue_P95_ms'] for r in trials])):.2f}",
                f"{critical_rank['core_GPU_ms']:.2f}",f"{100. if arm=='A0' else float(np.median(matches)):.2f}%",
                f"{float(np.median([sum(r['rank_without_local_request_tail_s']) for r in trials])):.3f}",'UNRESOLVED'])
    ctext+='## Operating-point diagnostics\n\n'
    ctext+=table(['C','Arm','Output tokens','Scheduled loop work','P95 queue, ms','Trace critical-rank core, ms',
        'Token-exact vs A0','Rank-summed no-local-request tail, s','Whole-GPU idle with backlog'],diagnostics)
    ctext+='\nClean statistics use medians across five paired repeats.\n'
    ctext+='Token parity is the median fraction of exact request outputs versus the paired A0 run.\n'
    ctext+='The core stream interval uses one separate trace run.\n'
    ctext+='The rank-summed tail adds independent rank resource seconds, not serial wall-clock delay.\n'
    ctext+='The observer report identifies trace runs whose output drift prevents a clean overhead estimate.\n\n'
    ctext+='## Waiting and actual GPU work\n\n'
    ctext+='`rank_time_analysis.csv` records per-rank core stream milliseconds and request/loop totals.\n'
    ctext+='The CUDA-event timeline records prefill, recurrent, prelude, and coda intervals.\n'
    ctext+='Post-completion tails identify ranks with no remaining local request while another rank continues.\n'
    ctext+='Sampled reservation blocks and admission state distinguish memory pressure from resident-cap pressure where possible.\n\n'
    ctext+='An unoccupied compute stream is not necessarily an idle whole GPU.\n'
    ctext+='Exact whole-GPU idle-with-backlog duration remains `UNRESOLVED`.\n'
    ctext+='Every required performance cell contains a live measured time.\n'
    ctext+='No simulated time replaces an unavailable execution.\n\n'
    ctext+='## Optional operating points\n\n'
    ctext+='C128/C256 are `NOT_RUN` optional sensitivities.\n'
    ctext+='The study prioritizes five repeats of every required configuration and 128 quality questions per domain.\n'
    ctext+='The fixed local memory limit would queue additional requests.\n'
    ctext+='No measured C128/C256 latency or memory-stability claim follows from that structural expectation.\n'
    ctext+='Rolling and Poisson arrivals are outside this completed burst factorial.\n'
    ctext+='They are not inferred from burst makespan ratios.\n'
    save('concurrency_and_memory.md',ctext)
    ftext='# CDB Refill × Cross-GPU Hindsight: final report\n\n'
    ftext+='## Verdict\n\n'
    ftext+=f'The final verdict is **`{decision["verdict"]}`**.\n\n'
    ftext+='\n'.join('- '+item for item in decision['findings'])+'\n\n'
    ftext+='All 125 primary speed cohorts and all 1280 quality generations ran on the specified physical GPUs.\n'
    ftext+='No weak or negative gain stopped a required configuration.\n'
    ftext+='No RL policy, live KV migration, gate change, or new serving engine is part of this study.\n\n'
    ftext+='## Five-row main table, C64\n\n'
    ftext+=table(['Method','GPUs','GSM8K accuracy, correct/N','MATH-500 accuracy, correct/N','Makespan, s','Speedup vs A0 [95% CI]','Peak allocated/GPU, GiB'],[
        [ARMS[a]['label'],len(ARMS[a]['gpus']),f"{score['gsm8k',a]['accuracy_pct']:.3f}% ({score['gsm8k',a]['correct']}/128)",
            f"{score['math500',a]['accuracy_pct']:.3f}% ({score['math500',a]['correct']}/128)",
            f"{point[64,a]['median_makespan_s']:.3f}",ratio(point[64,a]),f"{point[64,a]['median_peak_HBM_GiB']:.3f}"] for a in ARMS])
    ftext+='\nThe speed column uses the identical C64 bounded-progress request cohort and the same adaptive gate and KV policy.\n'
    ftext+='It divides the 1GPU A0 cohort makespan by the method cohort makespan.\n'
    ftext+='It is multi-request makespan scaling, not single-request latency speedup.\n\n'
    ftext+='Accuracy uses a separate identical full-answer cohort across all five configurations, with an 8192-token cap.\n'
    ftext+='Speed uses a 256-token natural-EOS cap. The two workloads do not share a latency denominator.\n'
    ftext+='Memory means PyTorch peak allocated memory, not the NVML maximum for the whole device.\n\n'
    ftext+='## Scaling\n\n'
    ftext+=table(['Method','C8 speedup','C16 speedup','C32 speedup','C64 speedup'],[
        [ARMS[a]['label']]+[f"{point[C,a]['speedup']:.3f}×" for C in [8,16,32,64]] for a in ARMS])
    ftext+='\nEach C uses its own matching A0 baseline.\nThe latency report includes raw seconds, confidence intervals, TTFT, TPOT, and P95 E2E.\n\n'
    ftext+='## Factorial contributions\n\n'
    ftext+=table(['C','Hindsight without refill','Hindsight with refill','Refill without hindsight','Refill with hindsight'],[
        [r['C'],ratio(r['no_refill_hindsight']),ratio(r['refill_hindsight']),ratio(r['CDB_without_hindsight']),ratio(r['CDB_with_hindsight'])]
        for r in increments])
    ftext+='\nThe matched 4GPU ratios isolate the measured incremental effects.\n'
    ftext+='They must not be replaced by the much larger A0-normalized four-GPU scaling value.\n'
    ftext+='The interaction report retains paired log-interaction intervals and the historical matched-placement sensitivity.\n\n'
    ftext+='## Direct answers\n\n'+ '\n\n'.join(item.replace('\n\n','\n\n    ') for item in decision['direct_answers'])+'\n\n'
    ftext+='## Validity and limits\n\n'
    ftext+='The five-arm short probe passes token/depth parity and 6408 exact terminal-KV copy checks.\n'
    ftext+='The earlier independent official-reference comparison remains `REFERENCE_MATCH_PARTIAL`.\n\n'
    ftext+='Full-answer parity and paired accuracy differences appear in `quality_five_arms.md`.\n'
    ftext+='Changed batch composition can change future outputs and actual depth work.\n'
    ftext+='The study reports those changes instead of assuming fixed execution traces across placements.\n\n'
    ftext+='The source audit proves local versus global scheduling scope.\n'
    ftext+='It does not prove that every bad refill result comes from one kernel or numerical cause.\n'
    ftext+='The selected synchronous, slot-homogeneous configuration is not every possible CDB implementation.\n\n'
    ftext+='The LPT hindsight signal combines future output length and depth. It is not a pure depth-only control.\n'
    ftext+='The burst speed cohort is historical and heavily capped. It is not a fresh full-answer serving benchmark.\n\n'
    ftext+='The five repeats provide limited timing uncertainty estimates.\n'
    ftext+='The question bootstrap does not include all benchmark-selection or parser uncertainty.\n'
    ftext+='Exact whole-GPU idle attribution remains unresolved where other CUDA streams can execute.\n'
    ftext+='Optional C128/C256, new RL training, and deployment-quality validation are not part of this campaign.\n\n'
    ftext+='## Evidence and next action\n\n'+decision['next_action']+'\n\n'
    ftext+='See `reproduce.md`, `timing_contract.md`, `source_scope_audit.md`, and the raw artifact manifests.\n'
    ftext+='The final ownership audit checks preserved dirty work and restored burners on physical GPUs 4/5/6/7.\n'
    ftext+='Branch: `flashvep/depth-cdb-factorial-hindsight-poc`.\n'
    ftext+='The final user response records the published commit SHA.\n'
    ftext+='\n## Figures\n\n'
    ftext+='- [Five-arm scaling](../figures/five_arm_scaling.png).\n'
    ftext+='- [Incremental hindsight and uncertainty](../figures/incremental_hindsight.png).\n'
    ftext+='- [C32 rank timeline](../figures/rank_timeline_C32.png).\n'
    ftext+='- [C64 rank timeline](../figures/rank_timeline_C64.png).\n'
    ftext+='- [Depth work versus measured core intervals](../figures/depth_work_vs_gpu_time.png).\n'
    ftext+='- [Active core batch distribution](../figures/no_refill_refill_batch_histogram.png).\n'
    ftext+='- [Five-arm accuracy](../figures/five_arm_quality.png).\n'
    save('final_report.md',ftext)
    print('REPORT_TABLES_WRITTEN')

if __name__=='__main__':main()
