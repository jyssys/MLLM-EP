"""Report measured primitives and unresolved gates without inferred speedups."""
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
from .common import *

FINAL=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_gated_loop_continuation_final_summary.md'


def load(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []


def fmt(value):
    if value is None or value=='':return 'NOT_MEASURED'
    if isinstance(value,float):return str(value)
    return str(value).replace('|','/')


def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(fmt(v) for v in record)+' |' for record in data])


def main():
    initialize();out=[]
    def add(text):out.extend((text,''))
    add('# Adaptive idle-gated Loop continuation: final evidence and unresolved gates')
    add('Campaign status: '+('GPU_EXPERIMENTS_CLOSED' if (ROOT/'state/CAMPAIGN_CLOSED.json').exists() else 'IN_PROGRESS_DRAFT_NOT_FINAL')+'.')
    add('## 1. Decision')
    add('**Ouro: INCONCLUSIVE_SLACK. Huginn: INCONCLUSIVE_SLACK. Common method: INCONCLUSIVE_COMPARISON.**')
    add('This campaign did not demonstrate the requested online Adaptive Loop-Split.\n'
        'The runtime completed exact forced heterogeneous Loop continuations.\n'
        'It did not select those cuts from a measured, validated online Safe Slack budget.\n'
        'No reported primitive or forced-cut request establishes an adaptive serving gain.\n'
        'These results are not proof that useful D headroom cannot exist.')
    decisions={model:dict(verdict='INCONCLUSIVE_SLACK',Gate_1='INCONCLUSIVE_SLACK',
        Gate_2='NOT_TESTED_ONLINE_FORCED_PROTOCOL_ONLY',Gate_3='NOT_TESTED_ADAPTIVE_DECODE_PROTECTION',
        Gate_4='NOT_TESTED_NET_SERVING_BENEFIT',Gate_5='INCONCLUSIVE_COMPARISON',
        actual_online_adaptive_offloads='NOT_MEASURED_NO_POLICY_RUN',
        full_prompt_mode='UNCONFIRMED',token_tiled_mode='NOT_RUN',
        scope='Missing qualified evidence is not a negative performance measurement.') for model in ('ouro','huginn')}
    write(ROOT/'final_decisions.json',decisions)
    add(table(['Gate','Ouro','Huginn'],[
        ['1: same-run P pressure and deadline-safe D budget','INCONCLUSIVE_SLACK','INCONCLUSIVE_SLACK'],
        ['2: variable nonzero online accepted Loop counts','NOT_TESTED','NOT_TESTED'],
        ['3: adaptive Decode protection','NOT_TESTED','NOT_TESTED'],
        ['4: net TTFT or goodput gain','NOT_TESTED','NOT_TESTED'],
        ['5: Dynamic Loop versus Dynamic Token','INCONCLUSIVE_COMPARISON','INCONCLUSIVE_COMPARISON']]))
    add('The failed Gate 1 evidence contract stopped method expansion.\n'
        'No new Dynamic Scheduler should use these unqualified budgets as a safety guarantee.\n'
        'Generic logical-layer execution at identical tied-weight boundaries remains EQUIVALENT_MECHANISM.\n'
        'No current record establishes a Loop-specific serving advantage.')
    add('## 2. Provenance, clock, and isolation')
    p=load(ROOT/'state/preservation.json',{})
    add(f'Publication parent: `{BASE}`. Branch: `{BRANCH}`.\n'
        f'Preserved checkout HEAD: `{p.get("worktree_HEAD")}`.\n'
        f'Preserved private runtime HEAD: `{p.get("private_HEAD")}`.\n'
        'The private runtime is `/home/esjung/external/looped-pd-abc-rlt`.\n'
        'The source archives and configuration inventory identify each executed revision.\n'
        'The new branch uses a separate Git index. The original checkout and staged entries remain unchanged.')
    add('The campaign reuses qualified Native graph replay, Token/Loop connectors, NIXL handoff, and exact state validators.\n'
        'No external runtime source or checkpoint changed.\n'
        'The `ref_loopserve` code uses a different runtime interface. No compatible boundary-negotiation implementation was established for direct reuse.\n'
        'Independent D-local probes and cross-depth batching are not substitutes for an actual adaptive P-to-D continuation.')
    add('The fixed wall-clock budget started at 2026-11-06 13:05:11 KST.\n'
        'The final deadline is 21:05:11 KST. The GPU experiment deadline is 20:15:11 KST.\n'
        'The controller completed its configured calibration list before the 18:58:13 resumption inspection.\n'
        'The monitor and deadline guard remained active, but they did not schedule further research.\n'
        'This idle interval remains part of the budget. It is not GPU experiment time.\n'
        'See [the operational incident](ouro_huginn_adaptive_idle_gated_loop_continuation/resumption_and_burner_incident.md).')
    add('The host has eight H100 80GB GPUs. Each scientific job uses an isolated two-GPU pair.\n'
        'All approved burners stop during those measurements.\n'
        'Heavy CPU trace analysis uses the same exclusive lane and does not overlap a GPU experiment.\n'
        'Driver version: 570.211.01. The collector uses CUDA 12.8 headers and link-search paths.\n'
        'The environment inventory retains topology, UUIDs, and process ownership.')
    software=load(ROOT/'software_versions.json',{})
    add(f'Python: `{software.get("Python","NOT_MEASURED")}`.\n'
        f'Torch: `{software.get("Torch","NOT_MEASURED")}`. Torch CUDA build: `{software.get("Torch_CUDA","NOT_MEASURED")}`.\n'
        'See `software_versions.json` for NIXL, Triton, CUPTI, and other installed package versions.\n'
        'The environment also contains the CUDA CUPTI package version 12.9.79.\n'
        'Per-worker resolved CUPTI library maps were not recorded. Cross-version effects remain untested.\n'
        'No export failure is attributed to a version difference without that test.')
    add('## 3. Models, requests, and frozen SLO')
    add(table(['Model','Precision / depth','Input / output','Low / medium / heavy QPS','Physical P/D pair','P/D slots'],[
        ['ByteDance/Ouro-1.4B','Original BF16 / R4','4096 / 32','1.23460875 / 3.4569045 / 6.356','0 / 1','2 / 8'],
        ['tomg-group-umd/huginn-0125','Original BF16 / R32','2048 / 16','0.37903355455 / 1.06129395274 / 1.971','4 / 5','2 / 8']]))
    add('The primary pool contains 128 distinct ShareGPT/LongBench controlled-prefix prompt IDs per model.\n'
        'The campaign reuses that text pool. It does not establish unseen-text generalization.\n'
        'Prefix caching is disabled. Output caps ignore EOS.\n'
        'Calibration uses eight disjoint text IDs per model and serial requests.\n'
        'Poisson offsets regenerate from each recorded QPS and seed. Paired controls retain identical offsets.\n'
        'No N512 campaign ran. No natural-EOS accuracy or original-HF equivalence test ran.')
    add('Pinned Ouro revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.\n'
        'Pinned Huginn revision: `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.\n'
        'The native contract inventory records actual precision, depth, revision, and prefix-cache state.\n'
        'Source defaults alone do not substitute for that runtime record.')
    slos=load(ROOT/'configs/frozen_SLO.json')
    add(table(['Model','TTFT ms','Request TPOT ms','Additional-work TPOT p95 ms','ITL p99 ms'],[
        [m,slos[f'{m}_L{length}']['TTFT_ms'],slos[f'{m}_L{length}']['TPOT_ms'],
         .9*slos[f'{m}_L{length}']['TPOT_ms'],slos[f'{m}_L{length}']['ITL_ms']]
        for m,length in (('ouro',4096),('huginn',2048))]))
    add('The Native-relative ITL p99 design target remains at most 5% degradation.\n'
        'The observer gate separately requires an absolute difference within 5% for all fixed local-tail windows.\n'
        'The windows cover TPOT p95, ITL p95/p99, and maximum ITL.\n'
        'The ITL p99 threshold is not a hard deadline guarantee for every token.')
    add('## 4. Native idle evidence and observer interference')
    qualification=[]
    for file in sorted((ROOT/'raw').glob('a_*/observer_qualification.json')):
        value=load(file)
        if value['observer']!='off':qualification.append(value)
    add(table(['Collector','Matched cohorts','Qualified local tails and coverage'],[
        [kind,sum(v['observer']==kind for v in qualification),sum(v['observer']==kind and v['qualified'] for v in qualification)]
        for kind in ('host','sparse','graph','submit_events')]))
    add('A host-only Huginn control passed the observer screen. It does not measure physical GPU idle.\n'
        'No graph or complete submission-event trace passed the complete current observer contract.\n'
        'Some complete traces changed local tails despite small cohort-duration changes.\n'
        'Some event traces improved measured tails by more than 5%. They also fail the absolute equivalence screen.\n'
        'A failed screen does not isolate instrumentation as the sole causal source of variation.')
    add(table(['Minimal-host event pilot','Minimum local ratio','Maximum local ratio','Cohort duration ratio'],[
        [m,load(ROOT/f'raw/a_{m}_heavy_submit_events_minimal_pilot_n128/observer_qualification.json',{}).get('minimum_ratio'),
         load(ROOT/f'raw/a_{m}_heavy_submit_events_minimal_pilot_n128/observer_qualification.json',{}).get('maximum_ratio'),
         load(ROOT/f'raw/a_{m}_heavy_submit_events_minimal_pilot_n128/observer_qualification.json',{}).get('cohort_duration_ratio')]
        for m in ('ouro','huginn')]))
    add('The final-flush correction recovered complete buffers in some runs.\n'
        'The Ouro drained pilot returned 2/2 buffers with zero invalid records.\n'
        'The Huginn drained pilot returned 10/10 buffers but contained one invalid timestamp record.\n'
        'The latter run completed its response cohort before export failed.\n'
        'This is not a model-correctness failure. Its invalid-record cause remains ROOT_CAUSE_INCONCLUSIVE.\n'
        'The retained audit distinguishes collector plumbing, coverage, perturbation, and model correctness.')
    add('Physical kernel idle, scheduler-visible idle, and deadline-safe future execution remain separate evidence classes.\n'
        'Graph envelopes include their internal gaps. CUDA Event enclosures also include waits and host submission gaps.\n'
        'Their complements are bounds, not exact kernel-idle measurements or future execution guarantees.\n'
        'The eligible traces remain diagnostic. A valid same-run Safe Slack distribution is NOT_MEASURED.')
    add(table(['Required opportunity distribution','Ouro','Huginn'],[
        ['Count and duration p50/p90/p95/p99','NOT_MEASURED','NOT_MEASURED'],
        ['Fractions at least 1/2/4/8/16/32 ms','NOT_MEASURED','NOT_MEASURED'],
        ['Same-run validated P-backlog plus safe future budget','NOT_MEASURED','NOT_MEASURED']]))
    add('Native A/A repeats diagnose run-to-run variation. They do not relax the frozen observer gate.\n'
        'They use the earlier minimal-host pilot as their reference. They do not isolate a simultaneous instrumentation effect.\n'
        'The host-observer ablation removes live HBM queries and optional transport observations without changing model computation.\n'
        'Its raw GC and long-call records remain diagnostic. Startup GC intervals do not count as measured-cohort pauses.\n'
        'See `measured_host_observer_intervals.csv` and `state/finish_controls.json`.')
    aa=load(ROOT/'state/finish_controls.json',{'results':[]})
    add(table(['Native A/A repeat','Minimum local ratio','Maximum local ratio','Cohort duration ratio','Inside pointwise 5% band'],[
        [r['name'],r['comparison']['minimum_ratio'],r['comparison']['maximum_ratio'],
         r['comparison']['cohort_duration_ratio'],r['comparison']['qualified']]
        for r in aa['results'] if 'comparison' in r]))
    uncertainty=load(ROOT/'observer_start_intervals.json',[])
    add(table(['Model','Window','Metric','Independent starts','Mean observer/Native ratio','95% CI'],[
        [r['model'],r['window'],r['metric'],r['independent_starts'],r['mean_ratio'],r['CI95']] for r in uncertainty]))
    add('These intervals resample independent start-level ratios, not tokens or requests.\n'
        'Three starts provide limited uncertainty evidence. The intervals do not override the frozen pointwise observer gate.\n'
        'No method-improvement confidence interval exists because no adaptive method cohort ran.')
    add('## 5. Full-prompt continuation primitive costs')
    primitives=load(ROOT/'full_prompt_primitive_costs.json',[])
    add(table(['Model','Remaining loops on D','P event sum p50 ms','D event sum p50 ms','Host handoff p50 ms','Serial Service TTFT p50 ms'],[
        [r['model'],r['remaining_loops'],r['streams']['prefill']['p50_ms'],r['streams']['decode']['p50_ms'],
         r['handoff']['p50_ms'],r['service_TTFT']['p50_ms']] for r in primitives]))
    add('Each row contains eight serial requests from one independent start.\n'
        'CUDA Events measure model-stream enclosures, not exclusive kernel compute.\n'
        'D prompt enclosures exclude transfer and separate first-output delivery.\n'
        'Huginn terminal continuation includes its required prompt Coda-KV work.\n'
        'Host handoff is an observed elapsed interval, not pure network time or incremental cost over Native.\n'
        'Do not add the component percentiles into a critical-path latency.\n'
        'Cross-start confidence intervals are NOT_MEASURED for these primitives.')
    add('The original Huginn one-loop serial row has lower Service TTFT than its Native control.\n'
        'The later one-loop-quantum control has a different handoff interval and higher Service TTFT.\n'
        'Their D compute enclosures remain similar. Compute timing alone therefore does not establish the request-latency outcome.\n'
        'These independent single-start observations do not isolate a recurrent redistribution benefit.')
    add('A full-prompt loop is much larger than the earlier 16-token independent probe.\n'
        'Small-tile timing cannot establish whole-prompt admission safety.\n'
        'A loop longer than one physical idle gap also does not prove that deadline-safe borrowing is impossible.\n'
        'Host preparation can overlap device execution. A validated budget or actual insertion test must establish that opportunity.')
    add('## 6. Matched one-loop execution quantum: Token versus Loop')
    comparison=rows('equal_quantum_comparison.csv')
    add(table(['Model','Family','D recurrent rows/request','P saved enclosure p50 ms','D total p50 ms','Largest D submission ms','Payload GiB','Serial Service TTFT p50 ms'],[
        [r['model'],r['family'],r['D_recurrent_rows_per_request'],r['P_saved_enclosure_p50_ms'],
         r['D_total_enclosure_p50_ms'],r['D_largest_model_submission_enclosure_ms'],r['payload_GiB'],r['service_TTFT_p50_ms']]
        for r in comparison]))
    add('Ouro compares Loop3 against Token3072. Huginn compares Loop31 against Token1984.\n'
        'Both families use the same legal D recurrent execution quantum of one loop.\n'
        'Both passed their separate N8 state tests.\n'
        'This is a matched-quantum diagnostic, not a search for the best Token baseline.\n'
        'Larger Token suffixes and other loop quanta require separate cost and Decode-safety tests.\n'
        'Equal recurrent token rows do not imply equal attention FLOPs.\n'
        'A Token suffix attends to the existing prefix at every recurrent depth.\n'
        'The total D enclosure and the largest scheduling quantum measure different costs.\n'
        'A lower total Loop cost does not establish lower Decode interference.\n'
        'These serial records provide no concurrent-Decode or adaptive serving advantage.')
    add(table(['Model','Family','Structural D attention pairs/core layer','D Huginn Prelude tokens','D Huginn Coda-KV tokens'],[
        [r['model'],r['family'],r.get('structural_D_causal_attention_pairs_per_recurrent_layer'),
         r.get('structural_D_Huginn_Prelude_tokens'),r.get('structural_D_Huginn_Coda_KV_tokens')]
        for r in comparison]))
    add('These counts are structural accounting, not measured FLOPs or speedups.\n'
        'Huginn Loop31 also moves substantially more prompt Coda-KV work than Token1984.\n'
        'Token1984 moves suffix Prelude work that Loop31 does not move.\n'
        'Therefore, equal recurrent rows do not provide an equal-total-work causal comparison.\n'
        'The measured total-cost difference cannot establish a Loop-specific advantage by itself.')
    add('## 7. Decode service diagnostic')
    service=load(ROOT/'serial_decode_service_diagnosis.json',[])
    add(table(['Model','Serial Decode steps','Delivery ITL p50 ms','Event-sum p50 ms','Per-step residual p50 ms'],[
        [r['model'],r['steps'],r['ITL_p50_ms'],r['event_sum_p50_ms'],r['residual_p50_ms']] for r in service]))
    add('The residual is delivery ITL minus the sum of recorded operation enclosures for that token.\n'
        'It is not physical idle or an available online borrowing budget.\n'
        'The sum can include stream waits. Medians in this table are not additive.\n'
        'Huginn has more host-side recurrent submissions than Ouro in this serial path.\n'
        'This diagnostic does not isolate CPU scheduling as the cause of online tail latency.')
    add('## 8. Forced mixed-cut correctness')
    ledger=load(ROOT/'state/forced_mixed_qualification.json',{'results':[]})
    exact=[]
    for r in ledger['results']:
        cfg=load(ROOT/'configs'/(r['name']+'.json'))
        cuts=cfg.get('native_request_cuts',[])
        exact.append([r['name'],r.get('passed'),r.get('actual_offload',{}).get('D_prompt_rows'),
            sum(c<cfg['depth'] for c in cuts) if cuts else 0,
            sorted(set(cfg['depth']-c for c in cuts)) if cuts else [0]])
    add(table(['Run','Qualified','D prompt recurrent rows','Forced offloaded requests','Remaining-loop counts'],exact))
    add('The cohort uses preassigned cuts. It does not observe a Safe Slack estimator.\n'
        'The reused connector communicates those cuts during reservation.\n'
        'It does not renegotiate a cut after observing each completed P loop.\n'
        'A host graph submission boundary does not establish GPU completion at that boundary.\n'
        'N32 means simultaneous submission, not 32 GPU-resident requests.\n'
        'The state tests compare exact output tokens, valid prompt KV planes, and recorded hidden state.\n'
        'They also validate full recurrent rows, transfer fingerprints, ACK-before-retirement, and empty final native allocations.\n'
        'Static full-PD branches are forced fallbacks, not live rejection or stale-credit tests.\n'
        'Real transfer-time Decode arrivals, overruns, queue aging, and adaptive admission remain NOT_TESTED.\n'
        'No intermediate kernel preemption or D-to-P return is assumed.')
    add('## 9. Native finite-cohort serving controls')
    serving=rows('serving_runs.csv');selected=[]
    for model in ('ouro','huginn'):
        for level in ('low','medium','heavy'):
            name=f'a_{model}_{level}_off_load_control_n128' if level!='heavy' else f'a_{model}_heavy_off_s1_n128'
            r=next((v for v in serving if v['name']==name),None)
            if r:selected.append([model,level,r['nominal_QPS'],r.get('realized_QPS'),r.get('finite_completed_QPS'),
                r['TTFT_p95_ms'],r['TPOT_p95_ms'],r['ITL_p99_ms'],r['max_ITL_ms'],r['E2E_p95_ms'],r['joint_goodput']])
    add(table(['Model','Load','Offered QPS','Realized QPS','Completed QPS','TTFT p95 ms','TPOT p95 ms','ITL p99 ms','Max ITL ms','E2E p95 ms','Joint goodput req/s'],selected))
    add('These are Native controls, not Dynamic Loop results. Each complete primary cohort offers 128 requests.\n'
        'The load grid uses the first declared Phase A arrival seed, 2026110611, at every QPS.\n'
        'Minimal-host ablations and later observer seeds remain separate controls.\n'
        'The complete CSV includes p50/p95/p99 latency, token throughput, SLO counts, and threshold exceedances.\n'
        'All offered requests remain the SLO denominator.\n'
        'The heavy setting can accumulate a finite overload queue. Completed cohort throughput is not sustainable capacity.\n'
        'Primary marker-free Service TTFT and actual-first-model Queue Latency are NOT_MEASURED.\n'
        'Admission timestamps alone do not replace actual model entry.\n'
        'The separately instrumented serial Service TTFT is not a replacement for clean online Service TTFT.')
    add('## 10. Oracle status')
    oracle=load(ROOT/'opportunity_oracle_status.json',{})
    add('The cost-fit diagnostic uses measured serial costs and optimistic recorded gap envelopes.\n'
        'It removes graph compute and ignores handoff, frontier timing, and admission costs equally for both action families.\n'
        'It is a clairvoyant relaxed cost-fit bound, not a Feasible Oracle.\n'
        'No calibrated prediction of online TTFT, E2E, goodput, or Decode tail is reported.\n'
        'Validation error is NOT_MEASURED. No unvalidated oracle speedup appears as an online result.\n'
        'No new token-tiled variable-loop primitive ran because whole-prompt infeasibility was not established.\n'
        'Missing slack evidence cannot trigger a claim that full-prompt loops never fit.')
    add(f'Diagnostic action count: {len(oracle.get("actions",[]))}. Diagnostic comparison rows: {oracle.get("records","NOT_MEASURED")}.\n'
        'See `relaxed_opportunity_oracle.csv` and `offline_budget_loop_decisions.csv`.\n'
        'Their offline cost-fit selections are not accepted handoffs or actual online cut decisions.')
    add('## 11. Memory and ownership')
    memory=rows('HBM_components.csv')
    add(table(['Model','Role','Component','GiB','Percent of NVML HBM'],[
        [r['model'],r['role'],r['component'],r['GiB'],r['percent_NVML_HBM']] for r in memory]))
    add('The denominator is the measured NVML capacity: 81,559 MiB per GPU.\n'
        'Allocator peak totals include component allocations. Do not stack them with weights and KV pool bytes.\n'
        'A receiving request already reserves its D KV lease before P starts.\n'
        'Its continuation must not require a second full-request lease.\n'
        'D compute slots and KV lease capacity are different limits.\n'
        'Recorded ACK protects source lifetime. No optimization releases an in-flight source buffer early.\n'
        'No proof of a future SLO-safe workspace or KV budget follows from average free HBM.')
    add('## 12. Failures, missing experiments, and attribution')
    add('The complete configuration inventory retains every failed and unrun cell.\n'
        'Interrupted controller ledgers can retain RUNNING status. Dispatch receipts and the closure receipt establish their actual final state.\n'
        'Misrouted early collector jobs completed Native serving rather than their intended synthetic self-test.\n'
        'Only the revised `_v2` self-tests establish their stated collector plumbing results.\n'
        'An empty or partial trace never becomes an idle GPU interval.\n'
        'The superseded invalid derived gap table remains named and preserved.\n'
        'Final-export failures remain distinct from model-state errors and serving-body progress stalls.')
    add('The untested method arms are Static Token/Loop clean serving under this campaign, Dynamic Idle-gated Token, and Dynamic Idle-gated Loop.\n'
        'Also untested are an adaptive mixed-cut policy, delayed-ACK fault injection, stale-credit rejection, and N512 persistence.\n'
        'P exclusive kernel union, SM active cycles, bandwidth saturation, and an online budget validation are NOT_MEASURED.\n'
        'No current same-run P compute-saturation plus accepted adaptive D-work proof is reported.\n'
        'The campaign does not change earlier accepted operational evidence into a new same-run result.\n'
        'It also does not reinterpret earlier Dynamic zero-offload campaigns as successful adaptive work redistribution.')
    add('Observed costs establish a coarse full-prompt continuation quantum and nonzero state-transfer cost.\n'
        'They do not establish that granularity or transfer alone makes the method impossible.\n'
        'The relative contributions of CPU scheduling, recurrent structure, transfer progress, and tail jitter remain unresolved.\n'
        'The result is INCONCLUSIVE, not NO_GO_IDLE_SLACK or NO_GO_NET_BENEFIT.\n'
        'A future campaign must first qualify a low-overhead same-run observer or validate actual conservative admission directly.\n'
        'It must then demonstrate genuinely variable online accepted cuts before claiming an Adaptive Loop-Split result.')
    add('## 13. Figures, reproducibility, and resource closure')
    manifest=load(ROOT/'figure_manifest.json',{'figures':[]})
    add(table(['Figure','Evidence class'],[[f'[{v["name"]}]'+'(ouro_huginn_adaptive_idle_gated_loop_continuation/'+v['files'][0]+')',v['scope']]
        for v in manifest['figures']]))
    add('The figures label diagnostic traces, serial enclosures, optimistic cost fits, and missing online arms.\n'
        'No figure shows an invented adaptive execution timeline or measured speedup.\n'
        'The operational restoration receipt records actual GPU engines, ownership, and utilization on all eight GPUs.\n'
        'It is separate from scientific GPU evidence.')
    add('Analysis commands from the repository root:')
    add('```bash\nPY=/home/esjung/.venvs/looped-pd-abc/bin/python\n'
        '$PY -m poc_flashvep.adaptive_idle_loop_continuation.cpu_regression\n'
        '$PY -m poc_flashvep.adaptive_idle_loop_continuation.analyze\n'
        '$PY -m poc_flashvep.adaptive_idle_loop_continuation.primitive_analysis\n'
        '$PY -m poc_flashvep.adaptive_idle_loop_continuation.decode_service_diagnosis\n'
        '$PY -m poc_flashvep.adaptive_idle_loop_continuation.opportunity_oracle\n'
        '$PY -m poc_flashvep.adaptive_idle_loop_continuation.inventory\n'
        '$PY -m poc_flashvep.adaptive_idle_loop_continuation.figures\n```')
    add('Existing immutable namespaces must not start new GPU jobs after closure.\n'
        'A scientific GPU rerun requires a new authorized namespace and explicit new clock.\n'
        'The archived per-run source, configuration hash, model revision, GPU mapping, and arrival offsets define each original run.\n'
        'Lossless archive manifests retain oversized originals without deleting local files.\n'
        'The ordinary push receipt records the evidence commit and matching remote SHA. No force push is used.')
    add('## 14. Answers to the four research questions')
    add('1. A validated same-run deadline-safe recurrent execution budget is not established. Diagnostic idle bounds alone are insufficient.\n'
        '2. Different forced Loop counts are exact, but actual online Safe Slack-based variable Cut selection is not demonstrated.\n'
        '3. Adaptive TTFT, E2E, goodput, and token-throughput gains with Native-like Decode tails remain unmeasured.\n'
        '4. The serial Token/Loop comparison exposes a total-cost versus execution-quantum trade-off. No fair Dynamic Serving comparison establishes a winner.')
    FINAL.write_text('\n'.join(out))
    print(json.dumps(dict(report=str(FINAL),verdict='INCONCLUSIVE_SLACK',written_KST=datetime.now(ZoneInfo('Asia/Seoul')).isoformat())))


if __name__=='__main__':main()
