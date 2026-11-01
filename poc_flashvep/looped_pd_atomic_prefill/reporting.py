"""Compile completed evidence. Do not convert missing gates into positive claims."""
import collections
import csv
import datetime
import statistics
import subprocess
from .common import *
from .evidence import records,rows,status,compare


def read(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else default


def cell(value):
    if value is None:return 'NOT_MEASURED'
    if isinstance(value,float):return f'{value:.4f}'
    if isinstance(value,(list,dict)):value=json.dumps(value,separators=(',',':'))
    return str(value).replace('|',' / ').replace('\n',' ')


def table(headers,values):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
    lines += ['| '+' | '.join(cell(v) for v in row)+' |' for row in values]
    return '\n'.join(lines)+'\n'


def report(name,text):
    p=REPO/'poc_flashvep/reports'/name
    p.write_text(text.rstrip()+'\n')


def model_decisions(runs,classifications,blocks):
    result=[]
    for model in MODEL_PINS:
        observed=[r for r in runs if r['model']==model and r['phase']=='final_repeat' and r['eligible_clean']]
        checked=[r for r in classifications if r['model']==model]
        groups=collections.defaultdict(list)
        runmap={r['name']:r for r in runs}
        for r in checked:
            source=runmap[r['name']]
            if source['phase']=='final_repeat':
                key=(r['length'],r['output_cap'],r['chunk'],source['transport_control'],source['target_QPS'])
                groups[key].append(r)
        repeated=[]
        for key,group in groups.items():
            starts={runmap[r['name']]['independent_start'] for r in group if r['category']=='TRUE_PREFILL_HEAVY'}
            if len(starts)>=3:repeated.append(key)
        affected=[v for k,v in blocks.items() if k==model or k.startswith(model+'|')]
        blocked=bool(affected)
        if repeated:verdict='TRUE_PREFILL_HEAVY_FINITE_REPEATED'
        elif blocked and not observed:
            verdict='BLOCKED_RUNTIME' if all(v.get('classification')=='RUNTIME_FAILURE_REQUIRES_TRIAGE' for v in affected) else 'BLOCKED_CORRECTNESS'
        elif observed:verdict='NO_CONFIRMED_PREFILL_HEAVY_ON_TESTED_GRID'
        else:verdict='NOT_RUN_FINAL_CONFIRMATION'
        result.append(dict(model=model,verdict=verdict,qualified_final_runs=len(observed),
            checked_PD_runs=len(checked),observed_categories=dict(collections.Counter(r['category'] for r in checked)),
            repeated_TRUE_cells=repeated,scope='Finite measured search. No structural impossibility, infinite-horizon capacity, or Dynamic-A speedup is established.'))
    write(ROOT/'final_model_decisions.json',result)
    return result


def main():
    stage=read('state/search_stage.json',{}).get('stage','NOT_AVAILABLE')
    assert stage in ('GPU_SEARCH_COMPLETE_ANALYSIS_PENDING','FINAL_COMPLETE','STOPPED_OPERATIONAL_24H_CHECKPOINT'),'Do not compile a final report while GPU search is active'
    from .analyze import serving,capacity
    from .matrix import main as matrix
    from .classify import main as classify
    from .native_capacity_controls import analyze as native_capacity
    runs=serving();cap=capacity();actual,requested=matrix();classifications=classify();native=native_capacity()
    blocks=read('state/online_correctness_blockers.json',{});decisions=model_decisions(runs,classifications,blocks)
    finals=[r for r in runs if r['phase']=='final_repeat']
    now=datetime.datetime.now(datetime.timezone.utc);elapsed=(time.time()-START_EPOCH)/3600
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    counts=collections.Counter(r['status'] for r in actual)
    write(ROOT/'state/report_compilation.json',dict(time=time.time(),elapsed_hours=elapsed,branch=BRANCH,
        compilation_source_commit=source,search_stage=stage,configuration_status_counts=dict(counts),
        evidence_scope='Completed raw measurements and explicitly labeled structural accounting. No inferred results fill missing cells.'))
    intro=f'Compiled at {now.isoformat()}. Elapsed time: {elapsed:.3f} hours.\n\n'
    version=f'Branch: `{BRANCH}`.\n\nCompilation source commit: `{source}`.\n\nEach measured run records its own source archive, source commit, and configuration hash.\nThe final publication commit can differ from the measurement commit.\nUse `git log -1 --format=%H` on this branch to obtain the publication SHA.\n'
    paths='Evidence root: `looped_pd_atomic_prefill_qps_crossover/`.\n\n'
    limitations=('All main generation is controlled with `ignore_eos=True`.\n'
        'Internal exact token parity does not establish original-HF fidelity or downstream answer accuracy.\n'
        'No Dynamic A, model-depth reduction, approximation, quantization, or training runs in this study.\n'
        'Finite-cohort throughput is not infinite-horizon sustainable capacity.\n')

    model_table=table(['Model','Revision','Depth','Precision'],[(MODEL_PINS[m][0],MODEL_PINS[m][1],MODEL_PINS[m][2],'BF16 weights and KV') for m in MODEL_PINS])
    runtime='# Atomic Prefill PD: Final Runtime Audit\n\n'+intro+version+'\n'+model_table
    runtime+='\n## Preserved sources\n\n'
    contract=read('configs/preservation_contract.json',{})
    runtime+=f'Private runtime HEAD: `{contract.get("RLT_HEAD")}`.\n\nPrivate working-diff SHA256: `{contract.get("RLT_diff_SHA256")}`.\n\n'
    runtime+=f'The preservation check protects {contract.get("unrelated_index_entries")} unrelated index entries.\n'
    runtime+='Previous source, reports, raw evidence, and checkpoints remain unchanged.\nProcess-local wrappers add observations and supported configuration controls.\n'
    runtime+='\n## Execution boundary\n\n'
    runtime+='Ouro and Huginn use the qualified RLT/FA4 native NIXL path.\nQwen uses installed native vLLM 0.26/FA3 and NIXL.\nThe Python environment uses Python 3.14.6 and PyTorch 2.11 with CUDA 12.9.\n'
    runtime+='The numerical variants remain `down_norm` and `down_norm_adapter`.\nSelected projections retain fixed 256-row GEMM tiles.\nA logical Full-Prompt span therefore does not imply one large Down GEMM.\n'
    runtime+='The internal arm string `PD_G1` selects native transport.\nActual Looped Full-Prompt P submissions use `loop_count=4` or `32`.\nThey are not staged-G1 Prefill solely because the arm string contains G1.\n'
    runtime+='Native Mixed is an existing RLT baseline, not a new method.\nSeparate-phase and Mixed colocated controls remain distinct.\nTheir actual scheduler, wavefront flag, token budget, KV pool, and batch histogram remain recorded.\n'
    runtime+='\n## Supported generic controls\n\n'
    runtime+='The inherited harness allocated P-sized row workspaces on a Decode-only worker.\nThe D128 control uses the supported Decode token budget of 128.\nLarger legal KV pools are separate settings, not free memory.\n'
    runtime+='Wire-credit controls change `PDConfig.max_inflight_bytes`, not the NIXL protocol or 64-MiB fragment limit.\nNo transport or memory control is a Loop-specific scheduling contribution.\n'
    runtime+='\n## GPU and data provenance\n\n'
    hardware=read('state/initial_hardware.json',{})
    runtime+='```text\n'+hardware.get('GPUs','NOT_AVAILABLE').strip()+'\n```\n\n'
    runtime+='```text\n'+hardware.get('topology','NOT_AVAILABLE').strip()+'\n```\n\n'
    runtime+='Correctness can use disjoint pairs concurrently.\nClean service, calibration, and QPS measurements use one exclusive quiet lane.\nApproved burners are absent from that lane.\nUnrelated processes are not signaled.\n'
    runtime+='The common pool contains 184 public sources: 28 ShareGPT and 156 LongBench.\nThe split contains 24 calibration and 160 test sources.\nExact model-tokenizer prefixes define input lengths.\n'
    runtime+='Optional 16K uses its own smaller test pool and an eight-source calibration extension.\nCyclic offered cohorts explicitly record repeated source use.\nThese data are neither original Azure prompts nor a natural output-length distribution.\n'
    runtime+='\n'+limitations
    report('looped_pd_atomic_prefill_runtime_audit.md',runtime)

    correctness='# Atomic Prefill PD: Final Correctness Evidence\n\n'+intro+version+'\n'
    evidence=[]
    for model,length,tag in [('ouro',8192,'state/model_gates.json'),('huginn',3900,'state/model_gates.json'),('ouro',16384,'state/optional16k_gate.json')]:
        value=read(tag,{})
        if 'optional16k' not in tag:value=value.get(model,{})
        evidence.append([model,length,value.get('output_sequences'),value.get('exact_output_parity'),value.get('exact_valid_KV_hidden_state'),
            value.get('work',{}).get('expected'),value.get('work',{}).get('observed'),value.get('passed'),tag])
    correctness+=table(['Model','Input','Sequences','Exact output','Valid KV/hidden bytes','Expected rows','Observed rows','Gate','Evidence'],evidence)
    correctness+='\nThe eight-request byte probes are heavy correctness measurements.\nTheir elapsed time is not clean Serving latency.\nWork counters count active recurrent rows, not padded physical GEMM rows.\n'
    correctness+='\n## Native AR failures\n\n'
    ar_cases=[('Default PD 8000/200, different arrival rates','pilot_qwen8_l8000_o200_cfull_f0p25_v1','pilot_qwen8_l8000_o200_cfull_f0p5_v1'),
        ('Official async PD 8000/200 repeat','diagnostic_qwen8_async_pd_l8000_o200_first_v1','diagnostic_qwen8_async_pd_l8000_o200_repeat_v1')]
    ar_rows=[]
    for label,a,b in ar_cases:
        if status(a)==status(b)=='COMPLETE':
            comparison=compare(a,b);ar_rows.append([label,comparison['compared']-len(comparison['failures']),comparison['compared'],comparison['passed'],a,b])
        else:ar_rows.append([label,None,None,False,a,b])
    correctness+=table(['Setting','Exact sequences','Compared','Pass','Reference','Candidate'],ar_rows)
    correctness+='The default short 8000/16 native-PD starts matched 8/8.\nThat finite result did not predict longer-output reproducibility.\nThe default native DP2 repeat also failed at 7/8 without NIXL.\n'
    correctness+='The first divergent internal operation and logit margin remain NOT_MEASURED.\nToken differences alone do not prove BF16 rounding or KV corruption.\nThe failed AR family does not support a qualified DuetServe reproduction.\n'
    correctness+='\n## Independent final gates\n\n'
    fq=read('state/final_qualification.json',{})
    correctness+=table(['Model','Full repeat compared','Full repeat passed','Tuned parity','Colocated parity'],[
        [m,v.get('compared'),v.get('passed'),v.get('settings',{}).get('tuned',{}).get('passed'),v.get('settings',{}).get('colocated',{}).get('passed')] for m,v in fq.items()])
    correctness+='The final qualification uses 32 requests, another arrival seed, and another sampling seed.\nIt uses new Prompt rows where the source pool permits them.\nEarlier byte checks remain separate evidence.\nA missing or failed gate never becomes a performance pass.\n'
    correctness+='\n## Complete gate and failure records\n\n'
    correctness+='See `state/live_correctness_audits.jsonl`, `state/online_correctness_blockers.json`, and `configuration_matrix.csv`.\n'
    correctness+='Workspace, wire-credit, chunk, Native Mixed, and optional-context gates have separate JSON records in `state/`.\nFailed runs retain their complete logs and source archives.\n'
    correctness+='\n'+limitations
    report('looped_pd_atomic_prefill_correctness.md',correctness)

    refs=read('configs/capacity_references.json',{})
    capacity_text='# Atomic Prefill PD: Final Independent Capacity\n\n'+intro+version+'\n'
    capacity_text+='Evidence type: MEASURED_FINITE_INDEPENDENT_SERVICE.\nThese synchronized host service boundaries are not pure GPU kernel durations.\nAll-ready D cohorts are optimistic relative to staggered online readiness.\n'
    capacity_text+='\n## P service by chunk\n\n'
    capacity_text+=table(['Model','Input','C','Selected P slots','Finite requests/s'],[
        [v['model'],v['length'],int(c),r['P_slots'],r['mu_P']] for v in refs.values() for c,r in v['chunks'].items()])
    capacity_text+='Each selected P concurrency is the smallest tested batch within 95% of that C setting\'s best rate.\nThe strongest C remains a mandatory tuning control.\n'
    capacity_text+='\n## Local prepared D\n\n'
    capacity_text+=table(['Model','Input','Output','D batch','Finite requests/s','Strongest P / D'],[
        [v['model'],v['length'],int(c),r['source']['batch'],r['mu_D'],max(t['mu_P'] for t in v['chunks'].values())/r['mu_D']]
        for v in refs.values() for c,r in v['D'].items()])
    capacity_text+='RLT P timing excludes first Coda sampling.\nRLT prepared D includes first Coda sampling.\nNative AR P includes first sampling.\nIts prepared D excludes preparation and first sampling.\nThese boundaries are not additive E2E components.\n'
    capacity_text+='\n## Actual native prepared D\n\n'
    capacity_text+=table(['Model','Input/output','Batch','D budget','D KV blocks','Independent starts','Requests/s','95% CI'],[
        [v['model'],f'{v["length"]}/{v["output_cap"]}',v['batch'],v['D_token_budget'],v['D_kv_blocks'],v['independent_starts'],v['finite_requests_per_s_mean'],
            [v['bootstrap95_low'],v['bootstrap95_high']]] for v in native])
    capacity_text+='These timers start after actual native NIXL import and valid-state synchronization.\nThey include first Coda and the native coordinator.\nThree starts remain a small inferential sample.\nAll-ready native service is still not sustainable online capacity.\n'
    capacity_text+='\n## KV accounting\n\n'
    capacity_text+=table(['Model','Baseline KV bytes/GPU','Blocks','Allocated planes','Valid Attention planes'],[
        ['Ouro R4',70866960384,22528,96,96],['Huginn R32',66437775360,24576,256,132]])
    capacity_text+='Evidence type: MEASURED_ALLOCATION plus STRUCTURAL_ACCOUNTING.\nHuginn has 124 unused storage planes, or 48.4375% of the uniform allocation.\nThis is not a measured speedup from compact KV storage.\nActual native transfer uses valid state, not every reserved storage plane.\n'
    capacity_text+=table(['Model/input','Measured bytes/request','Fragment completions/request'],[
        ['Ouro/8192',6442455040,97],['Huginn/3900',10872554880,163],['Huginn/3500',9757418880,146]])
    capacity_text+='The fourth Huginn 3500/32 reservation requires 28,288 blocks.\nThe larger measured KV pool occupies 76,472,647,680 bytes.\nIts D128 memory probe peaks at 83,746,289,664 allocated bytes.\nThis tight allocation does not guarantee arbitrary Prefill-continuation workspace.\n'
    capacity_text+='\nAll batches and blocked memory cases remain in `capacity_samples.csv` and `capacity_summary.csv`.\n'
    report('looped_pd_atomic_prefill_capacity.md',capacity_text)

    status_table=table(['Configuration status','Count'],sorted(counts.items()))
    search='# Atomic Prefill PD: Final QPS Search\n\n'+intro+version+'\n'+paths
    search+='## Completion ledger\n\n'+status_table
    search+='`configuration_matrix.csv` includes every actual configuration.\n`requested_search_grid.csv` retains every requested input/output/chunk cell.\nA configuration file alone is not an executed result.\n'
    search+='\n## Frozen SLOs\n\n'
    search+=table(['Cell','TTFT limit ms','Request-average TPOT limit ms','Secondary token ITL limit ms','Calibration requests'],[
        [k,v['TTFT_ms'],v['TPOT_ms'],v['ITL_ms'],v['requests']] for k,v in read('configs/frozen_SLO.json',{}).items()])
    search+='Thresholds were frozen before the corresponding held-out search.\nOptional 16K adds a new calibrated cell without changing existing thresholds.\nThe primary SLO uses request TTFT and request-average TPOT.\nToken ITL tails remain separate.\n'
    search+='\n## Independent final runs\n\n'
    search+=table(['Run','Input/output','Arm/C','Qualified clean','Target QPS','Offered/completed','TTFT p95 ms','TPOT p95 ms','ITL p99 ms','E2E p95 ms','SLO attainment','SLO goodput req/s'],[
        [r['name'],f'{r["length"]}/{r["output_cap"]}',f'{r["arm"]}/{r["chunk"]}',r['eligible_clean'],r['target_QPS'],f'{r["offered"]}/{r["completed"]}',
            r['TTFT_ms_p95'],r['request_TPOT_ms_p95'],r['token_ITL_ms_p99'],r['E2E_ms_p95'],r.get('SLO_attainment'),r.get('SLO_goodput')] for r in finals])
    search+='All mean/p50/p95/p99 values remain in `serving_runs.csv`.\nThat file separates target, realized, and completed rates.\nIt includes all offered requests in the SLO denominator.\nTimeouts and missing requests cannot increase goodput.\n'
    search+='\n## P/D evidence\n\n'
    chosen={r['name'] for r in finals}
    search+=table(['Run','Class','P rate','P finite reference','P kernel busy','D finite reserve','D credit clear','D TPOT margin','Staggered D pass'],[
        [r['name'],r['category'],r['actual_P_completion_rate'],r['muP_finite'],r['P_profiled_kernel_busy_fraction'],
            r['D_finite_service_reserve'],r['D_credit_clear'],r['D_TPOT_margin_pass'],r['D_staggered_service_validation']]
        for r in classifications if r['name'] in chosen])
    search+='The complete per-run gate table is `bottleneck_classification.csv`.\nP saturation requires actual kernel activity, service proximity, and growing ready backlog.\nD reserve requires compute, KV credit, readiness, and frozen SLO margin.\nNVML activity alone proves neither saturation nor spare compute.\n'
    search+='\n## Profiling and transport\n\n'
    profiles=read('profile_activity.json',[])
    search+=table(['Profile','Matched clean run','Wall overhead fraction','Eligible transfer to clean','P recorded kernel fraction','D graph-active fraction'],[
        [v.get('name'),v.get('clean_reference'),v.get('profile_wall_overhead_fraction'),v.get('eligible_profile_transfer_to_clean'),
            next((d.get('kernel_busy_fraction') for d in v.get('devices',[]) if d['role']=='prefill'),None),
            next((d.get('GPU_graph_active_fraction') for d in v.get('devices',[]) if d['role']=='decode'),None)] for v in profiles])
    search+='Graph-level tracing omits D graph-internal kernels.\nD graph-active time is not exclusive kernel time or SM occupancy.\nThe parser transfers a profile only to its matched clean run within the preregistered 5% wall-overhead limit.\n'
    search+='The model-free two-process NIXL probe measures 6-GiB payload completion at about 79.993 ms with 1-GiB wire credit.\nIts effective rate is about 80.55 GB/s with exact payload checks.\nThis contiguous-buffer operator measurement is not native paged-KV E2E latency or pure DMA time.\n'
    search+='The rejected single-process payload probe remains preserved.\nIt did not reproduce native process/progress geometry and cannot estimate native Serving bandwidth.\n'
    search+='\n## Uncertainty, figures, and host shares\n\n'
    search+='`paired_start_comparisons.csv` reports paired 95% bootstrap intervals across independent starts.\nThree starts remain a small inferential sample.\nQueued requests are not independent server starts.\n'
    search+='`slo_sensitivity.csv` contains secondary threshold sensitivity only.\nThose scores do not change the frozen primary decision.\n'
    search+='`figures/figure_manifest.json` maps each plot to actual runs.\nThe timeline separates measured host envelopes from recorded device intervals.\nBlank unrecorded intervals do not become measured GPU idle.\n'
    search+='`e2e_component_shares.csv` and `e2e_component_shares_summary.csv` report exclusive host interval percentages.\nTheir components sum to 100% per request or run.\nThey are not dispatch/GEMM/DMA critical-path percentages.\nOverlapping kernels and communication cannot be summed into exclusive E2E shares.\n'
    report('looped_pd_atomic_prefill_qps_search.md',search)

    summary='# Atomic Prefill PD: Final Crossover Summary\n\n'+intro+version+'\n'
    summary+='Search terminal state: `'+stage+'`.\n\n'
    summary+=table(['Model','Verdict','Qualified final runs','Classified PD runs','Observed categories'],[
        [r['model'],r['verdict'],r['qualified_final_runs'],r['checked_PD_runs'],r['observed_categories']] for r in decisions])
    summary+='\nThe strict TRUE label requires three independently confirmed starts at the same cell and rate.\nA negative tested-grid result is not proof that recurrent models cannot become Prefill-heavy.\nA correctness blocker is not a negative bottleneck result.\n'
    summary+='\n## Answers to the research questions\n\n'
    summary+='### Q1. Qwen3-8B and the DuetServe regime\n\n'
    summary+='The requested 8000/200 native-PD family failed exact output reproducibility across arrival rates.\nThe official asynchronous diagnostic also failed.\nTherefore this study does not establish a qualified AR reproduction or disprove the paper\'s regime.\n'
    summary+='The two observed default target rates are about 1.1433 and 2.2866 requests/s.\nThey do not test the paper\'s high-load region above approximately 4 requests/s.\n'
    summary+='[DuetServe Figure 2](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf) uses Qwen3-8B, two H100 GPUs, and 8000/200 tokens.\nIts aggregated budget is 8192 tokens, and its PD backend is Dynamo.\nOur native vLLM/NIXL backend and latency-SLO protocol differ.\nNo engine difference is attributed solely to recurrence.\n'
    summary+='\n### Q2–Q3. Ouro and Huginn\n\n'
    summary+='Use the model verdict table and per-run gate table above.\nOuro supports qualified internal 8K and 16K Prompt execution.\nHuginn stays within 4096 total tokens and tests outputs as short as 32.\n'
    summary+='Shorter outputs improve the independent P/D rate ratio.\nThat ratio alone does not prove usable online D reserve.\nKV reservation and native worker progress can limit admission before pure P compute saturation.\n'
    summary+='\n### Q4. Atomic versus tuned chunked PD\n\n'
    summary+='Atomic means one actual `[0,L)` Prompt span with full recurrent depth.\nIt does not mean one kernel or one transfer fragment.\nThe strongest finite C and the best measured Serving C are separate selections.\n'
    summary+='Wire credit, D workspace, KV pool, and Native Mixed geometry remain explicit controls.\nTheir effects cannot be attributed to the token-chunk label.\nSee the paired final table and `paired_start_comparisons.csv` for actual same-arrival comparisons.\n'
    summary+='\n### Q5–Q6. First saturation and usable D margin\n\n'
    summary+='`bottleneck_classification.csv` preserves every failed and passed prerequisite.\nA high TTFT or a busy P GPU is not sufficient.\nThe decisive evidence includes ready backlog, actual P service, P kernels, D credit, and Decode SLOs.\n'
    summary+='The finite prepared-D stress tests appear in `D_margin_summary.json` when eligible.\nA missing test is NOT_MEASURED, not a reserve pass.\nEven a passing resident-KV probe does not certify arbitrary Prefill-continuation admission.\n'
    summary+='\n### Q7. Architectural versus runtime causes\n\n'
    summary+='Causal attribution remains INCONCLUSIVE.\nThe engines, kernels, numerical geometry, memory layouts, and admission rules differ.\n'
    summary+='Huginn\'s unused uniform KV planes are a measured allocation issue.\nP-sized D workspaces and wire-credit limits are generic runtime controls.\nNone of these facts proves that recurrence itself prevents Prefill-heavy Serving.\n'
    summary+='\n### Q8. Is Dynamic A justified?\n\n'
    if any(r['repeated_TRUE_cells'] for r in decisions):
        summary+='A repeated finite Prefill-heavy cell supports a later bounded static offload experiment.\nIt does not yet prove that Loop Split beats Token Split, DP2, or an optimized static baseline.\n'
    else:
        summary+='No repeated strict TRUE cell qualifies a new Dynamic-A performance claim in this completed evidence.\nDo not start a large Dynamic-A campaign from independent capacity ratios alone.\nPrioritize the documented correctness, admission, or runtime limitations before another policy study.\n'
    summary+='No Dynamic A or new offload mechanism is implemented here.\n'
    summary+='\n## Completion and limitations\n\n'+status_table+'\n'+limitations
    summary+='All unsuccessful and unrun settings remain in `configuration_matrix.csv` and `requested_search_grid.csv`.\nRaw request/token evidence, source archives, and full failure logs remain preserved.\n'
    summary+='SM occupancy, full D graph-internal kernel breakdown, original-HF fidelity, and downstream accuracy remain NOT_MEASURED.\n'
    summary+='\n## Files and reproduction\n\n'
    summary+='See `looped_pd_atomic_prefill_reproduce.md` for commands and publication checks.\nThe evidence root contains all raw CSV/JSONL, source archives, configuration hashes, figures, and failure logs.\n'
    report('looped_pd_atomic_prefill_qps_crossover_final_summary.md',summary)

    reproduce='# Atomic Prefill PD: Final Reproduction Commands\n\n'+intro+version+'\n'
    reproduce+='```bash\ncd /home/esjung/MLLM-EP-github\nPYTHON=/home/esjung/.venvs/looped-pd-abc/bin/python\n'
    reproduce+='$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.bootstrap\n$PYTHON -m unittest poc_flashvep.looped_pd_atomic_prefill.test_contracts\n```\n\n'
    reproduce+='Executed commands, worker PIDs, source commits, archive hashes, and GPU IDs are in `commands.jsonl`.\nEach raw namespace includes its exact immutable configuration and source ZIP.\n'
    reproduce+='\n## Resume without duplicate execution\n\n'
    reproduce+='Inspect `budget.json`, `state/controller.json`, and actual process command lines first.\nNever launch another controller while an existing controller owns the queue.\nThe dispatcher skips completed namespaces and refuses to overwrite incomplete or failed namespaces.\n'
    reproduce+='```bash\n$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.campaign\n```\n\n'
    reproduce+='A justified retry needs a new namespace.\nDo not replace Prompt IDs, calibration thresholds, or arrival offsets in an old configuration.\n'
    reproduce+='\n## CPU-only analysis\n\n```bash\n'
    for module in ('analyze','matrix','profiles','classify','uncertainty','sensitivity','figures','reporting'):
        reproduce+='$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.'+module+'\n'
    reproduce+='```\n\n'
    reproduce+='CPU analysis cannot turn a configuration into an executed GPU result.\nProfile overhead must pass its matched control before activity transfers to clean timing.\n'
    reproduce+='\n## Large evidence and publication\n\n'
    reproduce+='Original large files remain local.\nLossless gzip parts and `archives/archive_manifest.json` preserve them for GitHub.\nVerify the manifest SHA256 after restoration.\n'
    reproduce+='```bash\n$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.archive\n'
    reproduce+='$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.publish --final --push\n'
    reproduce+='git rev-parse HEAD\ngit ls-remote origin refs/heads/'+BRANCH+'\n```\n\n'
    reproduce+='Publication uses a temporary index and owned paths only.\nIt checks the unrelated staged-entry contract before and after the commit.\nIt never force-pushes.\n'
    reproduce+='\n## Resource restoration\n\n'
    reproduce+='The approved helper explicitly selects physical GPUs 0–7.\nIt starts burners only on unused devices after all GPU measurements end.\nIt does not signal an unrelated process.\n'
    reproduce+='```bash\n$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.resources status\n```\n'
    report('looped_pd_atomic_prefill_reproduce.md',reproduce)
    print('REPORTS_COMPILED',len(REPORTS),stage)


if __name__=='__main__':main()
