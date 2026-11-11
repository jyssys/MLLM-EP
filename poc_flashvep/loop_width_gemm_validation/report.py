"""Write the current campaign report from retained analysis files."""
import statistics
from .common import *


def fmt(value,digits=3):
    if value is None:return 'NOT_MEASURED'
    if isinstance(value,bool):return 'PASS' if value else 'FAIL'
    if isinstance(value,(int,float)):return f'{value:.{digits}f}'
    return str(value)


def ratio(value):
    if 'ratio' not in value:return value.get('status','NOT_MEASURED')
    return f'{value["ratio"]:.3f} [{value["CI95"][0]:.3f}, {value["CI95"][1]:.3f}]'


def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    initialize();a=ROOT/'analysis';serving=json.loads((a/'serving_results.json').read_text())
    pairs=json.loads((a/'paired_start_confirmation.json').read_text())
    direct=json.loads((a/'paired_method_comparisons.json').read_text())
    inventory=json.loads((a/'campaign_inventory.json').read_text())
    serial=json.loads((a/'serial_path_costs.json').read_text())
    micro=json.loads((a/'fixed_policy_confirmation.json').read_text())
    prefix=ROOT.name
    def link(name,path):return f'[{name}]({prefix}/{path})'
    findings=[]
    for r in pairs:
        if r['model']=='huginn' and r['cap']==4 and r['arm']=='loop28':
            success=r['all_exact'] and r['absolute_Decode_SLO'] and r['relative_tail_CI_gate'] and r['TTFT_improvement_CI_gate'] and r['E2E_p95_ms'].get('CI95',[0,2])[1]<=1
            findings.append('Huginn 2048/4: '+('`SERVING_LOOP_BENEFIT`, conditional on this tested workload.' if success else '`INCONCLUSIVE_OR_FAILED_RELATIVE_GATE`. The individual ratios below determine the limitation.'))
            findings += ['Loop28 passes the tested p99/TPOT gates. Its maximum ITL still increases.',
                'Loop31 provides the smaller TTFT gain with a more stable observed maximum ITL.',
                'The seed tables retain both effects. A p99 pass is not a guarantee for every Decode interval.']
    findings += ['Huginn 2048/16: the tested Loop28 reduces TTFT but fails Native-relative Decode protection.',
        'Ouro 4096/32: both tested split arms failed the N64 Decode screen. Unsafe split arms did not receive N128 repetition.']
    lines=['# Loop width versus Token chunk after GEMM tuning — final summary','',
        '## Outcome','',*findings,'',
        'A practical width advantage survives the tested GEMM search in Huginn.',
        'This is not universal Loop dominance or a globally optimal-kernel proof.',
        'It is also not proof of exclusive recurrent-model novelty.',
        'An equivalent logical-layer frontier remains an equivalent mechanism.','',
        '## Contract and experiment scope','',
        'The campaign preserves the original BF16 checkpoints and fixed R32/R4.',
        'It preserves Attention, full recurrent work, numerical normalization, Native handoff, ACK, and source lifetime.',
        'It does not use quantization, early exit, approximate Attention, or changed SLOs.',
        'All compared serving arms use the same frozen prompt-projection policy.',
        'Small Decode projections keep the qualified original implementation.',
        'Huginn uses GPUs 4/5. Ouro uses GPUs 0/1. Timing uses one isolated pair.',
        'The serving methods are static splits under online Poisson arrivals.',
        'This campaign does not implement a new adaptive online scheduler.','',
        'The preserved pool contains 128 distinct controlled ShareGPT/LongBench prompt IDs.',
        'N128 uses those 128 IDs. Arrival seeds and process starts are independent.',
        'The prompt texts are not a new dataset relative to earlier campaigns.',
        'Prefix caching remains disabled. Output uses a fixed cap and ignores EOS.',
        'Huginn uses QPS 1.971. Ouro uses QPS 6.356.',
        'Output4 is a separate discovery workload. It does not replace the Output16 anchor.',
        'The frozen TTFT/TPOT/ITL thresholds remain unchanged.','',
        'ITL uses the frozen output-delivery contract.',
        'Coalesced delivery intervals do not reveal individual GPU token-generation times.',
        'The seed-level evidence records coalesced events and threshold exceedances.','']
    closure=ROOT/'state/CAMPAIGN_CLOSED.json'
    if closure.exists():
        v=json.loads(closure.read_text())
        lines += [f'Campaign start: {datetime.fromtimestamp(START,timezone.utc).isoformat()}.',
            f'GPU work closed: {datetime.fromtimestamp(v["time"],timezone.utc).isoformat()}.',
            f'Elapsed time to GPU closure: {(v["time"]-START)/3600:.3f} hours.','']
    rows=[]
    for model,cap in (('huginn',4),('ouro',32)):
        c=json.loads((ROOT/'configs'/f'serve_v2_{model}_zero_O{cap}_N64_a2026111110.json').read_text())
        rows.append([model,c['model_revision'],c['P_slots'],c['D_slots'],c['kv_blocks'],
            c['D_kv_blocks'],c['deep_graph_budget'],f'{c["warmup_count"]} requests, {c["warmup_cap"]} outputs'])
    lines += [table(['Model','Pinned revision','P slots','D slots','P KV blocks','D KV blocks','Graph-shape budget','Initial warm-up'],rows),'',
        'The common policy reserves the primary graph shape before measurement.',
        'Warm-up requests do not enter serving statistics.',
        'The same model-specific KV pool and graph budget apply to every compared serving arm.',
        'The runtime also primes configured recurrent graph geometries. Raw configurations retain those details.','',
        table(['Model','Arm label','P frontier','D prompt work'],[
            ['Huginn R32, input 2048','zero','All 32 Loops','No prompt continuation'],
            ['Huginn R32, input 2048','loop','Loop31','Full prompt × 1 Loop'],
            ['Huginn R32, input 2048','loop28','Loop28','Full prompt × 4 Loops'],
            ['Huginn R32, input 2048','token / token_group32','Prefix 1984, all depths','Suffix 64 × 32 Loops'],
            ['Huginn R32, input 2048','token1792','Prefix 1792, all depths','Suffix 256 × 32 Loops'],
            ['Ouro R4, input 4096','loop','Loop3','Full prompt × 1 Loop'],
            ['Ouro R4, input 4096','token','Prefix 3072, all depths','Suffix 1024 × 4 Loops']]),'',
        'Token suffixes of 64 and 256 are the measured Huginn serving candidates.',
        'This finite serving search does not prove the globally best Token cut.',
        'The projection experiment covers more row geometries than the serving experiment.','',
        '## Kernel search and state qualification','',
        'The search contains 441 projection cells across 12 independent GPU processes.',
        'Each geometry has a 64 MiB workspace cap and up to 32 algorithm configurations.',
        'The candidates include library heuristics, valid no-split-K variants, and canonical-shape algorithm reuse.',
        'Canonical reuse executes the real unpadded row count.',
        'Each map selects kernels from start 1 only. Starts 2 and 3 do not reselect kernels.',
        'Additional search covers actual batched P shapes, including Huginn M3968 and Ouro M6144/M8192.',
        'Existing geometry choices remain unchanged in the v2 map.',
        'The loaded library is cuBLASLt 120901. Torch is 2.11.0+cu129.',
        'The bridge uses CUDA 12.8 headers. The evidence retains loaded libraries and algorithm attributes.',
        'cuBLASLt supports reusable geometry-specific choices. See the [NVIDIA API documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgogetheuristic).','',
        f'The recorded state qualification contains {inventory["qualified_test_requests"]} test requests.',
        f'All recorded N8/N32 qualification gates: {fmt(inventory["all_recorded_qualifications_passed"])}.',
        'Those counts include repeated prompts across methods and kernel policies.',
        'The checks compare exact output IDs, every recorded valid KV-plane hash, and prompt Hidden State.',
        'They also check full recurrent work, Native wire identity, ACK order, credit retirement, and normal worker exits.',
        'This is tested internal Native parity. It does not prove equality for every possible input.',
        link('Qualification and source inventory','analysis/campaign_inventory.json'),'',
        '## Width after kernel tuning','',
        'The table uses the fixed start-1 policy across three independent starts.',
        'It sums isolated projection timings with structural layer multiplicities.',
        'It is not an executed recurrent core or serving speedup.','']
    rows=[]
    for model in ('huginn','ouro'):
        for b in (1,2,4,8):
            values=[r['token_over_loop'] for r in micro['width'] if r['model']==model and r['token_batch']==b]
            if values:rows.append([model,b,fmt(statistics.mean(values)),f'{min(values):.3f}–{max(values):.3f}'])
    lines += [table(['Model','Token batch geometry','Token / Loop dense-time sum','Range across starts'],rows),'',
        'Compatible Token batch shapes nearly remove the isolated dense gap.',
        'That experiment does not provide ready requests, Attention state, or free KV credit.',
        'A single full-prompt Loop obtains its width from one request.',
        'The 64-token suffix requires 32 dependent passes in Huginn.',
        'A different GEMM cannot make later nonlinear recurrent inputs ready in advance.',
        'Batching or a different fused execution design can change the frontier.',
        link('Fixed-policy projection confirmation','analysis/fixed_policy_confirmation.json'),'',
        '## Actual legal recurrent core','',
        'These controls measure real Native-path continuation with valid Attention and KV state.',
        'The core intervals exclude separate Prelude and prompt Coda calls.',
        'They include graph output clones and possible host submission gaps.',
        'They are CUDA Event enclosures, not exclusive kernel unions.','']
    rows=[]
    for r in serial:
        if not r['name'].startswith('core_cost_') or '_zero_' in r['name']:continue
        c=r['core_component_enclosures'].get('decode',{}).get('packed_recurrent_graph',{})
        rows.append([r['name'].replace('core_cost_huginn_','').replace('_serial8_s1',''),fmt(c.get('p50')),fmt(r['streams']['decode']['p50']),fmt(r['passed'])])
    lines += [table(['Control','Core sum p50 ms','Full D prompt sum p50 ms','Output/work/lifetime'],rows),'',
        'The primary geometry is 2048 tokens × one Loop versus 64 tokens × 32 Loops.',
        'Both contain 2048 recurrent token rows.',
        'Their recurrent dense FLOPs match at 6,706,062,950,400 operations.',
        'Their Attention and boundary work do not match.',
        'Per core layer, causal-pair counts are 2,098,176 for Loop and 4,129,792 for Token.',
        'D performs 2047 prompt Coda tokens for Loop and 63 for Token.',
        'Token also performs 64 Prelude tokens on D.',
        'Do not infer equal total FLOPs or equal P saving from equal recurrent rows.',
        link('Structural accounting','analysis/structural_accounting.json'),'',
        'The separate cold-matrix roofline is a modeled explanation, not measured HBM traffic.',
        'It assumes one HBM read per GEMM matrix and one output write.',
        'It uses theoretical H100 SXM rates, not measured sustained rates.',
        'Cache reuse, fusion, Attention, and submission costs remain outside that model.',
        'It does not prove an optimal-kernel lower bound.',
        'See [NVIDIA GEMM fundamentals](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) and [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).',
        link('Conditional roofline','analysis/conditional_cold_matrix_roofline.json'),'',
        '## Realized v2 quantum and serial redistribution','',
        'P saving is a paired difference against the same-policy serial Native control.',
        'Do not add percentile components to reconstruct E2E latency.',
        'A model-submission enclosure is not one non-preemptive GPU kernel.','']
    rows=[]
    for r in serial:
        if not r['name'].startswith('cost_v2_'):continue
        p=r['streams']['prefill']['per_request'];z=next(v for v in serial if v['name']=='cost_v2_huginn_zero_tuned_serial8_s1')['streams']['prefill']['per_request']
        from .cost_analysis import quantile
        rows.append([r['name'].replace('cost_v2_huginn_','').replace('_tuned_serial8_s1',''),
            fmt(quantile([z[k]-v for k,v in p.items()],50)),fmt(r['streams']['decode']['p50']),
            fmt(r['largest_D_prompt_submission_ms']),fmt(r['D_prompt_submissions_per_request']),
            fmt(r['payload_bytes_per_request']/1e9)])
    lines += [table(['Arm','P saved enclosure p50 ms','D prompt sum p50 ms','Largest D submission ms','D submissions/request','Wire GB/request'],rows),'',
        'Native also transfers KV. The complete split payload is not all incremental transfer cost.',
        'Serial handoff fields retain observed host boundaries. They are not isolated DMA durations.',
        link('All serial costs and per-request values','analysis/serial_path_costs.json'),'',
        '## N64 discovery: retain every arm','',
        'All fifteen N64 cells passed exact output, full work, and lifetime checks.',
        'Every measured packed-prefill graph selection hit an existing graph.',
        'Measured graph captures and eager fallbacks were zero in those cells.',
        'The table shows marker-free coordinator delivery latency.',
        'Clean online Service TTFT is NOT_MEASURED. Host receipts do not replace it.','']
    rows=[]
    for r in serving:
        if not r.get('complete') or r['count']!=64:continue
        rows.append([r['model'],r['cap'],r['width_arm'],fmt(r['TTFT_p95_ms']/1000),fmt(r['E2E_p95_ms']/1000),fmt(r['ITL_p99_ms']),fmt(r['TPOT_p95_ms']),r['ITL_limit_misses'],fmt(r['Decode_SLO_screen'])])
    lines += [table(['Model','Output','Arm','TTFT p95 s','E2E p95 s','ITL p99 ms','TPOT p95 ms','ITL misses','Decode screen'],rows),'',
        '## Held-out N128 confirmation','',
        'The campaign defined the selection rule before N64 results.',
        'The campaign added two Output4 challengers before any N128 result.',
        'They test equal recurrent rows and the strongest absolute-SLO Token throughput candidate.',
        'All added challengers remain in the report, regardless of outcome.',
        'Arrival seeds are 2026111111, 2026111112, and 2026111113.',
        'Run order rotates between starts. The independent start is the bootstrap resampling unit.',
        'The small number of starts limits population-level uncertainty claims.',
        'These intervals describe the observed trace/start pairs. They do not establish coverage for untested workloads.',
        'The following ratios use Candidate / matched tuned Native-path Zero.',
        'Intervals are paired percentile-bootstrap 95% intervals. They are not simultaneous multi-arm confidence intervals.','']
    rows=[]
    for r in pairs:
        rows.append([r['cap'],r['arm'],r['TTFT_p95_ms'].get('starts',0),ratio(r['TTFT_p95_ms']),ratio(r['E2E_p95_ms']),ratio(r['ITL_p99_ms']),ratio(r['TPOT_p95_ms']),fmt(r['relative_tail_CI_gate'])])
    lines += [table(['Output','Arm','Starts','TTFT ratio [CI]','E2E ratio [CI]','ITL ratio [CI]','TPOT ratio [CI]','Relative tail gate'],rows),'',
        'The next table retains finite-cohort rates and maximum delivery intervals.',
        'A ratio above one means higher goodput or completion rate. It means worse maximum ITL.','']
    rows=[[r['cap'],r['arm'],ratio(r['SLO_goodput']),ratio(r['completed_QPS']),ratio(r['maximum_ITL_ms']),r['ITL_threshold_misses'],r['baseline_ITL_threshold_misses']] for r in pairs]
    lines += [table(['Output','Arm','Goodput ratio [CI]','Completed QPS ratio [CI]','Maximum ITL ratio [CI]','Candidate ITL misses','Native ITL misses'],rows),'',
        'Seed-level confirmation follows. Each row is one independent server start.','']
    rows=[]
    for r in serving:
        if not r.get('complete') or r['count']!=128:continue
        rows.append([r['model'],r['cap'],r['arrival_seed'],r['width_arm'],
            fmt(r['TTFT_p95_ms']/1000),fmt(r['E2E_p95_ms']/1000),fmt(r['ITL_p99_ms']),
            fmt(r['TPOT_p95_ms']),fmt(r['maximum_ITL_ms']),fmt(r['SLO_goodput']),
            fmt(r['completed_QPS']),r['offloaded_requests'],r['actual_D_prompt_recurrent_rows'],
            fmt(r['exact_output_work_lifetime_passed'])])
    lines += [table(['Model','Output','Seed','Arm','TTFT p95 s','E2E p95 s','ITL p99 ms','TPOT p95 ms','Maximum ITL ms','Goodput req/s','Completed req/s','Offloaded requests','D recurrent rows','Exactness'],rows),'',
        'These heavy finite cohorts do not represent an all-request SLO-compliant steady state.',
        'The following table retains arrival rates, token rates, and attainment denominators.','']
    rows=[]
    for r in serving:
        if not r.get('complete') or r['count']!=128:continue
        rows.append([r['model'],r['cap'],r['arrival_seed'],r['width_arm'],fmt(r['nominal_QPS']),
            fmt(r['realized_scheduled_QPS']),fmt(r['actual_submit_QPS']),fmt(r['completed_QPS']),
            fmt(r['input_tokens_s']),fmt(r['output_tokens_s']),fmt(100*r['TTFT_attainment']),
            fmt(100*r['TPOT_attainment'])])
    lines += [table(['Model','Output','Seed','Arm','Offered QPS','Scheduled QPS','Submitted QPS','Completed QPS','Input token/s','Output token/s','TTFT attainment % of 128','TPOT attainment % of 128'],rows),'',
        'Direct Loop / Token comparisons follow. Recurrent-row matching is not full-FLOP matching.','']
    rows=[[r['numerator']+' / '+r['denominator'],ratio(r['TTFT_p95_ms']),ratio(r['E2E_p95_ms']),ratio(r['ITL_p99_ms']),ratio(r['TPOT_p95_ms'])] for r in direct]
    lines += [table(['Output4 comparison','TTFT ratio [CI]','E2E ratio [CI]','ITL ratio [CI]','TPOT ratio [CI]'],rows),'',
        link('Seed-level serving metrics','analysis/serving_results.json'),
        link('Native-relative paired statistics','analysis/paired_start_confirmation.json'),
        link('Direct method statistics','analysis/paired_method_comparisons.json'),'',
        'All rates describe finite-cohort completion and drain.',
        'They do not establish sustainable capacity.',
        'Queueing is included in Request TTFT.',
        'No component-percentile sum substitutes for a request-level percentile.','',
        '## Completion of the frozen confirmation plan','']
    progress=json.loads((ROOT/'state/confirmation_progress.json').read_text())
    plan=json.loads((ROOT/'state/confirmation_selection.json').read_text())
    completed={r['name'] for r in progress['results'] if r.get('status')=='COMPLETE'}
    missing=[name for name in plan['jobs'] if name not in completed]
    lines += [f'The frozen plan contains {len(plan["jobs"])} cells. Completed cells: {len(completed)}.',
        'A missing cell is NOT_RUN or FAILED. It is not a zero-valued performance result.']
    for name in missing:
        record=next((r for r in progress['results'] if r['name']==name),{})
        receipt=ROOT/'raw'/name/'dispatcher_result.json'
        actual=json.loads(receipt.read_text()) if receipt.exists() else {}
        lines.append(f'- `{name}`: {record.get("status",actual.get("status","NOT_RUN_AFTER_BUDGET_CUTOFF"))}.')
    lines += ['',
        'All 24 Huginn confirmation cells completed. One of three secondary Ouro Native controls completed.',
        'The new-job cutoff prevented the remaining two controls. This is not a measured model failure.',
        'The inherited budget field says 11 hours. Its configured timestamps enforce the six-hour operational bound.',
        link('Budget interpretation','state/budget_interpretation.json'),'',
        '## Retained failures and measurement limits','',
        'The first core observer missed the BulkWrittenGraphs override.',
        'Its absent recurrent records are NOT_MEASURED, not zero time.',
        'Fresh controls observe the actual graph instance.',
        'One Ouro observer startup assumed prepared Prelude/Coda methods that Ouro does not expose.',
        'That failed before measurement. Its log remains intact. Fresh s2 controls passed.',
        'Initial nominal Token cap-1 controls did not cap Prelude.',
        'They actually grouped all 32 Loops. Their total-throughput costs remain valid.',
        'They do not provide small-quantum Decode-safety evidence.',
        'V2 caps Prelude and recurrent phases. Actual submission audits are mandatory.',
        'The report does not hide those configuration aliases.',
        'No intrusive CUPTI or Nsight collector supplies primary serving latency in this campaign.',
        'Physical kernel idle, HBM traffic, Tensor Core occupancy, and exclusive kernel time are NOT_MEASURED.',
        'Exact percentages of full-core cost caused by memory traffic versus submission overhead are NOT_IDENTIFIED.',
        'The best tested GEMM is not the globally optimal GEMM.',
        'The single Ouro N128 Native control records six ITL threshold exceedances.',
        'Its maximum ITL is 219.213 ms, although its ITL p99 passes.',
        'Two planned Native repetitions did not run. This control cannot establish Native maximum-tail stability.',
        link('Failure, coverage, and alias inventory','analysis/campaign_inventory.json'),'',
        '## Research decision and follow-up boundary','',
        'The wide-token hypothesis has practical evidence after finite kernel tuning.',
        'Small Token suffixes preserve all recurrent dependencies and matrix invocations.',
        'Larger Token suffixes and compatible batches can reduce their efficiency disadvantage.',
        'Shorter Token quanta can still protect Decode better in a constrained execution budget.',
        'Actual serving must determine the useful region. Dense throughput alone cannot decide it.',
        'No result establishes universal Loop superiority or exclusive Loop-specific novelty.',
        'The follow-up requires minimum Output16 and qualified physical-idle measurements.',
        'Current Output4 findings cannot satisfy that follow-up gate.',
        'Its ten-hour budget starts only after this campaign is reviewed, published, and resource-verified.','',
        '## Evidence and preservation','',
        f'Publication parent: `{BASE}`.',f'Publication branch: `{BRANCH}`.',
        'Every run retains configuration hashes, source snapshots, model revisions, GPU mapping, commands, and seeds.',
        'The lossless evidence bundle contains raw logs and repeated source snapshots.',
        'The publication uses a private Git index. It does not replace the active checkout.',
        'The final verification records the preserved staged index and external-runtime diff.',
        link('Figures and evidence classification','figures/figure_manifest.json'),
        link('Final source and preservation checks','state/final_source_verification.json'),
        link('Burner process and generation verification','state/final_burner_verification.json'),
        link('Evidence bundle manifest','evidence_bundle/manifest.json'),'']
    target=REPO/'poc_flashvep/reports/ouro_huginn_loop_width_vs_token_chunk_gemm_validation_final_summary.md'
    formatted=[];prose_lines=0
    for line in lines:
        structured=not line or line.startswith(('#','|','- '))
        if structured:
            if line.startswith('- ') and formatted and formatted[-1] and not formatted[-1].startswith('- '):formatted.append('')
            prose_lines=0
        else:
            if prose_lines>=2:formatted.append('');prose_lines=0
            prose_lines+=1
        formatted.append(line)
    target.write_text('\n'.join(formatted))
    print(target)


if __name__=='__main__':main()
