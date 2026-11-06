"""Write the final account from completed evidence, without synthetic results."""
import csv
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from .common import *

FINAL=REPO/'poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_final_summary.md'
PREFIX=ROOT.name+'/'


def rows(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []


def n(value,digits=3):
    if value in (None,''):return 'NOT_MEASURED'
    if isinstance(value,bool):return 'PASS' if value else 'FAIL'
    try:return f'{float(value):.{digits}f}'
    except (TypeError,ValueError):return str(value)


def markdown(headers,values):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in values])+'\n'


def main():
    finished=ROOT/'state/experiments_finished.json'
    assert finished.exists(),'Do not publish a final report while GPU work is pending'
    end=json.loads(finished.read_text());decisions=json.loads((ROOT/'final_model_decisions.json').read_text())
    counts=json.loads((ROOT/'closeout_counts.json').read_text());preserve=json.loads((ROOT/'state/preservation.json').read_text())
    serving=rows('serving_runs.csv');opportunities=rows('readiness_opportunity.csv')
    text=['# D-local prefill and cross-depth batching: final PoC summary\n']
    text += ['## 1. Decision\n',markdown(['Model','Incremental batching verdict','Generic D-local N32 state gates'],
        [[model,n(v['verdict']),', '.join(k+' '+n(value) for k,value in v['N32_generic_state_qualification'].items())] for model,v in decisions.items()])]
    text += ['The campaign did not establish a new loop-specific online serving advantage.\n',
        'Native shared-core batching can reduce separate-execution cost. Same-depth batching also captures that benefit.\n',
        'A stronger same-depth control can advance one legal prefill step before joining decode.\n',
        'This control separates arithmetic batching gains from an avoidable depth-alignment wait.\n',
        'Do not develop a new dynamic scheduler from these results alone.\n',
        'The verdict applies to this bounded workload and runtime. It does not prove that every deflection policy lacks headroom.\n']
    text += ['The batching verdict concerns the incremental microbenchmark gate. Qualified online decode-safe kernel gaps remain unmeasured.\n',
        'Thus, the campaign does not establish a structural impossibility result for D-local prefill.\n']
    start=datetime.fromtimestamp(START,ZoneInfo('Asia/Seoul')).isoformat()
    stop=datetime.fromtimestamp(end['time'],ZoneInfo('Asia/Seoul')).isoformat()
    text += ['## 2. Environment and budget\n',
        f'Start: `{start}`. GPU campaign stop: `{stop}`.\n',
        f'Elapsed through GPU closeout: {(end["time"]-START)/3600:.3f} hours. The final wall-clock limit stays 14:21:36 KST.\n',
        f'Guarded worker jobs started: {counts["executed_jobs"]}. Complete job receipts: {counts["complete_jobs"]}.\n',
        f'Failure or correctness-block records: {counts["failed_or_blocked_records"]}. Configured jobs without a worker start: {counts["configured_not_run"]}.\n',
        'A primitive grid is one worker job with multiple legal-state cells. It is not a matrix of online serving runs.\n',
        'Clean serving uses isolated H100 pairs. Ouro uses physical GPUs 0/1. Huginn uses 4/5.\n',
        'Other GPUs stay quiet during timing. All eight approved burners must return after closeout.\n',
        'The serving harness uses the native engine API and real GPU processes. It is not an HTTP endpoint benchmark.\n',
        'Ouro retains BF16 weights and R4. Huginn retains BF16 weights and R32.\n',
        'The Huginn prelude, recurrent injection, coda, and per-depth KV semantics remain unchanged.\n',
        'The campaign does not use early exit, quantization, approximation, or model training.\n',
        f'Publication base: `{BASE}`. Publication branch: `{BRANCH}`.\n',
        f'Executed checkout HEAD: `{preserve["worktree_HEAD"]}`. Private RLT HEAD: `{preserve["private_HEAD"]}`.\n',
        f'Preserved private diff SHA256: `{preserve["private_diff_SHA256"]}`.\n',
        'Ouro checkpoint revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.\n',
        'Huginn checkpoint revision: `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.\n',
        f'Execution preservation metadata: [{PREFIX}state/preservation.json]({PREFIX}state/preservation.json).\n',
        f'Software versions: [software_inventory.json]({PREFIX}state/software_inventory.json).\n',
        'The checkout stayed on its existing branch. Process-local hooks and an alternate Git index protect unrelated work.\n']
    text += ['## 3. Workload and SLO contract\n',
        markdown(['Model','Input/output tokens','Low/medium/heavy nominal QPS','P/D slots'],[
            ['Ouro','4096/32','1.23460875 / 3.4569045 / 6.356','2/8'],
            ['Huginn','2048/16','0.37903355455 / 1.06129395274 / 1.971','2/8']]),
        'The input pool is the preserved ShareGPT/LongBench controlled-prefix pool. Each N128 cohort uses 128 distinct prompt IDs.\n',
        'The campaign does not add an Azure trace or claim natural response lengths. Fixed output caps use `ignore_eos=True`.\n',
        'Prefix caching stays disabled. Multiple starts reuse the pool and do not establish new-text generalization.\n',
        'Arrivals are open-loop Poisson. The primary arrival seed is 2026110701.\n',
        'Raw records separate nominal QPS, actual send QPS, completed QPS, and send delay.\n',
        markdown(['Model','Frozen TTFT ms','Frozen request TPOT ms','Additional-work TPOT p95 screen ms','ITL p99 screen ms'],[
            ['Ouro','1112.9925646819174','53.70702898374411','48.3363260853697','65.08287642151119'],
            ['Huginn','1963.9920117333531','186.5715059141318','167.91435532271862','192.8212538175284']]),
        'The statistical screen does not guarantee each token deadline. Maximum ITL and individual deadline misses remain in the raw analysis.\n',
        'Finite-cohort goodput includes the full drain interval. It is not sustainable capacity.\n']
    text += ['## 4. Whole-request D-local mechanism and correctness\n',
        'A selected request starts prefill on D. It never starts prefill on P.\n',
        'D retains the request for decode. The local request has no intermediate P-to-D KV handoff.\n',
        'The input still reaches D through the host control path. Native requests retain native NIXL transfer and ACK handling.\n',
        'All-local N8 tests precede N32 qualification. N32 means simultaneous submission, not 32 simultaneous GPU residents.\n',
        'The gate compares every output token, valid prompt KV plane, prompt-last hidden state, finish reason, and fixed depth.\n',
        'Runtime reference parity is not original HF equivalence. Original HF equivalence remains NOT_MEASURED.\n',
        f'CPU contract checks: [cpu_regression_final.json]({PREFIX}cpu_regression_final.json).\n',
        f'Exactness records: [qualification.csv]({PREFIX}qualification.csv). Work and retirement checks: [work_accounting.csv]({PREFIX}work_accounting.csv).\n',
        'The early paced path failed because a partial fused prelude did not retain the complete staged hidden chunk.\n',
        'The N32-qualified path uses the native zero-loop embedding stage before bounded recurrent steps.\n',
        'The failed namespace remains blocked. Its timings did not enter the serving comparison.\n']
    primary=[]
    for model in ('ouro','huginn'):
        names=[f'local_{model}_native_n128_s2026110701_off',f'serve_{model}_native_paired_full32',
            f'serve_{model}_full_c128_fixed25_v5',f'serve_{model}_chunked_c128_fixed25_v5',f'serve_{model}_paced_c128_fixed25_v5',
            f'serve_{model}_paced_heavy_fixed25_'+('clean_restart_v2' if model=='ouro' else 'v1')]
        for name in names:
            r=next((v for v in serving if v['name']==name),None)
            if r:primary.append([name,r['observed_requests'],n(r['offered_QPS']),n(r['TTFT_p95_ms']),n(r['TPOT_p95_ms']),n(r['ITL_p99_ms']),n(r['E2E_p95_ms']),n(r['joint_SLO_goodput']),r['decode_screen_pass']])
    text += ['## 5. Marker-free serving results\n',
        markdown(['Run','N','QPS','TTFT p95 ms','TPOT p95 ms','ITL p99 ms','E2E p95 ms','Joint goodput req/s','Decode screen'],primary),
        'Compare N32 naive-full with its paired N32 native control. Compare the N128 generic arms with N128 native.\n',
        'V5 uses common full-context decode-geometry warm-up and an eight-entry graph budget. V4 remains a tuning diagnostic.\n',
        'The generic controls fix 25% of request IDs on D. Their local active limit is one.\n',
        'That limit includes the local decode lifetime. It can serialize local requests.\n',
        'Strict ready-decode priority can also starve prefill. A passing global decode screen does not prove useful concurrent borrowing.\n',
        'These controls are not the official Kairos algorithm or a complete deadline-prediction policy.\n',
        'Their poor latency must not become a structural rejection of every optimized generic deflection policy.\n',
        'The measured generic controls did not improve the primary Native comparison. They do not establish GENERIC_ONLY success.\n',
        f'All latency percentiles, rates, misses, failures, and token IDs: [serving_runs.csv]({PREFIX}serving_runs.csv) and `raw/*/requests.jsonl`.\n']
    text += ['## 6. Why native PD protects ITL\n',
        'Native D does not execute prompt slices. A migrated or local prefill sequence can block newly ready decode on the same execution stream.\n',
        'Ready-queue priority cannot preempt work that the runtime already submitted. Average utilization cannot establish a safe insertion deadline.\n',
        'The naive-full control isolates this interference without intermediate KV migration for local requests.\n',
        'The serial slice measurements also show the effect of recurrent quantum. They do not prove online safe windows.\n',
        f'Slice distributions: [slice_costs.csv]({PREFIX}slice_costs.csv).\n',
        'Historical Token/Loop paths also changed runtime, transfer, and graph costs. The new controls do not attribute every historical difference to one cause.\n']
    micro=[]
    for v in opportunities:
        if v.get('name','').endswith(('_alignment_v1','_d_device_v1')):
            micro.append([v['model'],v.get('chunk'),v.get('physical_GPU'),v['method'],n(v['local_first_ms']),n(v['cohort_host_ms']),n(v.get('TPOT_p95_ms')),n(v.get('ITL_p99_ms')),v['output_exact'],
                v.get('mixed_phase_different_depth_submissions','NOT_MEASURED'),
                v.get('whole_prefill_within_existing_decode_repetitions','NOT_MEASURED')+'/'+v['warm_repetitions']])
    text += ['## 7. Shared-weight batching and the stronger same-depth control\n',
        'The full-core grid uses legal ready states. It checks hidden rows and written query KV bytes.\n',
        'It includes causal attention and graph input staging. Initial metadata construction and outer scheduling remain outside the primitive timer.\n',
        'V5 keeps the decode batch intact and removes an unnecessary output concatenation from the timed separate control.\n',
        'Older state checks remain valid. Older separate timings are not primary.\n',
        'At identical input, weight pointer, row count, and graph, a dense depth label does not change arithmetic.\n',
        'Actual cross-depth value must therefore come from ready-work availability or scheduling, not the label itself.\n',
        'The larger-chunk control includes C128 and C248. Each method gets the same graph and token budgets.\n',
        'The stronger same-depth control executes at most one legal prefill step to remove a depth-distance-one alignment wait.\n',
        'That step can delay ready decode. Its measured TPOT and ITL remain visible. It is not strict priority.\n',
        markdown(['Model','Chunk','Physical GPU','Method','Local first token ms','Cohort ms','BG TPOT p95 ms','BG ITL p99 ms','Output exact','Mixed-phase different-depth submissions','Local prefill inside BG lifetime'],micro),
        'These are standalone closed-loop controls with two background decode streams and one local prompt. They are not online PD goodput.\n',
        'The local micro request produces two tokens. Background caps stay 32 for Ouro and 16 for Huginn.\n',
        'The separate ready-state primitive grid caps its background streams at eight tokens. Those cells are not primary serving workloads.\n',
        'The initial micro device is physical GPU 0 or 4. The D-device confirmation uses physical GPU 1 or 5.\n',
        'The lifetime column counts whole local-prefill intervals inside existing background token-to-finish intervals. It is not a GPU idle-time measurement.\n',
        'Independent C32 starts have separate receipts. Their paired confidence intervals are preliminary and do not establish online capacity.\n',
        f'Primitive data: [matched_core_micro.csv]({PREFIX}matched_core_micro.csv). Dense controls: [dense_micro.csv]({PREFIX}dense_micro.csv).\n',
        f'Readiness controls: [readiness_opportunity.csv]({PREFIX}readiness_opportunity.csv). Paired-start intervals: [phase_c_gate.json]({PREFIX}phase_c_gate.json).\n',
        'The qualified numerical patch retains 256-row tiles for sensitive projections. Ordinary batching can amortize that extra work.\n',
        'This runtime effect is not a new loop-specific contribution. Actual weight-read bytes remain NOT_MEASURED.\n']
    primitive=[]
    for v in rows('matched_core_micro.csv'):
        if (v.get('primary_timing_eligible')=='True' and v['chunk']=='32' and v['decode_batch']=='2'):
            primitive.append([v['model'],v['depth_class'],n(v['separate_p50_ms']),n(v['batched_p50_ms']),
                n(100*(1-float(v['batch_over_separate']))),v.get('graph_eager_fallbacks','NOT_MEASURED')])
    text += [markdown(['Model','Depth compatibility','Separate core ms','Batched core ms','Batch saving %','Eager fallbacks'],primitive),
        'Each row compares the same ready work with separate or batched execution. Same-depth and cross-depth rows can have different attention metadata.\n',
        'The matched dense-label control instead keeps the tensor and graph identical. It isolates the depth label from attention geometry.\n']
    profiles=rows('micro_kernel_diagnostics.csv')
    text += [markdown(['Model','Diagnostic method','Kernels','Summed recorded kernel ms','Named attention ms','Named dense ms'],[
        [v['model'],v['method'],v['kernel_count'],n(v['profiled_kernel_sum_ms']),
            n(v['named_attention_kernel_sum_ms']),n(v['named_dense_kernel_sum_ms'])] for v in profiles]),
        'These profiler diagnostics support kernel attribution only. Their times do not establish clean online speedup or exclusive device occupancy.\n',
        f'Profile scopes: [micro_kernel_diagnostics.csv]({PREFIX}micro_kernel_diagnostics.csv).\n']
    text += ['## 8. Slack evidence and instrumentation limits\n',
        'Collector coverage and low-overhead serving are separate gates. Zero dropped kernels do not establish acceptable overhead.\n',
        'The current kernel-activity traces failed the matched local-tail screen. They remain diagnostic, not clean slack proof.\n',
        'CUDA-event model intervals do not replace exclusive kernel activity. SM cycles, achieved bandwidth, and qualified clean kernel gaps remain NOT_MEASURED.\n',
        'First-entry markers provide diagnostic Service TTFT. Marker-free runs do not contain a measured first-model timestamp.\n',
        'Their Service TTFT and queue decomposition therefore remain NOT_MEASURED. Do not fabricate those components from host admission.\n',
        'Per-request marked decompositions retain all later handoff and continuation costs. Component p95 values must not be added.\n',
        f'Interference records: [instrumentation_qualification.csv]({PREFIX}instrumentation_qualification.csv).\n',
        f'First-entry marker checks: [service_marker_qualification.csv]({PREFIX}service_marker_qualification.csv).\n',
        f'Sampled retirement lower bounds: [sampled_retired_overlap.csv]({PREFIX}sampled_retired_overlap.csv).\n',
        'A zero sampled overlap bound does not prove that no overlap occurred. Sparse snapshots can miss retired boundaries.\n',
        'Do not combine a separate P saturation run with a standalone D microbenchmark into a same-run success claim.\n']
    diagnostic=[]
    for r in serving:
        if r['name'].endswith(('service_once128','service_pool128')):
            diagnostic.append([r['name'],n(r.get('TTFT_p95_ms')),n(r.get('queue_p95_ms')),n(r.get('service_TTFT_p95_ms')),'DIAGNOSTIC_ONLY'])
    text += [markdown(['Marked native run','TTFT p95 ms','Pre-model latency p95 ms','Service TTFT p95 ms','Eligibility'],diagnostic),
        'These components come from per-request timestamps. Their p95 values do not add to TTFT p95.\n']
    text += ['## 9. KV, HBM, and work accounting\n',
        'Each D-local request retains full fixed-depth KV. Removing intermediate KV handoff does not remove local KV memory.\n',
        'Huginn retains its physical dead-plane cost. The campaign does not apply ExactLoop-style KV compaction.\n',
        'Coordinator reservation credit, allocated KV, compute readiness, and active execution are separate quantities.\n',
        'Memory tables use GiB and a stated HBM denominator. Unique-storage components overlap allocator totals and must not be counted twice.\n',
        'Opaque graph and driver allocations are not fully attributed. NVML reports sampled memory peaks, not continuous exact peaks.\n',
        f'Memory: [gpu_memory_inventory.csv]({PREFIX}gpu_memory_inventory.csv). Credits and host readiness: [system_summary.csv]({PREFIX}system_summary.csv).\n',
        'Real recurrent token rows are not FLOPs. They exclude prelude, coda, and numerical-padding work.\n',
        'Local completion counts must not include old independent 16-token probes or migrated P requests.\n',
        'The runtime records transfer byte counters. Summed transport-observation durations are not exclusive communication time.\n']
    memory=rows('gpu_memory_inventory.csv');memory_table=[]
    for model in ('ouro','huginn'):
        selected=[v for v in memory if v['name']==f'local_{model}_native_n128_s2026110701_off']
        for role in ('prefill','decode'):
            group={v['component']:v for v in selected if v['role']==role}
            def component(key):
                v=group.get(key)
                return n(v['GiB'])+' / '+n(v['HBM_percent'])+'%' if v else 'NOT_MEASURED'
            memory_table.append([model,role,component('weight_and_model_buffer_storage_bytes'),component('KV_storage_bytes'),
                component('exposed_runner_workspace_storage_bytes'),component('exposed_NIXL_hidden_staging_storage_bytes'),component('allocator_allocated_bytes')])
    text += [markdown(['Model','Role','Weights GiB / HBM %','KV pool GiB / HBM %','Workspace GiB / HBM %','NIXL staging GiB / HBM %','Allocator GiB / HBM %'],memory_table),
        'These percentages use NVML physical HBM. CUDA allocatable capacity is a separate field in the memory snapshots.\n']
    local_memory=[]
    for model in ('ouro','huginn'):
        value=next((r for r in rows('work_accounting.csv') if r['name']==f'serve_{model}_paced_c128_fixed25_v5'),None)
        if value:
            local_memory.append([model,value['selected_local_requests'],value['actual_local_serving_requests'],
                value['max_single_local_reserved_blocks'],n(value.get('single_local_reservation_GiB')),
                n(value.get('single_local_reservation_physical_HBM_percent')),value.get('pool_only_theoretical_local_reservations'),
                value['full_fixed_work_exact'],value['final_device_requests_and_KV_blocks_empty']])
    text += [markdown(['Model','Selected local','Retired local','Quota blocks/request','Quota GiB/request','Quota / HBM %','Pool-only quotient','Fixed rows exact','Final device/KV empty'],local_memory),
        'The pool-only quotient is structural accounting from measured block geometry. It ignores native reservations and compute-slot limits.\n',
        'A quota is not measured live attention bytes. Unused physical KV planes remain allocated.\n']
    text += ['## 10. Prior work and novelty boundary\n',
        '[Load-Aware Prefill Deflection](https://arxiv.org/abs/2607.02043) already studies whole-request D placement and decode-aware prefill chunks. The fixed controls here are not its official system.\n',
        '[FluidPD](https://arxiv.org/abs/2610.06917) uses pressure-aware mechanisms, including prefix/suffix work transfer. This PoC does not run FluidRole or FluidToken.\n',
        '[CDB](https://arxiv.org/abs/2608.09444) already uses tied-core work at different recurrent depths. Native RLT also exposes this capability.\n',
        '[ExactLoop](https://arxiv.org/abs/2611.05150) studies exactness-graded runtime optimizations. Their symmetric application and remaining headroom are not measured here.\n',
        'The reference LoopServe code uses a different vLLM V1 ABI. Its state structures were not substituted into RLT.\n',
        'Combining deflection and existing CDB does not, by itself, establish a new method.\n',
        markdown(['Prior work','Relevant mechanism','Use in this PoC','Unverified extension'],[
            ['Load-Aware Prefill Deflection','Whole-request D placement and decode-aware chunks','Placement control and priority baseline','Official policy and calibrated deadline prediction'],
            ['FluidPD / FluidToken','Pressure-aware prefix/suffix transfer','Comparison boundary only','Official FluidToken execution'],
            ['CDB','Shared-core work at different recurrent depths','Native RLT depth mixing and matched controls','New contribution beyond existing depth scheduling'],
            ['ExactLoop','Exactness-graded runtime optimization','Exactness boundary and source review','Symmetric optimization ablation']]),
        f'Detailed audit: [source_and_prior_work_audit.md]({PREFIX}source_and_prior_work_audit.md).\n']
    text += ['## 11. Failures, missing tests, and restart\n',
        'Warm-up ownership, post-admission coda readiness, and partial-prelude state retention required separate corrections. All failed logs remain.\n',
        'A daemon restart removed dispatch controllers. One worker completed its cohort without its dispatcher receipt.\n',
        'Its recovered receipt has an unknown process exit code and an unqualified timing label. The raw cohort is not a clean primary comparison.\n',
        'A fresh named heavy control avoids reusing that interrupted timing as primary. The original result remains preserved.\n',
        'The campaign omitted optional Huginn C32 N32 cells to prioritize mandatory C128 N32 and warm-matched serving. Their N8 tests remain.\n',
        'Cross-depth online N128, N512 persistence, natural-EOS variability, and the ExactLoop ablation did not pass the expansion gate.\n',
        'The campaign does not claim a complete calibrated generic deflection controller or a production scheduler.\n',
        f'Every execution and missing configuration: [configuration_matrix.csv]({PREFIX}configuration_matrix.csv). Failures: [failure_inventory.csv]({PREFIX}failure_inventory.csv).\n',
        f'Full measurement limits: [measurement_contract_and_limitations.md]({PREFIX}measurement_contract_and_limitations.md).\n']
    figures=json.loads((ROOT/'figure_manifest.json').read_text())
    text += ['## 12. Figures and reproduction\n']
    for f in figures:text.append(f'- [{f["name"]}]({PREFIX}{f["PNG"]}). {f["scope"]}\n')
    text += ['\nUse the qualified private runtime and the pinned checkpoints. Existing names return saved receipts, not fresh experiments.\n',
        'Use a fresh configuration name for a real rerun. Do not overwrite a prior namespace.\n',
        '```bash\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.run NEW_RUN_NAME\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.analyze\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.micro_analysis\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.figures\n```\n',
        '## 13. Publication and resource restoration\n',
        f'Branch: `{BRANCH}`. The working checkout and real staged index remain untouched.\n',
        f'The committed [publication receipt]({PREFIX}state/publication_receipt.json) records the evidence commit and verified remote SHA.\n',
        f'The local [final remote check]({PREFIX}state/final_remote_verification.json) records the final metadata commit.\n',
        f'Burner restoration and actual GPU-process checks: [resource restoration]({PREFIX}state/final_resource_restoration.json).\n']
    text += ['The first GPU 1 burner startup met a TCPStore port collision. The retry leaves the other seven burner sessions running.\n',
        f'Resource recovery evidence: [burner_startup_retry.json]({PREFIX}state/burner_startup_retry.json).\n']
    FINAL.write_text('\n'.join(text))
    print(str(FINAL))


if __name__=='__main__':main()
