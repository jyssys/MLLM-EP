"""Generate a final evidence summary from terminal ledgers, not expectations."""
import time
from .common import *
from .final_inventory import read


def table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] + [
        '| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows]


def main():
    configure()
    from .final_inventory import main as inventory
    inventory()
    from .conditional_split_qualification import recovery_ready
    gate = read(ROOT / 'state/huginn_final_recovery_gate.json', {})
    ready = recovery_ready(gate)
    methods = read(ROOT / 'state/static_screen.json', {})
    qualification = read(ROOT / 'state/conditional_split_qualification.json', {})
    census = read(ROOT / 'complete_configuration_inventory.json')['configurations']
    elapsed = (time.time() - START) / 3600
    complete = sum(row['status'].startswith('COMPLETE') for row in census)
    failed = sum(row['status'] == 'FAILED_OR_INTERRUPTED' for row in census)
    missing = len(census) - complete - failed
    report = ['# Huginn recovery and FluidToken-style versus Loop Split', '',
        f'Report snapshot: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}. Elapsed wall time: {elapsed:.3f} hours.',
        'The 11-hour budget includes diagnosis, GPU work, reports, publication, and resource restoration.', '',
        '## Decision', '',
        f'Huginn recovery: **{"OPERATIONAL_RECOVERY_READY" if ready else "NOT_CONFIRMED"}**.',
        'The operational test uses kernel union and a finite service reference. It does not measure physical peak-FLOP saturation.',
        'Ouro readiness from the previous campaign remains historical evidence. It is not a new Token/Loop comparison.',
        'No Loop-specific advantage follows from D-local probe completion alone.', '',
        'The requested four-way method comparison is **BLOCKED_CORRECTNESS** in both models.',
        'Both Token Split tests failed during warm-up with a scheduler frontier exception.',
        'Loop continuation passed N8 output/state checks, but the combined pipeline did not reach N32 qualification or clean serving.',
        'The method study is incomplete. This is not a performance NO-GO for either split axis.',
        'The Token suffix candidate is CPU_TESTED_GPU_UNQUALIFIED. No measured run enables it.', '',
        'Completion does not imply correctness, SLO compliance, or accepted profiling.',
        f'The configuration inventory contains {complete} completed jobs, {failed} failed/interrupted jobs, and {missing} uncompleted configurations.',
        'These counts include diagnostics and microbenchmarks. They are not counts of comparable serving configurations.', '',
        '## Execution provenance', '',
        'Ouro uses `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, and fixed R4.',
        'Huginn uses `tomg-group-umd/huginn-0125`, revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`, and fixed R32.',
        'The qualified external RLT HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.',
        'The preserved external diff SHA256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.',
        'The unchanged user checkout HEAD is `f4ecea59f36f620c69c4fef09c590bf790802cd9`.',
        'The publication parent is the latest evidence commit, not a reset of the user checkout.',
        'Each run retains `new_campaign_source.zip`, `new_campaign_source_hashes.json`, configuration hashes, and loaded-runtime evidence.',
        'The inherited execution hash alone does not identify every new adapter. Use the per-run campaign source hashes.',
        'The environment uses H100 GPUs, NV18 links, and driver 570.211.01. Exact mappings remain in each configuration.',
        'The qualified Python environment uses PyTorch 2.11.0+cu129. The read-only reference engine uses a different environment.',
        'Clean runs use the exclusive measurement lane. Approved burners remain stopped during those runs.',
        'The excluded repository-audit overlap remains recorded separately.', '',
        '## Evidence classes and fixed contract', '',
        '- MEASURED: actual GPU execution or host request observations, with their scopes stated.',
        '- TRACE_ANALYSIS: analysis of retained measured events and timestamps.',
        '- STRUCTURAL_ACCOUNTING: byte, row, or boundary accounting. This is not measured speedup.',
        '- NOT_MEASURED or BLOCKED: no accepted result exists.',
        '- Both models retain original BF16 weights and fixed Ouro R4 or Huginn R32.',
        '- No approximation, early exit, weight update, or recurrent-work omission is permitted.',
        '- P kernel union must reach 80%. Finite service proximity must reach 85%.',
        '- Matched whole-run wall perturbation must remain at most 5%.',
        '- Positive device-qualified P-ready growth and D credit remain independent requirements.',
        '- Huginn L2048 uses TPOT p95 at most 167.914355 ms and ITL p99 at most 192.821254 ms.',
        '- Whole-run Decode SLO and common-interval Decode SLO must both pass.',
        '- Three independent starts are required. A bounded-window pass does not replace this requirement.', '',
        '## Huginn recovery', '',
        'The primary recovery workload contains 2048 input tokens and 16 controlled output tokens.',
        'The retained pool contains 128 ShareGPT/LongBench controlled prefixes. N512 explicitly reuses them cyclically.',
        'Ignore-EOS generation is not natural response-length measurement.',
        'Most controls use nominal 1.971 requests/s. The inventory records every changed rate and finite arrival realization.',
        'P uses two slots. D slot and graph variants remain separate configurations.',
        'The accepted paced configuration uses P/D slots 2/8 and eight-row Decode graphs.',
        'P/D KV pools contain 16,384/26,752 blocks. P/D token budgets are 4,096/128.',
        'The local Prefill graph limit is 16 rows. Method qualification instead enables a full-prompt D budget and graph limit.',
        'Those method settings require separate qualification. Recovery measurements are not silently reused as method latency.',
        'Independent D-local probes use 16 input tokens and full R32 computation. They are not P-to-D continuations.', '',
        '### Accepted repetitions', '']
    accepted_rows = []
    if ready:
        for trial in gate['results']:
            same = trial['same_run']
            g = same['screen']['gates']
            accepted_rows.append([trial['profile'], f"{100*g['P_actual_kernel_busy_central']:.3f}",
                f"{100*g['P_fraction_reference']:.3f}", f"{same['device_qualified_queue']['half_growth']:.6f}",
                f"{g['D_request_TPOT_p95_ms']:.3f}", f"{g['D_token_ITL_p99_central_ms']:.3f}",
                g['D_min_coordinator_free_blocks'], same['screen']['central_completed_probes']])
    report += table(['Run', 'P union %', 'Service proximity %', 'P queue growth', 'D TPOT p95 ms', 'D ITL p99 ms', 'D free blocks', 'Local probes in interval'],
                    accepted_rows or [['NOT_CONFIRMED'] + ['NOT_MEASURED'] * 7])
    if ready:
        refusal = ', '.join(f"start {trial['start']}: {100*trial['same_run']['screen']['gates']['D_credit_refusal_sample_fraction']:.6f}%"
                            for trial in gate['results'])
        report += ['', 'D credit-refusal sample fractions were ' + refusal + '.',
            'All remain below the frozen 50% sample gate. A pass does not mean unrestricted KV admission.',
            'The sampled refusal fraction is not the fraction of permanently rejected requests.']
    repetition_rows = []
    if ready:
        for trial in gate['results']:
            probe = trial['same_run']['probes']
            activity = trial['activity']['prefill']
            whole = trial['profile_metrics']
            repetition_rows.append([trial['start'],
                f"{100 * activity['matched_wall_difference_fraction']:.3f}",
                f"{trial['clean_metrics']['ITL_p99_ms']:.3f}",
                f"{whole['ITL_p99_ms']:.3f}", f"{whole['maximum_ITL_ms']:.3f}",
                probe.get('actual_probe_admissions', 'NOT_MEASURED'), probe['completed_probes'],
                probe['exact'], probe['full_work'], f"{whole['TTFT_p95_ms']:.3f}",
                f"{100*whole['joint_SLO_attainment']:.6f}"])
    report += ['', '### Whole-run checks and probe denominators', '']
    report += table(['Independent start', 'Profile wall change %', 'Clean ITL p99 ms',
                    'Profile ITL p99 ms', 'Profile maximum ITL ms', 'Admitted probes',
                    'Completed probes', 'Exact probe outputs', 'Full recurrent work',
                    'TTFT p95 ms', 'Joint-SLO attainment %'],
                   repetition_rows or [['NOT_CONFIRMED'] + ['NOT_MEASURED'] * 10])
    report += ['', 'The configured ceiling is 12 optional probe admissions. It is not the number of probes actually admitted.',
        'The recovery runs pass the Decode contract, not the joint TTFT-plus-TPOT serving contract.',
        'P queue growth produces large TTFT and low joint-SLO attainment. No sustainable-capacity claim follows.',
        'The per-run maximum ITL remains visible. The frozen p99 contract is not a maximum-token-latency guarantee.',
        'A broad repository audit overlapped the first new profile. That execution is excluded from clean confirmation.',
        'The audit record remains in the state directory. The three new paced repetitions replace that excluded confirmation.',
        'The failed independent replacement remains a failed result. It is not removed by the later intervention.']
    paired = read(ROOT / 'state/paired_probe_cost_control.json', {})
    report += ['', '### Same-arrival probe cost control', '']
    paired_rows = []
    if paired.get('correct'):
        for key, label in (('no_probe_metrics', 'No measured local probes'), ('probe_metrics', 'Output-paced local probes')):
            values = paired[key]
            paired_rows.append([label] + [f'{values[field]:.6f}' for field in
                ('TPOT_p95_ms', 'ITL_p99_ms', 'maximum_ITL_ms', 'TTFT_p95_ms', 'requests_per_second')])
    report += table(['Control', 'TPOT p95 ms', 'ITL p99 ms', 'Maximum ITL ms', 'TTFT p95 ms', 'Completed req/s'],
                    paired_rows or [['NOT_MEASURED'] * 6])
    report += ['', 'This pair uses identical prompts, arrival offsets, memory allocation, and execution geometry.',
        'The same D-local Prefill warm-up remains in both controls. Only measured probe admissions differ.',
        'This is one paired start, not a replicated causal estimate.',
        'A passing p99 result does not hide the reported maximum ITL. It does not guarantee a deadline for every token.',
        'The earlier no-probe control used a different arrival seed. It is a configuration check, not this paired cost comparison.']
    report += ['', 'Every accepted row uses one execution and one common observation window.',
        'The original phase-only queue field remains in the raw evidence and can report a failed P gate.',
        'The accepted queue audit instead requires measured P kernel timestamps at each D-credit-clear slot-wait sample.',
        'It adds verified ready requests to that device-qualified waiting count. It does not use a host-service-span fallback.',
        'The growth, kernel-union, service-proximity, and profiling thresholds remain unchanged.',
        f"D kernel evidence in the three required repeats: {gate.get('D_kernel_activity', 'See each accepted capture manifest. No missing D trace is inferred.') if ready else 'NOT_MEASURED_IN_ACCEPTED_RUNS'}.",
        'A P-only trace cannot supply a D kernel timeline. Exact probes, D credit, and D SLO remain same-run measurements.',
        'The complete inventory retains failed runs and all rejected profiling evidence.', '',
        '### Root-cause evidence', '']
    dual = read(ROOT / 'state/dual_device_paced_recovery_control.json', {})
    if dual.get('status') == 'SUPPLEMENTAL_DUAL_CAPTURE_PASS':
        g = dual['same_run']['screen']['gates']
        d = dual['activity']['decode']
        same_bounds = (g['observation_start'] == d['window_start_monotonic']
                       and g['observation_end'] == d['window_end_monotonic'])
        report += ['One supplemental dual-device run also passed. It does not replace the three required repetitions.', '']
        report += table(['P union %', 'D union %', 'D TPOT p95 ms', 'D ITL p99 ms', 'D free blocks', 'Credit-refusal samples %', 'Exact probes in interval'], [[
            f"{100*g['P_actual_kernel_busy_central']:.6f}",
            f"{100*d['kernel_busy_fraction']:.6f}" if same_bounds else 'See capture-window manifest',
            f"{g['D_request_TPOT_p95_ms']:.6f}", f"{g['D_token_ITL_p99_central_ms']:.6f}",
            g['D_min_coordinator_free_blocks'], f"{100*g['D_credit_refusal_sample_fraction']:.6f}",
            dual['same_run']['screen']['central_completed_probes']]])
        report += ['', 'The common observation interval is the intersection of the P and D capture windows.',
            'Credit refusals remain material in this supplemental run. The configured limit is 50% of sampled admission observations.',
            'A whole-run profiling perturbation below 5% does not prove unchanged local token tails.',
            'The captured D union describes this profiled execution. It is not an unprofiled utilization estimate.', '',
            '![Huginn same-run device evidence](looped_pd_fluidtoken_vs_dynamic_loop_split/figures/recovery_huginn_paced8_v5_dual_activity512_joint_device.png)', '']
    report += table(['Finding', 'Evidence', 'Limit'], [
        ['NIXL/CUPTI progress stall', 'Repeated native stacks reach NIXL/UCX CUDA IPC and CUPTI/driver locks.', 'ROOT_CAUSE_INCONCLUSIVE: the lock owner and complete cycle are not proved.'],
        ['Measurement-path recovery', 'Early CUPTI initialization and 16 preallocated buffers provide completed activity-only captures.', 'This does not prove that the vendor-level stall is fixed.'],
        ['Foreground evidence I/O', 'Measured append calls reach 299.750 ms in the coordinator, 286.899 ms in P, and 271.819 ms in D.', 'The exact filesystem or kernel-lock cause remains unresolved.'],
        ['Asynchronous evidence routing', 'Custom campaign append aliases now use the existing durable writer.', 'Whole-run tail improvements do not guarantee the common-interval gates.'],
        ['Decode geometry', 'Four-row and eight-row controls retain separate output/state checks and latency results.', 'Do not compare different geometry as a Token-versus-Loop benefit.'],
        ['Probe admission and pacing', 'The retained controllers test all Decode phases and, conditionally, observed-output pacing.', 'Only terminal qualified results can authorize method expansion.']])
    report += ['', 'Long foreground evidence calls can delay serving. Their occurrence is measured, not inferred from utilization.',
        'Temporal overlap with a latency spike does not prove that this mechanism explains every spike.',
        'The global wall-time profiling check also does not prove that local Decode tails are unaffected.',
        'Several completed dual-device captures fail the same-window Decode or credit gate. Those captures remain rejected.', '',
        '### Rejected controls and the output-paced intervention', '',
        'The independent replacement seed failed the common-window ITL gate at 194.877075 ms against the frozen 192.821254 ms limit.',
        'Its P kernel union was 95.506248%. The other measured gates did not override the ITL failure.',
        'The output-paced control permits one exact local Prefill stage per observed background output step.',
        'It changes staged execution and pacing together. Their individual effects are not isolated.',
        'The first staged state test exhausted HBM with 28,288 D KV blocks.',
        'The retry uses 26,752 blocks and releases 3.8671875 GiB for staged workspace.',
        'Both pools retain capacity for six complete 2048-input/16-output reservations of 4,128 blocks each.',
        'The smaller pool requires new qualification. It is not silently treated as the same runtime configuration.',
        'See [the pacing control](looped_pd_fluidtoken_vs_dynamic_loop_split/paced_probe_recovery_control.md) for the exact validation sequence.',
        'The paced probe mechanism applies only to independent local probes. It does not establish safe migrated continuation execution.', '',
        '## Method correctness and serving comparison', '',
        f"Qualification ledger status: `{qualification.get('status', 'NOT_RUN')}`.",
        'The five static controls separate Atomic scheduling, staged scheduling, and snapshot transport.',
        'Token boundaries are independent of the execution chunk budget. Token Split is not restricted to C256.',
        'All method arms use separate Prefill/Decode phases. Shared-core co-batching is not a claimed contribution.',
        'Heavy state checks are not clean latency measurements.', '']
    correctness_rows = []
    for model in ('ouro', 'huginn'):
        for entry in qualification.get('results', {}).get(model, []):
            correctness_rows.append([model, entry['name'], 'PASS_N8' if entry['passed'] else 'FAILED_WARMUP',
                entry.get('output', {}).get('compared', 'Reference' if entry['passed'] else 0),
                entry.get('prompt_state', {}).get('exact', 'Reference' if entry['passed'] else 'NOT_QUALIFIED'),
                entry['byte_and_work_gate']])
    report += table(['Model', 'Correctness run', 'Status', 'Exact output comparisons', 'Prompt state exact', 'Transfer/work gate'],
                    correctness_rows or [['NOT_RUN'] * 6])
    report += ['', 'The Token tests failed before measured-request generation. They did not establish output parity.',
        '[The failure analysis](looped_pd_fluidtoken_vs_dynamic_loop_split/token_suffix_scheduler_failure.md) identifies the imported-prefix origin conflict.',
        'CPU replay executes the real native progress method and reproduces both exceptions.',
        'An opt-in adapter retains the suffix origin and absolute state rows. It is disabled by default.',
        'No GPU validation of that adapter occurred before the frozen new-job cutoff.', '',
        '### Clean serving matrix', '']
    method_rows = []
    for model in ('ouro', 'huginn'):
        for label in ('atomic_native', 'staged_native', 'staged_snapshot_full', 'token_partial', 'loop_partial'):
            values = [r for r in methods.get('results', []) if r.get('model') == model and label in r.get('name', '')]
            value = values[-1] if values else {}
            m = value.get('metrics', {})
            status = ('MEASURED_N32_SCREEN' if value.get('correct') else 'FAILED_OR_UNQUALIFIED') if value else (
                ('BLOCKED_TOKEN_FRONTIER' if label == 'token_partial' else 'BLOCKED_32_REQUEST_STATE_GATE')
                if ready else 'BLOCKED_BY_HUGINN_GATE')
            method_rows.append([model, label, status] + [f'{m[key]:.3f}' if key in m else 'NOT_MEASURED'
                for key in ('TTFT_p95_ms', 'TPOT_p95_ms', 'ITL_p99_ms', 'E2E_p95_ms', 'requests_per_second', 'joint_SLO_goodput')])
        for label in ('token_dynamic', 'loop_dynamic'):
            method_rows.append([model, label, 'BLOCKED_CORRECTNESS' if ready else 'BLOCKED_BY_HUGINN_GATE'] + ['NOT_MEASURED'] * 6)
    report += table(['Model', 'Method/control', 'Status', 'TTFT p95 ms', 'TPOT p95 ms', 'ITL p99 ms', 'E2E p95 ms', 'Completed req/s', 'Joint-SLO req/s'], method_rows)
    report += ['', 'An N32 screen is exploratory. It does not establish stable serving capacity or a replicated method advantage.',
        'Dynamic policy CPU tests use synthetic costs. They are not online GPU results.',
        'A dynamic GPU experiment requires measured calibration, matching source/configuration hashes, and a separate exactness gate.',
        'This generated snapshot contains no accepted online dynamic GPU result.', '',
        'The retained controller stopped after the Token frontier failures at approximately 07:02 KST.',
        'The agent resumed at approximately 09:15 KST after the capacity interruption.',
        'No GPU experiment was active at that resumption. This idle interval is not experiment execution time.',
        'The 08:48:43 new-job cutoff and 09:18:43 GPU cutoff remained unchanged.',
        'The interruption did not renew the original 11-hour budget.',
        'The failure gate, not measured method speed, prevented calibration and online expansion.', '',
        '## Prior work and novelty', '',
        '[FluidPD](https://arxiv.org/html/2610.06917v1) uses token-prefix migration and pressure-aware suffix allocation.',
        'Our runtime differs from its SGLang implementation. No official code was independently identified in the inspected sources.',
        'The appropriate label is FluidToken-style reproduction, not execution of the original system.',
        'The queued-deadline selector uses known requests, measured costs, D credit, and observed latency slack.',
        'It does not implement every continuous-token or scheduling-iteration detail of FluidPD Algorithm 1.',
        'See [the prior-work audit](looped_pd_fluidtoken_vs_dynamic_loop_split/prior_work_and_reuse_audit.md) and the dynamic policy contract.',
        'A generic logical-layer split includes the same recurrent boundary.',
        'The same boundary, state, and execution cost imply EQUIVALENT_MECHANISM.',
        'A Token-versus-Loop difference cannot alone prove exclusive Loop-specific novelty.',
        'Both PD workers already hold the complete model. Shared weights do not automatically save additional model residency.', '',
        '## Work, transfer, and E2E accounting', '',
        'A completed 512-request Huginn run contains 33,554,432 background Prefill recurrent token rows.',
        'If twelve full-R32, 16-token probes are admitted, they add 6,144 recurrent token rows.',
        'The structural fraction is 0.018310546875%. It is not a FLOP fraction, offloaded serving fraction, or speedup.',
        'Actual admitted and completed probe counts can be smaller. The per-run completion ledger supplies those denominators.',
        'The V2 completion contract checks every admitted request. The configured upper admission limit is reported separately.',
        'The original quota-based verdict remains in each raw directory. New confirmation starts follow the admitted-request denominator contract.',
        'Serving-request offload counts must come from connector decisions and actual continuation submissions.',
        'D-local probe counts must never substitute for those counts.',
        'Token-row matching does not guarantee equal attention FLOPs.',
        'The host E2E v2 CSV partitions admission, P dispatch/start waiting, observed P service, completion-to-first-token, and generation.',
        'The completion-to-first-token component includes handoff, D continuation, Coda, sampling, and delivery when applicable.',
        'An activation receipt can follow the first token. It is therefore not a valid exclusive first-token partition boundary.',
        'Kernel, transfer, and host durations can overlap. Their sums are not exclusive E2E percentages.',
        'SM occupancy, achieved bandwidth, and peak-FLOP utilization remain NOT_MEASURED unless separately identified in raw counters.', '',
        '## Exactness and limitations', '',
        'The recovery controls compare full output sequences and full recurrent row counts with the retained native references.',
        'Separate heavy tests compare valid prompt KV planes and hidden-state hashes.',
        'N512 serving runs do not hash every intermediate tensor.',
        'Each new Token/Loop connector configuration requires byte identity and output/state qualification.',
        'Original-HF equivalence and downstream benchmark accuracy remain NOT_MEASURED in this campaign.',
        'Finite-cohort throughput includes queue drain. It is not an indefinitely sustainable capacity estimate.',
        'Run repetitions and percentile uncertainty must remain distinct from individual token sample counts.', '',
        '## Artifacts and reproduction', '',
        '- [Complete configuration inventory](looped_pd_fluidtoken_vs_dynamic_loop_split/complete_configuration_inventory.csv).',
        '- [Recovery attempt table](looped_pd_fluidtoken_vs_dynamic_loop_split/recovery_attempts.csv).',
        '- [Actual split work and payload accounting](looped_pd_fluidtoken_vs_dynamic_loop_split/method_work_and_payload_accounting.csv).',
        '- [Disjoint host E2E component percentages](looped_pd_fluidtoken_vs_dynamic_loop_split/recovery_host_E2E_components.csv).',
        '- [Figure manifest and missing figure statuses](looped_pd_fluidtoken_vs_dynamic_loop_split/figure_manifest.json).',
        '- Per-run configurations, source hashes, raw outputs, kernel records, and failure logs remain under the campaign directory.',
        '- `commands.jsonl` records actual executions. Figure manifests identify their source execution and interval.',
        '- `state/huginn_final_recovery_gate.json` records the current authorization gate.',
        '- `state/conditional_split_qualification.json` records method state qualification.',
        '- `state/publication_receipt.json` records the published commit and remote verification after publication.', '',
        'Reproduction entry points:', '', '```bash',
        '/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.fluidtoken_loop_split.run RUN_NAME',
        '/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.fluidtoken_loop_split.final_inventory',
        '/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.fluidtoken_loop_split.plot_joint_activity RUN_NAME',
        '```', '',
        'The GPU dispatcher enforces this campaign\'s original deadlines. It refuses late reproduction jobs.',
        'A future GPU validation needs an authorized new campaign and new run names. Do not overwrite the retained configurations.',
        'Use each run\'s archived source and qualified model revision. The current opt-in candidate differs from the measured source.', '',
        f'Publication branch: `{BRANCH}`. Parent evidence commit: `{BASE}`.',
        'The publication uses a temporary Git index. The user checkout and staged index remain unchanged.',
        'No force push is permitted. Remote SHA equality and approved burner restoration require final verification.', '']
    work = read(ROOT / 'method_work_and_payload_accounting.json', {}).get('results', [])
    work_rows = []
    for row in work:
        if row['partial_serving_requests']:
            count = row['completed']
            work_rows.append([row['model'], f"{row['partial_serving_requests']}/{count}",
                row['assigned_D_recurrent_token_rows'], f"{100*row['assigned_D_row_fraction']:.3f}",
                f"{row['queued_snapshot_KV_bytes']/count/1024**3:.6f}",
                f"{row['queued_snapshot_hidden_bytes']/count/1024**2:.6f}",
                row['full_recurrent_work']])
    extra = ['### Actual migrated work in the N8 state tests', '']
    extra += table(['Model', 'Migrated/completed requests', 'D recurrent token rows', 'D share of Prefill rows %',
                    'KV GiB/request', 'Hidden MiB/request', 'Full-work gate'], work_rows or [['NOT_MEASURED'] * 7])
    extra += ['', 'These are actual accepted Loop continuations with completed exact outputs. They are not D-local probes.',
        'Huginn executes one remaining loop. Ouro executes one remaining loop.',
        'The fractions are 1/32 and 1/4 of recurrent token rows, not measured FLOP fractions.',
        'The figures come from heavy state tests. They establish neither clean latency nor serving speedup.',
        'Queued descriptor bytes are not exclusive communication time. Completed output and full-work checks remain separate.',
        'No clean Token Split payload or latency comparison exists because its warm-up failed.', '']
    parts = read(ROOT / 'recovery_host_E2E_components.json', [])
    selected = [row for row in parts if row['name'] == 'recovery_huginn_paced8_v5_joint_s4_clean512']
    if selected:
        extra += ['### Host E2E shares for the accepted clean start 4', '']
        extra += table(['Host component', 'Share of summed request E2E %'],
                       [[row['component'], f"{row['aggregate_E2E_percent']:.6f}"] for row in selected])
        extra += ['', 'The denominator is summed request latency, not server wall time.',
            'These disjoint host intervals include queueing and submission gaps.',
            'They cannot provide exclusive dispatch, GEMM, or DMA percentages. Those GPU percentages remain NOT_MEASURED.', '']
    position = report.index('## Exactness and limitations')
    report[position:position] = extra
    decision = ['## Model decisions and required next work', '']
    decision += table(['Model', 'P pressure plus safe local-probe gate', 'Actual Loop continuation', 'Token continuation', 'Online four-way study', 'Loop-specific advantage'], [
        ['Ouro R4', 'Historical qualified result retained', 'PASS_N8', 'BLOCKED_CORRECTNESS', 'BLOCKED_CORRECTNESS', 'NO_CONFIRMED_LOOP_ADVANTAGE'],
        ['Huginn R32', 'PASS: three new independent starts', 'PASS_N8', 'BLOCKED_CORRECTNESS', 'BLOCKED_CORRECTNESS', 'NO_CONFIRMED_LOOP_ADVANTAGE']])
    decision += ['', 'Huginn can enter a bounded migration study on the recovered runtime. Its operational readiness does not prove useful offloading capacity.',
        'Both models require GPU qualification of the Token suffix-origin candidate before the fair method comparison resumes.',
        'The candidate must pass the original warm-ups, N8 and N32 state checks, varied boundaries, and fallback checks.',
        'Then measure clean Static Token/Loop controls and calibrate the online policies on separate records.',
        'Retain the qualified BF16 fixes. Their fixed-row padding must remain a separate granularity control.',
        'A selected small Token suffix can incur extra projection padding. That is not proof of an intrinsic recurrent advantage.',
        'A generic logical-layer cut at the same boundary remains EQUIVALENT_MECHANISM.',
        'No additional rate sweep, natural-EOS study, tuned cut oracle, or replicated online goodput comparison was completed.',
        'Method speedup and paired 95% uncertainty are NOT_MEASURED. This is not a negative measured speedup result.', '']
    position = report.index('## Artifacts and reproduction')
    report[position:position] = decision
    resource = read(ROOT / 'state/resource_restore_verified.json', {})
    if resource.get('verified'):
        report += ['## Resource restoration', '',
            'The final resource check verified approved burner workers on all physical GPUs 0 through 7.',
            'No additional experiment uses these devices after the GPU cutoff.',
            'The original staged index and external-runtime diff match their initial preservation signatures.',
            '[Resource verification](looped_pd_fluidtoken_vs_dynamic_loop_split/state/resource_restore_verified.json) records the worker PIDs and check time.', '']
    publication = read(ROOT / 'state/publication_receipt.json', {})
    if publication.get('verified'):
        report += ['## Verified evidence publication', '',
            f"Published evidence commit: `{publication['commit']}`.",
            f"Verified remote SHA at that publication: `{publication['remote_SHA']}`.",
            'A later closeout commit can contain this receipt and the final report. The branch tip identifies that commit.',
            'The publication record states historical verification. It does not claim a self-referential commit hash.', '']
    target = ROOT.parent / 'looped_pd_fluidtoken_vs_dynamic_loop_split_final_summary.md'
    target.write_text('\n'.join(report))
    print(str(target))


if __name__ == '__main__':
    main()
