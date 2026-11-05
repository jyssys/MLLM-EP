"""Write the final report from retained gates, not from expected outcomes."""
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo
from .common import *

FINAL = REPO / 'poc_flashvep/reports/ouro_huginn_loop_specific_native_split_deep_validation_final_summary.md'


def load(name, default):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else default


def number(value, digits=3):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) else 'NOT_MEASURED'


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(str(cell).replace('|', '/') for cell in row) + ' |' for row in rows])


def ratio_cell(entry):
    if not entry or 'geometric_mean_ratio' not in entry:
        return 'NOT_MEASURED'
    interval = entry.get('ratio_CI95')
    return (number(entry['geometric_mean_ratio'], 4) +
            (f' [{interval[0]:.4f}, {interval[1]:.4f}]' if interval else ' [CI_NOT_ESTABLISHED]') +
            f', n={entry["starts"]}')


def main():
    initialize()
    status = load('analysis_status.json', {})
    qualifications = load('state/qualification.json', {}).get('models', {})
    rows = load('serving_results.json', [])
    clean = [row for row in rows if row['data_split'] == 'heldout' and not row['markers']
             and row['correctness_passed'] and not row['calibration'] and not row.get('oracle_validation_only')]
    accepted_markers = {row['on'] for row in load('marker_interference.json', [])
                        if row['global_wall_screen_passed']}
    observed = [row for row in rows if row['name'] in accepted_markers and row['data_split'] == 'heldout'
                and row['load'] == 'medium' and row['correctness_passed'] and not row.get('oracle_validation_only')]
    pairs = load('paired_confirmation_ratios.json', {})
    graph_rows = load('graph_cache_results.json', [])
    provenance = load('state/heldout_data_provenance.json', {})
    validation = load('oracle_static_validation.json', {})
    oracle = load('oracle_headroom.json', {})
    failures = load('failure_inventory.json', [])
    selection = load('state/frozen_static_selection.json', {}).get('selected', {})
    preservation = load('state/preservation.json', {})
    resources = load('state/gpu_campaign_closed.json', {})
    cpu = load('state/final_CPU_regression.json', {})
    decision = {}
    for model in ('ouro', 'huginn'):
        partials = []
        negative = []
        loop_targets = []
        for method, metrics in pairs.get(model, {}).get('marker_free', {}).items():
            if method in ('native', 'zero'):
                continue
            sample = [row for row in clean if row['model'] == model and row['load'] == 'medium' and row['method'] == method]
            entry = metrics.get('TTFT_p95_ms', {})
            interval = entry.get('ratio_CI95')
            if len(sample) >= 3 and interval and interval[1] < 1 and all(row['Decode_SLO_screen'] and row['graph_coverage_passed'] for row in sample):
                partials.append(method)
            if method.startswith(('loop', '2d_')):
                loop_targets.append(method)
                if len(sample) >= 3 and (not all(row['Decode_SLO_screen'] for row in sample) or
                                         (interval and interval[0] >= 1)):
                    negative.append(method)
        methods = {row['method'] for row in clean if row['model'] == model and row['load'] == 'medium'}
        comparable = all(any(method.startswith(axis) for method in methods) for axis in ('native', 'token', 'loop', '2d_'))
        repeated = comparable and all(sum(row['model'] == model and row['load'] == 'medium' and row['method'] == method for row in clean) >= 3
                                      for method in methods)
        decision[model] = dict(
            practical_verdict='CONDITIONAL_GO' if partials else
                'NO_GO_ON_MEASURED_STATIC_CANDIDATES' if repeated and loop_targets and len(negative) == len(loop_targets) else 'INCONCLUSIVE',
            confirmed_TTFT_candidates=partials,
            rejected_or_confidently_slower_loop_candidates=negative,
            exclusive_loop_novelty='NO_GO_LOOP_SPECIFIC' if repeated else 'INCONCLUSIVE',
            oracle_verdict=oracle.get('status', 'NOT_MEASURED'),
            scope='A practical TTFT result does not establish goodput improvement or an exclusive recurrent mechanism. An identical generic logical-layer boundary is equivalent.')
    write(ROOT / 'final_decisions.json', decision)
    text = [
        '# Deep validation of native Token, Loop, and 2D splits',
        '## 1. Outcome and scope',
        table(['Model', 'Measured static-method verdict', 'Repeated TTFT candidates', 'Exclusive Loop-specific verdict'],
              [[model, value['practical_verdict'], ', '.join(value['confirmed_TTFT_candidates']) or 'NONE_CONFIRMED', value['exclusive_loop_novelty']]
               for model, value in decision.items()]),
        'Huginn Loop31 / Native, primary marker-free estimates:',
        table(['Metric', 'Paired ratio [95% CI]'], [[label, ratio_cell(
            pairs.get('huginn', {}).get('marker_free', {}).get('loop31', {}).get(key))]
            for label, key in [('TTFT p95', 'TTFT_p95_ms'), ('E2E p95', 'E2E_p95_ms'),
                               ('Joint SLO-goodput', 'joint_SLO_goodput')]]),
        'A confidence interval that contains one does not confirm improvement. A favorable early start cannot replace the full repeated estimate.',
        'Ouro 2D(c3,k3072) / Native TTFT p95 ratio: ' + ratio_cell(
            pairs.get('ouro', {}).get('marker_free', {}).get('2d_c3_k3072', {}).get('TTFT_p95_ms')) + '.',
        'The static-candidate verdict covers tested cuts and finite cohorts only. It does not establish a global impossibility result.',
        'Generic logical-layer execution can reproduce the same Loop boundary, state ownership, and aliased weights. This case is `EQUIVALENT_MECHANISM`.',
        'A wider 2D candidate space does not establish novelty. This campaign does not present a new online dynamic scheduler.',
        '## 2. Environment, preservation, and time',
        f'Start: {datetime.fromtimestamp(START, ZoneInfo("Asia/Seoul")).isoformat()}.',
        f'Report time: {datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}.',
        f'Elapsed wall time: {(time.time() - START) / 3600:.3f} hours. The fixed overall budget is 11 hours.',
        f'GPU stop: {datetime.fromtimestamp(GPU_DEADLINE, ZoneInfo("Asia/Seoul")).isoformat()}.',
        ('Actual last scientific dispatch result: ' +
         datetime.fromtimestamp(resources['last_dispatch_result_epoch'], ZoneInfo('Asia/Seoul')).isoformat() + '.'
         if resources.get('last_dispatch_result_epoch') else 'Actual GPU completion: NOT_CLOSED.'),
        f'Approved burner workers restored: {len(resources.get("approved_burner_workers", {}))}/8. '
        'The closeout receipt checks GPU process ownership and burner session IDs.',
        f'Source parent: `{BASE}`. Publication branch: `{BRANCH}`.',
        'The publication receipt records the evidence commit. The final remote-verification file records the final metadata commit and remote SHA.',
        'The active checkout and real staged index remain separate from the publication branch. The external runtime remains unchanged.',
        'The source archives record every loaded intervention and the qualified private-runtime diff for each run.',
        f'Preserved active HEAD: `{preservation.get("worktree_HEAD", "NOT_MEASURED")}`. Private runtime HEAD: `{preservation.get("private_HEAD", "NOT_MEASURED")}`.',
        f'Preserved staged-index SHA256: `{preservation.get("index_entries_SHA256", "NOT_MEASURED")}`.',
        f'Preserved private-diff SHA256: `{preservation.get("private_diff_SHA256", "NOT_MEASURED")}`.',
        'Hardware: eight H100 80GB devices. Each serving job uses one isolated two-GPU 1P1D pair.',
        'Ouro uses GPUs 0/1. Huginn uses GPUs 4/5. Other GPUs remain quiet during clean timing to limit shared-host interference.',
        'The campaign does not reserve exclusive CPU cores. Background host jitter is not fully eliminated.',
        'This campaign does not claim an eight-GPU serving speedup. Approved burners resume only when the experiment does not own the quiet lane.',
        'Preservation details: [preservation.json](ouro_huginn_loop_specific_native_split_deep_validation/state/preservation.json).',
        '## 3. Models, data, loads, and SLOs',
        table(['Model', 'Precision / depth', 'Input / output tokens', 'Low / medium / heavy nominal QPS', 'P/D slots'], [
            ['ByteDance/Ouro-1.4B', 'Original BF16 / R4', '4096 / 32', '1.23460875 / 3.4569045 / 6.356', '2 / 8'],
            ['tomg-group-umd/huginn-0125', 'Original BF16 / R32', '2048 / 16', '0.37903355455 / 1.06129395274 / 1.971', '2 / 8']]),
        'All methods retain original weights, qualified numerical fixes, exact Attention, and full recurrent work. Huginn retains Prelude, Coda, and reinjection.',
        'The prompt pool uses pinned ShareGPT and LongBench text. Inputs are length-controlled tokenizer prefixes. Outputs use fixed caps with ignore-EOS.',
        'These lengths are not natural response distributions or Azure traffic. Huginn remains inside its 4096-token context limit.',
        'Discovery uses the previous pool. Confirmation uses 128 new prompt IDs. Separate calibration uses 32 additional IDs.',
        'Model revisions: Ouro `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. Huginn `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.',
        'ID, source-hash, and model-specific first-128-token-prefix checks exclude overlap between previous, held-out, and calibration pools.',
        'The data manifest retains actual token IDs, source revisions, and dataset counts. Prefix caching remains disabled.',
        'All paired methods share prompt IDs, generation seeds, fixed output caps, and open-loop Poisson arrival offsets.',
        'SLOs remain unchanged from the previous campaign:',
        table(['Model', 'TTFT, ms', 'Request-average TPOT, ms', 'Token ITL p99, ms'], [
            ['Ouro', '1112.9925646819174', '53.70702898374411', '65.08287642151119'],
            ['Huginn', '1963.9920117333531', '186.5715059141318', '192.8212538175284']]),
        'The additional-work screen requires TPOT p95 at most 90% of its limit and ITL p99 at most its frozen limit.',
        'Joint goodput counts requests that meet both request TTFT and request-average TPOT. Failed and timed-out offered requests remain in the denominator.',
        'Finite overloaded cohorts are not sustainable serving capacity.',
        '## 4. Scheduler recovery and exactness',
        'An unused D-local-probe pacing hook changed the effective D scheduler. The new configs disable this unused hook for every method.',
        'The runtime checks requested and effective non-Wavefront settings after stage construction. The native one-depth Wavefront guard remains unchanged.',
        'The isolated Huginn Native and Token1984 N8 tests separate this configuration correction from the later graph intervention.',
        f'Final CPU regressions: {cpu.get("tests", "NOT_MEASURED")} tests. '
        f'Failures: {cpu.get("failures", "NOT_MEASURED")}. Errors: {cpu.get("errors", "NOT_MEASURED")}. '
        'CPU tests do not replace GPU state qualification.',
    ]
    qualification_rows = []
    for model, item in qualifications.items():
        for method, gate in item['qualification'].items():
            checks = gate.get('checks', [])
            qualification_rows.append([model, method, 'PASS' if gate['passed'] else 'FAIL_OR_INCOMPLETE',
                                       ', '.join(str(row.get('name', row.get('status'))) for row in checks)])
    text += [table(['Model', 'Method', 'N8 and N32 gate', 'Retained checks'], qualification_rows),
        'N32 means simultaneous submission of 32 requests. It does not mean 32 active GPU requests.',
        'Exact gates compare output tokens, valid prompt KV hashes, the recorded prompt hidden state, transferred valid bytes, full recurrent rows, and source lifetime.',
        'The hidden-state gate covers the recorded boundary. It does not compare every intermediate tensor after every operation.',
        'Source retirement follows native ACK. Final coordinator credits and native allocations must retire. Forced full-PD branches are not a complete online admission policy.',
        'Original-HF fidelity, downstream accuracy, and every-token hidden-state equivalence remain NOT_MEASURED.',
        '## 5. Graph-cache diagnosis and control',
        'The native packed graph cache uses exact query/context geometry. Startup order can fill four entries before primary single-request shapes appear.',
        'The cache does not evict old entries. A full cache sends unseen shapes to eager execution.',
        'The common intervention reserves primary B1/B2 geometries and performs one serial full-prompt warm-up before ordinary warm-up.',
        'The cache4 qualification exposed a three-request D geometry. The main comparison therefore uses the common eight-entry cap and exact D-geometry warm-up.',
        'All main methods use the same policy. Warm-up requests do not enter measured latency or throughput.',
        'Ouro 2D requires four P geometries: 4096, 8192, 3072, and 6144 query rows. The query length remains part of each key.',
        'The plan includes Legacy4, reserved4, and reserved8 sensitivity controls. Only explicit completed receipts establish their execution.',
        'These control packages change admission and warm-up together. A missing control remains NOT_RUN_TIME_BUDGET, not a measured equivalence result.',
        'A primary-shape hit is not a GPU kernel count. Capture time and host preparation time are not exclusive kernel time.',
    ]
    allocation_rows = []
    wire_inventory = load('actual_wire_payload_accounting.json', [])
    for model, entry in load('oracle_cost_model_frozen_v2.json', {}).get('models', {}).items():
        name = f'd_{model}_native_heldout_s1_off128'
        files = (ROOT / 'raw' / name).glob('memory_inventory_initial_*.json')
        memory = next((value for path in files if (value := json.loads(path.read_text()))['role'] == 'decode'), None)
        native_wire = next((row for row in wire_inventory if row['name'] == name and row['role'] == 'prefill'), None)
        if memory is None or native_wire is None:
            continue
        pool = memory['KV_storage_bytes']
        quota = pool / entry['D_KV_blocks'] * entry['request_KV_blocks']
        physical = 81559 * 1024**2
        allocation_rows.append([model, number(pool / 2**30), number(100 * pool / physical),
            entry['request_KV_blocks'], number(quota / 2**30), number(100 * quota / physical),
            number(native_wire['KV_bytes'] / native_wire['count'] / 2**30),
            entry['D_credit_limit'], entry['D_slots']])
    text.insert(text.index('## 4. Scheduler recovery and exactness'),
        'Valid wire KV and reserved pool quota are different quantities. '
        'The quota uses full-depth block reservation times measured storage bytes per block. '
        'It is accounting inside an existing pool, not a new CUDA allocation per request. '
        'The derived credit limit includes receiving requests. It is not measured peak Decode concurrency. '
        'HBM percentages use the recorded NVML total of 81559 MiB per GPU.\n\n' +
        table(['Model', 'D KV pool GiB', 'Pool / HBM %', 'Reserved blocks / req', 'Quota GiB / req',
               'Quota / HBM %', 'Valid Native KV wire GiB / req', 'Derived KV credit limit', 'D compute slots'], allocation_rows) +
        '\n\nThe current last_exited layout reserves complete layer blocks across recurrent depths. '
        'Valid nonrecurrent KV does not require every reserved depth plane. '
        'This allocation is not the model\'s theoretical minimum exact KV storage. '
        'A compact heterogeneous-layer pool was not implemented or qualified in this campaign.')
    g = [row for row in graph_rows if row['name'].startswith(('d_', 'h_')) and row['role'] == 'prefill']
    text += [table(['Run', 'P hit ratio', 'Cold entries', 'Eager fallbacks', 'Cache entries'],
                  [[row['name'], number(row.get('measured_existing_hit_ratio'), 4), row['cold_entries'], row['fallbacks'], len(row['entries'])] for row in g]),
        '## 6. Held-out clean serving and load dependence',
        'The following cells average per-run percentiles across independent starts. They are not pooled request percentiles.',
        'Full p50/p95/p99 metrics and offered/realized/completed rates remain in serving_results.csv and raw per-request files.',
    ]
    performance = []
    keys = sorted({(row['model'], row['load'], row['method']) for row in clean})
    for model, load_level, method in keys:
        sample = [row for row in clean if (row['model'], row['load'], row['method']) == (model, load_level, method)]
        mean = lambda key: statistics.mean(row[key] for row in sample)
        performance.append([model, load_level, method, len(sample), number(mean('TTFT_p95_ms')),
            number(mean('TPOT_p95_ms')), number(mean('ITL_p99_ms')), number(mean('E2E_p95_ms')),
            number(mean('joint_SLO_goodput'), 4), f'{sum(row["Decode_SLO_screen"] for row in sample)}/{len(sample)}'])
    text += [table(['Model', 'Load', 'Method', 'Starts', 'TTFT p95 ms', 'TPOT p95 ms', 'ITL p99 ms', 'E2E p95 ms', 'Joint goodput req/s', 'Decode-screen passes'], performance),
        '## 7. Huginn Loop31 and paired uncertainty',
        'Marker-free confirmation is the primary latency comparison. The unit of replication is an independent start, not an individual request or token.',
        'Intervals use a paired t interval on log ratios. Three to five starts require the stated small-sample distribution assumption.',
        'These intervals condition on the fixed held-out prompt pool. They describe start and arrival variability, not every possible text distribution.',
        'A TTFT confidence interval below one supports a TTFT reduction. It does not establish an E2E or goodput improvement.',
        'If all tested Token partials fail the Decode screen, Native full-PD remains the safe Token-only endpoint. An unsafe diagnostic partial is not a strong safe baseline.',
    ]
    ratio_rows = []
    for model in ('ouro', 'huginn'):
        for method, fields in pairs.get(model, {}).get('marker_free', {}).items():
            if method == 'native':
                continue
            ratio_rows.append([model, method, ratio_cell(fields.get('TTFT_p95_ms')),
                               ratio_cell(fields.get('E2E_p95_ms')), ratio_cell(fields.get('joint_SLO_goodput'))])
    text += [table(['Model', 'Method / Native', 'TTFT ratio [95% CI]', 'E2E ratio [95% CI]', 'Goodput ratio [95% CI]'], ratio_rows),
        'Marker ON/OFF runs remain separate. Service TTFT is NOT_MEASURED in marker-free runs.',
        'The safe Token reference also considers Native and zero-offload full-prefix endpoints. Discovery fixes this choice before held-out results exist.',
        'The absolute global-wall interference screen remains 5%. Separate columns report local TTFT and ITL tail changes.',
        'A global pass does not prove negligible tail perturbation. Rejected traces do not become clean performance evidence.',
        '## 8. TTFT decomposition and actual migration',
        'Native versus feature-enabled zero-offload is a same-boundary transport/runtime control. Loop versus zero is a separate redistribution comparison.',
        'A Native-to-Loop difference must not automatically become a Loop-compute gain. The control ratios remain in paired_zero_path_reference_ratios.json.',
        'Request TTFT starts at scheduled arrival. Queue/admission ends at the first actual model-stream Prefill entry.',
        'This interval includes scheduling, input preparation, and device queueing. It is not exclusively P-ready queue waiting.',
        'Service TTFT starts at that entry and includes native handoff, D continuation, Coda, and first output delivery.',
        'The per-request timestamp file reports clock-mapping brackets. Nonlinear clock drift is not fully bounded.',
        'Per-request intervals precede percentiles. Component p95 values are not added. Overlapping host/device/transfer durations are not added as speedup.',
        'Work counters report actual serving continuations. They do not count the previous independent D-local probes.',
        'Recurrent token rows are not FLOPs. Token suffix Attention reads the completed prefix, so equal rows do not imply equal Attention work.',
        'Transfer payload counters exclude control messages and transport headers. Commit, ACK, and source-retention intervals remain separate.',
        'HBM snapshots report GiB and percentages of the recorded physical HBM total. Allocator totals overlap component tensors.',
        'The matched reservation policy retains full-depth request KV allocation. Fewer valid transferred rows do not imply released physical HBM.',
        'Exposed graph tensors do not account for every opaque graph-pool or library allocation. Warm-up memory is not claimed as clean peak HBM.',
        'Current kernel union, SM cycles, achieved bandwidth, and achieved GEMM throughput are NOT_MEASURED unless a separately qualified profile is present.',
        'CUDA event enclosures can include launch gaps and waits. They are not exclusive GPU kernel time.',
        'A B2 operation can assign the same device enclosure to both requests. A larger per-request enclosure does not establish doubled FLOPs.',
        '## 9. Structural attribution and logical-layer equivalence',
        'The qualified numerical fix uses fixed 256-row Down-projection GEMMs. A full prompt does not imply one large Down-projection GEMM.',
        'For B1 Huginn, Loop31 moves 2048 semantic rows through one loop. Token1984 moves 64 rows through 32 loops.',
        'The unchanged Down projection uses eight fixed tiles per core layer in the first case and 32 in the second.',
        'This fourfold tile difference covers that projection only. Batching can change it. It is not a measured fourfold total-runtime difference.',
        'For Ouro, Loop3 and Token3072 have equal semantic D rows and equal fixed Down-tile counts at B1. Their Attention and submission costs differ.',
        'Huginn Loop31 also changes full-prompt Coda placement. Its 3.125% recurrent-row fraction excludes Coda and is not a FLOP fraction.',
        'Small service changes near capacity can amplify queue delay. The observed queue and Service TTFT results must support that explanation.',
        'A generic logical-layer split can reproduce these boundaries with shared recurrent-weight aliases. Both workers already hold the full model.',
        'This equivalence is structural. This campaign did not measure a separate generic-layer serving implementation.',
        'Therefore a favorable native-runtime trade-off does not establish exclusive Loop-specific architecture novelty or a weight-residency saving.',
        '## 10. Validated offline oracle',
        f'Validation status: `{validation.get("status", "NOT_MEASURED")}`. Headroom status: `{oracle.get("status", "NOT_MEASURED")}`.',
        'Calibration prompts remain disjoint from discovery and confirmation. Version 1 froze before confirmation dispatches.',
        'Version 1 incorrectly included queued GPU work in its preparation enclosure. Its frozen model and predictions remain unchanged as failed evidence.',
        'Version 2 uses the host ModelRunner entry after input preparation. This host enclosure can still include synchronization, not just CPU work.',
        'Version 2 fitted calibration data only. It froze after four Huginn confirmation dispatches began, so those dispatches cannot validate this version.',
        'Every included Version 2 prediction precedes its own GPU dispatch. The protocol receipt records this deviation and all excluded dispatches.',
        'Validation checks frozen hashes and prediction timestamps against the estimated subprocess start.',
        'Neither version uses confirmation outcomes to fit costs. Version 2 does not inherit Version 1 validation coverage.',
        f'Validated prediction directory: `{validation.get("predictions_directory", "NOT_MEASURED")}`. Cost model: `{oracle.get("cost_model_file", "NOT_MEASURED")}`.',
        'The event model includes P/D serialization, homogeneous batching, native transfer, ACK/source leases, D KV receiving credit, and fixed-depth Decode.',
        'Missing batch costs require explicit interpolation or extrapolation. Event enclosures and residuals are not exclusive GPU/CPU execution times.',
        'The validation screen requires median absolute relative error at most 15% and p95 error at most 25%. It also rejects false Decode-safe predictions.',
        'The report retains start-cluster error intervals. An inaccurate model cannot establish Feasible Oracle headroom.',
        'The restricted offline search knows future arrival times and output caps. It is not an online method or a global optimality certificate.',
        'Mixed-frontier GPU scheduling remains unqualified. Static-model accuracy alone does not prove accuracy for every dynamic plan.',
        '## 11. Prior work and reusable implementation',
        'The Token baseline is a FluidToken-style reproduction, not the official FluidPD system.',
        'This campaign measures the static prefix/suffix primitive. It does not implement the full PPI/DPI controller or FluidRole.',
        'FluidPD describes pressure-aware Token deflection with PPI/DPI and Decode slack. Our experiment retains native worker roles and exact prefix/suffix handoff.',
        'FluidPD permits stream-average TPOT slack borrowing. This experiment also retains its stricter frozen ITL p99 gate. The SLO contracts are not identical.',
        'The current native runner groups the selected D continuation into one operation. A Token suffix can therefore contain more recurrent iterations per scheduling quantum.',
        'Alternative suffix pacing or a common smaller D recurrence quantum is NOT_MEASURED. A result against these atomic candidates does not dominate every Token scheduler.',
        'Source: [FluidPD](https://arxiv.org/html/2610.06917v1). The inspected paper did not identify reusable official implementation code.',
        'The LoopServe reference uses a different pooled cache and execution ABI. The campaign reuses compatible scheduling lessons, not an unverified runtime transplant.',
        '## 12. Figures and complete evidence',
        'The figure manifest records measured, structural, modeled, and missing evidence separately.',
        '[Figure manifest](ouro_huginn_loop_specific_native_split_deep_validation/figure_manifest.json).',
        '[Serving results](ouro_huginn_loop_specific_native_split_deep_validation/serving_results.csv).',
        '[Paired intervals](ouro_huginn_loop_specific_native_split_deep_validation/paired_confirmation_ratios.json).',
        '[Graph results](ouro_huginn_loop_specific_native_split_deep_validation/graph_cache_results.csv).',
        '[Work and payload](ouro_huginn_loop_specific_native_split_deep_validation/work_and_payload.csv).',
        '[Fixed-tile accounting](ouro_huginn_loop_specific_native_split_deep_validation/fixed_tile_accounting.csv).',
        '[HBM inventory](ouro_huginn_loop_specific_native_split_deep_validation/gpu_memory_gib_percent.csv).',
        '[Oracle validation](ouro_huginn_loop_specific_native_split_deep_validation/oracle_static_validation.json).',
        '## 13. Failures, missing conditions, and research decision',
        f'Completed dispatches: {status.get("completed_dispatches", "NOT_MEASURED")}. Failed dispatches: {status.get("failed_dispatches", "NOT_MEASURED")}.',
        f'Completed-dispatch offered requests: {status.get("completed_offered_requests", "NOT_MEASURED")}. This count includes exactness and calibration, not just clean serving.',
        'The configuration inventory retains every prepared and attempted condition. Failure logs remain intact.',
        'N512 persistence is NOT_RUN unless its explicit state receipt exists. N128 repetition does not become 512 independent texts.',
        'A missing or inaccurate feasible oracle remains INCONCLUSIVE. It cannot establish structural absence of headroom.',
        'Do not begin a full dynamic-scheduler campaign solely because a selected static cut improves TTFT. Require Decode-safe repeated gains and a justified contribution.',
        '## 14. Reproduction and publication',
        'Use the qualified virtual environment and the preserved private runtime. Check the source and configuration hashes before each replay.',
        'Do not rerun immutable configuration names into existing raw directories. Use a fresh evidence namespace for a replay.',
        '```bash\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.native_split_deep_validation.audit\n/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest discover -s poc_flashvep/native_split_deep_validation -p "test_*.py" -t .\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.native_split_deep_validation.analyze\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.native_split_deep_validation.prospective_oracle_v2 validate\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.native_split_deep_validation.figures\n/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.native_split_deep_validation.report\n```',
        'The publication uses a temporary Git index and ordinary push. The publisher checks the real staged index and private-runtime diff before and after publication.',
        'The state/ directory retains final resource and remote-SHA receipts.',
    ]
    location = text.index('## 13. Failures, missing conditions, and research decision')
    for entry in load('figure_manifest.json', {}).get('figures', []):
        png, pdf = entry['files']
        text.insert(location, f'### {entry["figure"]}\n\nEvidence status: `{entry["status"]}`.\n\n'
            f'![{entry["figure"]}]({ROOT.name}/{png})\n\n[Vector PDF]({ROOT.name}/{pdf})')
        location += 1
    error_rows = []
    for model, values in validation.get('by_model', {}).items():
        error_rows.append([model] + [number(100 * values[key]) if values.get(key) is not None else 'NOT_MEASURED'
            for key in ('median_absolute_relative_error', 'p95_absolute_relative_error',
                        'service_median_absolute_relative_error', 'service_p95_absolute_relative_error')] +
            [values['error_screen_passed']])
    text.insert(text.index('## 11. Prior work and reusable implementation'),
        'The frozen prediction-error limits are 15% for the median and 25% for p95. '
        'The following values are absolute relative errors, not serving speedups.\n\n' +
        table(['Model', 'Clean median error %', 'Clean p95 error %', 'Service median error %',
               'Service p95 error %', 'Model error screen'], error_rows) +
        f'\n\nFalse predicted Decode-safe cases: {len(validation.get("false_predicted_Decode_safe", []))}. '
        'A zero false-safe count does not repair timing errors or missing family-level validation coverage. '
        'Start-cluster error intervals and per-family coverage remain in oracle_static_validation.json.')
    coverage_rows = []
    for model in ('ouro', 'huginn'):
        methods = list(dict.fromkeys(['native', 'zero'] +
            [entry['method'] for entry in selection.get(model, {}).values()]))
        for method in methods:
            counts = [sum(row['model'] == model and row['method'] == method and row['load'] == level
                          for row in clean) for level in ('low', 'medium', 'heavy')]
            coverage_rows.append([model, method] + [count if count else 'NOT_RUN_TIME_BUDGET' for count in counts])
    text.insert(text.index('## 14. Reproduction and publication'),
        'The following coverage counts include marker-free N128 starts only. '
        'Missing load comparisons do not inherit calibration or previous-campaign results.\n\n' +
        table(['Model', 'Method', 'Low starts', 'Medium starts', 'Heavy starts'], coverage_rows) +
        '\n\nOptional Huginn starts four and five, N512 persistence, and legacy-cache latency controls remain unrun unless their own receipts exist. '
        'Prepared oracle-only supplementary configurations are not additional primary serving starts.')
    token_pairs = load('paired_token_reference_ratios.json', {})
    token_rows = []
    for model, comparisons in token_pairs.items():
        for method, fields in comparisons['comparisons'].items():
            token_rows.append([model, method + ' / ' + comparisons['reference'],
                ratio_cell(fields.get('TTFT_p95_ms')), ratio_cell(fields.get('E2E_p95_ms')),
                ratio_cell(fields.get('joint_SLO_goodput'))])
    text.insert(text.index('## 8. TTFT decomposition and actual migration'),
        table(['Model', 'Method / selected Token', 'TTFT ratio [95% CI]', 'E2E ratio [95% CI]', 'Goodput ratio [95% CI]'], token_rows))
    partial_rows = []
    for model, entry in load('paired_selected_token_partial_ratios.json', {}).items():
        for method, fields in entry['comparisons'].items():
            partial_rows.append([model, method + ' / ' + entry['reference'],
                f'{entry["actual_Decode_screen_passes"]}/{entry["actual_reference_starts"]}',
                ratio_cell(fields.get('TTFT_p95_ms')), ratio_cell(fields.get('E2E_p95_ms')),
                ratio_cell(fields.get('joint_SLO_goodput'))])
    text.insert(text.index('## 8. TTFT decomposition and actual migration'),
        'The next table compares every method with the prospectively selected Token partial. The actual held-out Decode screen, not its old discovery label, determines eligibility.\n\n' +
        table(['Model', 'Method / selected Token partial', 'Token Decode-screen passes', 'TTFT ratio [95% CI]',
               'E2E ratio [95% CI]', 'Goodput ratio [95% CI]'], partial_rows))
    marker_rows = []
    accepted_pairs = load('paired_marker_accepted_ratios.json', {})
    shifts = load('marker_relative_advantage_shifts.json', {}).get('results', {})
    interference = load('marker_interference.json', [])
    for model, methods in accepted_pairs.items():
        for method, fields in methods.items():
            if method == 'native':
                continue
            sample = [row for row in interference if row['model'] == model and row['method'] == method]
            marker_rows.append([model, method,
                f'{sum(row["global_wall_screen_passed"] for row in sample)}/{len(sample)}',
                ratio_cell(fields.get('TTFT_p95_ms')),
                ratio_cell(shifts.get(model, {}).get(method, {}).get('TTFT_p95_ms'))])
    text.insert(text.index('## 8. TTFT decomposition and actual migration'),
        'Accepted ON comparisons remain separate from marker-free comparisons. '
        'A ratio of ratios below one means markers show a larger apparent relative gain. '
        'This finite control also contains scheduling noise. It does not identify a pure marker effect.\n\n' +
        table(['Model', 'Candidate', 'Global marker passes / pairs', 'ON candidate / Native TTFT [95% CI]',
               '(ON ratio) / (OFF ratio) [95% CI]'], marker_rows))
    decomposition = []
    for model, method in sorted({(row['model'], row['method']) for row in observed}):
        sample = [row for row in observed if (row['model'], row['method']) == (model, method)]
        def mean_metric(key):
            values = [row[key] for row in sample if isinstance(row.get(key), (int, float))]
            return number(statistics.mean(values)) if values else 'NOT_MEASURED'
        decomposition.append([model, method, len(sample)] + [mean_metric(key) for key in
            ('TTFT_p95_ms', 'queue_ms_p95', 'service_TTFT_ms_p95', 'P_prompt_device_enclosure_ms_p95',
             'observed_handoff_ms_p95', 'D_prompt_device_enclosure_ms_p95')])
    location = text.index('## 9. Structural attribution and logical-layer equivalence')
    text.insert(location, 'Accepted marker runs provide the following decomposition. Cells average per-run p95 values. Columns are not additive critical-path components.\n\n' +
        table(['Model', 'Method', 'Accepted starts', 'TTFT p95 ms', 'Queue p95 ms', 'Service TTFT p95 ms',
               'P device enclosure p95 ms', 'Observed handoff p95 ms', 'D prompt enclosure p95 ms'], decomposition))
    primitive_rows = []
    calibration = load('oracle_cost_model_frozen_v2.json', {}).get('models', {})
    for model, entry in calibration.items():
        for method, action in entry['actions'].items():
            p_steps = ', '.join(phase + '=' + number(batches.get('1', {}).get('median'))
                                for phase, batches in action['P_steps'].items())
            payload = action.get('payload_bytes') or {}
            primitive_rows.append([model, method, p_steps,
                number(action['D_prompt'].get('1', {}).get('median')),
                number((action.get('serial_handoff_ms') or action.get('handoff_ms') or {}).get('median')),
                number(payload['median'] / 2**30, 5) if 'median' in payload else 'NOT_MEASURED'])
    text.insert(text.index('## 9. Structural attribution and logical-layer equivalence'),
        'Independent instrumented calibration provides these operation costs. B1 means one request in the recorded prompt operation. '
        'Each cell is a separate median, not a paired critical-path saving. '
        'Device enclosures can include launch gaps or waits. Observed handoff is not pure wire time. '
        'The wire column reports actual accepted KV and hidden bytes per request.\n\n' +
        table(['Model', 'Method', 'P B1 operation medians, ms', 'D B1 prompt median, ms',
               'Serial observed handoff median, ms', 'KV + hidden GiB / request'], primitive_rows))
    work = {row['name']: row for row in load('work_and_payload.json', [])}
    wire = {row['name']: row for row in load('actual_wire_payload_accounting.json', []) if row['role'] == 'prefill'}
    migration = []
    for model, method in sorted({(row['model'], row['method']) for row in clean if row['load'] == 'medium'}):
        sample = [row for row in clean if (row['model'], row['method'], row['load']) == (model, method, 'medium')
                  and row['name'] in work and row['name'] in wire]
        if not sample:
            continue
        offered = sum(row['offered'] for row in sample)
        moved = sum(work[row['name']]['completed_offloaded_serving_requests'] for row in sample)
        recurrent = sum(work[row['name']]['D_prompt_recurrent_rows'] for row in sample) / offered
        payload = lambda key: sum(wire[row['name']][key] for row in sample) / offered
        migration.append([model, method, f'{moved}/{offered}', number(recurrent, 1),
            number(payload('KV_bytes') / 1024**3), number(payload('hidden_bytes') / 1024**2),
            number(payload('accepted_submissions'), 1), number(payload('descriptors'), 1)])
    text.insert(text.index('## 9. Structural attribution and logical-layer equivalence'),
        'These counts cover actual held-out serving executions. Repeated starts do not create additional independent prompt texts.\n\n' +
        table(['Model', 'Method', 'Offloaded / offered executions', 'D recurrent rows / req', 'KV GiB / req',
               'Hidden MiB / req', 'Submissions / req', 'Descriptors / req'], migration))
    receipt_rows = []
    receipt_fields = ('P_complete_receipt_to_commit_receipt_ms_p95',
        'commit_receipt_to_received_receipt_ms_p95', 'received_receipt_to_P_ACK_ms_p95',
        'P_ACK_to_P_forget_receipt_ms_p95', 'P_complete_receipt_to_P_credit_release_ms_p95')
    lifecycle = load('native_handoff_receipt_summary.json', [])
    for model, method in sorted({(row['model'], row['method']) for row in clean if row['load'] == 'medium'}):
        sample = [row for row in lifecycle if row['model'] == model and row['method'] == method
                  and not row['markers'] and row['load'] == 'medium']
        cells = []
        for field in receipt_fields:
            values = [row[field] for row in sample if row.get(field) is not None]
            cells.append(number(statistics.mean(values)) if values else 'NOT_MEASURED')
        receipt_rows.append([model, method, len(sample)] + cells)
    text.insert(text.index('## 9. Structural attribution and logical-layer equivalence'),
        'The next intervals use existing native host-message receipts in marker-free runs. '
        'Cells average per-run p95 values. They are not exclusive DMA or CPU times. '
        'Source credit retirement is not a complete physical-buffer lifetime measurement. '
        'Negative observer-order intervals remain in the per-request file. These percentiles are not additive.\n\n' +
        table(['Model', 'Method', 'Starts', 'P-complete to commit ms', 'Commit to received ms',
               'Received to P ACK ms', 'P ACK to P forget ms', 'P-complete to P credit release ms'], receipt_rows))
    gap_rows = []
    gap_fields = ('received_receipt_to_D_prompt_entry_ms_p95',
                  'P_model_stream_end_to_D_prompt_entry_ms_p95', 'D_prompt_end_to_first_token_ms_p95')
    for model, method in sorted({(row['model'], row['method']) for row in observed
                                if row['method'].startswith(('loop', 'token', '2d_'))}):
        names = {row['name'] for row in observed if row['model'] == model and row['method'] == method}
        sample = [row for row in lifecycle if row['name'] in names]
        cells = []
        for field in gap_fields:
            values = [row[field] for row in sample if row.get(field) is not None]
            cells.append(number(statistics.mean(values)) if values else 'NOT_MEASURED')
        gap_rows.append([model, method, len(sample)] + cells)
    text.insert(text.index('## 9. Structural attribution and logical-layer equivalence'),
        'Accepted marker runs also expose post-import continuation gaps. '
        'The gaps use per-request host receipts and model-stream clock projections before percentile calculation. '
        'They include scheduling, input preparation, and device queueing. '
        'A large gap does not prove CPU inefficiency or an idle GPU. Decode-priority work can occupy the device during that gap.\n\n' +
        table(['Model', 'Method', 'Accepted starts', 'Received to D prompt entry p95 ms',
               'P stream end to D prompt entry p95 ms', 'D prompt end to first output p95 ms'], gap_rows))
    oom = load('state/huginn_2d_startup_oom_diagnosis.json', {})
    if oom:
        text.insert(text.index('## 6. Held-out clean serving and load dependence'),
            'Huginn 2D(c28,k1536) failed during batch-four D geometry warm-up. The allocation requested 84.00 MiB with 78.50 MiB free. '
            'This failure occurred before measurement. It is BLOCKED_RESOURCE, not a slow serving result or an output divergence. '
            'The unchanged OOM action does not enter later calibration or oracle expansion. '
            'Graph retention, transient workspace, and allocator fragmentation are not fully isolated. '
            'A smaller common KV pool is an untested control, not an applied repair.')
    observer = load('state/warmup_hook_control.json', {})
    if observer:
        controls = []
        for entry in observer.get('controls', []):
            result = entry['result']
            values = entry.get('ratios', {})
            controls.append([result['model'], result['method'],
                number(values.get('response_span_s'), 4), number(values.get('TTFT_p95_ms'), 4),
                number(values.get('TPOT_p95_ms'), 4), number(values.get('ITL_p99_ms'), 4)])
        text.insert(text.index('## 6. Held-out clean serving and load dependence'),
            'The D geometry hook originally reread its terminal JSON marker during main scheduling. '
            'The exact closure replay produced ten reads for ten terminal-state callbacks. '
            'Terminal-state caching reduced this to one read while retaining the active-import barrier and native readiness predicate. '
            'The warmup_hook_control.json file retains the fixed Huginn Token1984 N8/N32 state receipts.\n\n'
            'Historical discovery keeps the original observer. New calibration and held-out configurations explicitly retire the terminal hook. '
            'These observer versions do not enter one clean paired estimate. '
            'Cut selection used the historical observer. Cuts were not retuned after this correction. '
            'The selected Token partial is not a proven best cut under the corrected observer. '
            'The following single before/after controls share inputs and arrival traces. '
            'They are not randomized repeated causal estimates. The CPU replay proves repeated I/O, not that I/O explains every latency change.\n\n' +
            table(['Model', 'Method', 'New / old response span', 'TTFT p95 ratio', 'TPOT p95 ratio', 'ITL p99 ratio'], controls) +
            '\n\nArrival-dominated response span can change little while request tails change more. '
            'A global 5% pass does not prove a negligible TPOT or ITL perturbation.')
    FINAL.write_text('\n\n'.join(text) + '\n')
    print(FINAL)


if __name__ == '__main__':
    main()
