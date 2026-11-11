"""Write the campaign result with explicit evidence and entry-gate limits."""
import collections,json,statistics,time
from datetime import datetime,timezone
from .common import *


def main():
    initialize();v=json.loads((ROOT/'analysis/final_campaign_inventory.json').read_text())
    rows={r['configuration']['name']:r for r in v['serving']}
    physical='profile_huginn_L3072_O16_P2_mu115_N64_a2026111230_pinned_capture'
    p=json.loads((ROOT/'raw'/physical/'physical_interval_analysis.json').read_text());d=p['decode']
    h=json.loads((ROOT/'analysis'/('final_host_'+physical+'.json')).read_text())
    rel=ROOT.name
    def link(label,target):return f'[{label}]({rel}/{target})'
    def fmt(x,n=3):return 'NOT_MEASURED' if x is None else f'{x:.{n}f}'
    status=collections.Counter(r['status'] for r in v['runs'])
    closed=json.loads((ROOT/'state/CAMPAIGN_CLOSED.json').read_text());duration=(closed['time']-START)/3600
    text=f'''# Ouro/Huginn Prefill-heavy discovery and conservative Loop offloading

## Result

The requested TO-BE is not demonstrated in this campaign.
Neither model passed every Phase 1 entry gate.
Phase 2 guarded Serving migration is NOT_RUN.
This is not a measured zero-speedup result or a proof of structural impossibility.

| Model | Phase 1 | Phase 2 | Main reason |
| --- | --- | --- | --- |
| Ouro R4 | INCONCLUSIVE_MEASUREMENT | NOT_RUN | No qualified repeated physical activity observation |
| Huginn R32 | INCONCLUSIVE_MEASUREMENT | NOT_RUN | One qualified short pilot, but no repeated qualification or useful verified migration capacity |

Huginn has one qualified physical-kernel pilot at 3072/16.
Its common interval lasts {d['window_seconds']:.9f} seconds.
P kernel busy is {p['prefill']['kernel_busy_percent']:.6f}%.
D kernel inactivity is {d['physical_kernel_idle_percent']:.6f}%.
The longest D gap is only {d['idle_ms']['maximum']:.6f} ms.
Total inactivity is therefore not one usable execution window.

The fixed-trace enclosure screen completes no full Loop, even with free state and zero handoff costs.
That result applies only to this trace and its measured enclosure costs.
It does not establish a universal physical capacity bound.

## Scope, clock, and preservation

GPU discovery started at 2026-11-11 12:27:21 UTC.
GPU discovery ended at {datetime.fromtimestamp(closed['time'],timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.
The discovery and development interval lasted {duration:.3f} hours.
The ten-hour total deadline remains 2026-11-11 22:27:21 UTC.
Final analysis, publication, and resource checks follow GPU closure.

This campaign contains {len(v['runs'])} immutable planned configurations.
Status counts are `{dict(status)}`.
These are configuration cells, not independent successful experiments.
The inventory includes {len(v['serving'])} completed Serving analysis cells and {len(v['state_qualifications'])} passed state qualifications.
The eight priority width-comparison state checks remain in their earlier published report.
They are not counted again as Phase 2 success.

Original BF16 weights, exact Attention, and fixed R4/R32 remain unchanged.
No early exit, quantization, or approximate computation is used.
Prefix caching remains disabled.
Huginn input plus output never exceeds the verified 4096-token context.
All primary GPU timing uses an isolated pair under one quiet-lane lock.
No unvalidated pair-parallel timing contributes to these results.

The active checkout and original staged index remain separate from the publication branch.
The external qualified runtime and its original patch remain preserved.
Publication uses a private Git index.
The final preservation and resource receipts provide the exact signatures.

## Workload search and Native controls

The search started from the previous 4096/32 Ouro and 2048/16 Huginn anchors.
Old D-local probes did not count as P-to-D migration.
Longer natural document prefixes increased real Prefill work.
The search did not insert artificial sleep or change model semantics.

Calibration uses separate natural documents when long inputs require a new pool.
Short discovery pilots can repeat calibration texts with caching disabled.
They do not represent 64 independent texts.
The Huginn N128 confirmation uses 128 distinct controlled prompt IDs and three arrival seeds.

The following marker-free cells summarize important search decisions.
Queue growth is the audited second-half minus first-half mean.
Refusal is admission-snapshot prevalence, not a measured causal share.
P service is a finite-window rate divided by a matched independent reference.

| Cell | Input/output | QPS | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | P queue growth / max | P service ratio | D refusal fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
'''
    selected=[
        ('Ouro recovered memory','memory_ouro_L16384_O16_P2_mu115_N64_a2026111230'),
        ('Ouro longer legal input','extended_ouro_L18432_O16_P2_mu130_pilot64_a2026111272'),
        ('Ouro TPOT failure','extended_ouro_L24576_O16_P2_mu130_pilot64_a2026111272'),
        ('Huginn pressure candidate','reserve_huginn_L3072_O16_P2_mu115_N64_a2026111230'),
        ('Huginn longer input','reserve_huginn_L3900_O16_P2_mu115_N64_a2026111230'),
        ('Huginn context edge','edge_huginn_L4080_O16_P2_mu120_pilot64_a2026111310')]
    for label,name in selected:
        r=rows[name];c,m,s=r['configuration'],r['metrics'],r['host']
        text+=f"| {label} | {c['length']}/{c['cap']} | {fmt(c['QPS'],6)} | {fmt(m['TTFT_p95_ms'])} | {fmt(m['TPOT_p95_ms'])} | {fmt(m['ITL_p99_ms'])} | {fmt(s['queue']['half_growth'])} / {s['queue']['maximum']} | {fmt(s['P_service_fraction'])} | {fmt(s['D_credit_refusal_snapshot_fraction'])} |\n"
    text+='''
Ouro 16384/16 recovered legal D credit without increasing its P memory pool.
Its 18432-block D pool removed the observed credit refusals in that pilot.
Ouro 18432/16 also passed the frozen Decode screen and retained audited pressure.
Neither condition acquired a qualified repeated physical timeline.
Ouro 24576/16 exceeded the unchanged 53.70702898374411 ms request-average TPOT limit at p95.
The search did not relax that limit.

Huginn 3900/16 initially failed the 3 GiB free-HBM startup guard.
The legal recovery reduced the P pool from 24832 to 24064 blocks.
It preserved the guard and retained 24832 D blocks.
Its subsequent Native pilot still lacked growing audited compute-ready pressure.
The 3900/1024 length mixture also had a decreasing audited queue.
Its refusal prevalence was 0.8727272727272727.
This does not establish that every length mixture fails.

Huginn 4080/16 reached the actual 4096-token limit without changing it.
Its audited queue maximum was one request and its half-growth was negative.
Frequent reservation refusals remained present.
It did not qualify as the requested P compute-pressure condition.

Increasing source transfer slots alone did not improve Huginn 3072/16.
That probe retained two compute slots and added a third transfer lease.
Its TTFT p95 reached 15081.992389727384 ms.
The report does not mislabel that setting as three compute slots.

True three-compute-slot Native probes also failed to improve the baseline.
At 3500 tokens, the paired TTFT ratio was 3.872020065434701.
The completed-QPS ratio was 0.7872705788644426.
At 3072 tokens, four complete D targets fit instead of three.
Even there, the TTFT ratio was 1.2279638994819815.
The E2E ratio was 1.1987190696949572.
The TPOT and ITL ratios were 1.075852085643028 and 1.0642760408700698.
The completed-QPS ratio was 0.9534870856916501.
These slower settings were rejected rather than adopted as weaker baselines.

## Independent capacity reference correction

The Local calibration role used an empty set of Native Prefill graph geometries.
Consequently, earlier Local references used eager fallback despite requesting Prefill graph support.
For example, the earlier Huginn 3072 reference reported zero captures, zero replays, and 320 fallbacks.

The new optional calibration flag supplies Native P geometries before graph-policy installation.
It changes only the independent Local calibration role.
It does not change Serving, weights, Attention, or SLO thresholds.
The corrected references check parity, full work, and actual graph replay.
The final service screen uses the strongest qualified batch within the configured P compute-slot limit.
It also retains the unrestricted reference and the original reference identity.
The required service ratio remains 85%.

These are finite Local calibration cohorts, not sustainable capacity claims.
Prepared D calibration also uses an isolated Local device, not the concurrent online D workload.
Pool mismatches remain explicit in the machine-readable host screens.
They cannot establish exact HBM-matched maximum capacity.

## Physical activity and profiling qualification

The collector computes interval unions across CUDA streams.
It does not sum overlapping kernel durations.
Kernel records, copies, graph enclosures, and CPU API records retain distinct types.
Kernel inactivity does not establish that HBM, PCIe, or RDMA resources are unused.
No SM occupancy or HBM-bandwidth counter is inferred from these intervals.

The acceptance screen remains two-sided at five percent.
It checks response wall time, TTFT p95, TPOT p95, ITL p99, and maximum ITL.
It also checks local ITL p99, local maximum ITL, and Native A/A variation.
All outputs and normal worker exits are required.
This is a point-screen qualification, not a statistical proof of zero observer effect.

### Qualified Huginn pilot

'''
    text+=f'''| Metric | Value | Evidence class |
| --- | --- | --- |
| Common window | {d['window_seconds']:.9f} s | MEASURED |
| P kernel union | {p['prefill']['kernel_busy_percent']:.9f}% | MEASURED |
| D kernel union | {d['kernel_busy_percent']:.9f}% | MEASURED |
| D physical inactivity | {d['physical_kernel_idle_seconds']:.9f} s / {d['physical_kernel_idle_percent']:.9f}% | MEASURED |
| D idle windows | {d['idle_window_count']} | MEASURED |
| Gap mean / p50 | {d['idle_ms']['mean']:.9f} / {d['idle_ms']['p50']:.9f} ms | MEASURED |
| Gap p95 / p99 / maximum | {d['idle_ms']['p95']:.9f} / {d['idle_ms']['p99']:.9f} / {d['idle_ms']['maximum']:.9f} ms | MEASURED |
| Same-window P queue half-growth / maximum | {h['queue']['half_growth']} / {h['queue']['maximum']} | MEASURED host ownership |
| Same-window P service ratio | {h['P_service_fraction']:.9f} | MEASURED finite reference ratio |
| Request-feasible Decode-safe Loop capacity | NOT_MEASURED | No validated counterfactual handoff |

| Gap threshold ms | Count | Total seconds |
| --- | --- | --- |
'''
    for t,z in d['thresholds'].items():text+=f"| {t} | {z['count']} | {z['seconds']:.9f} |\n"
    text+='''
Threshold populations overlap and must not be added.
Most gaps occur inside an existing graph replay.
The graph-identity analysis classifies 729965 such gaps.
Their combined duration is 0.43740768916904926 seconds.
Their maximum is only 0.0031031668186187744 ms.
Another 15962 gaps lie inside recorded Native ModelRunner enclosures.
Only 2304 gaps lie outside those recorded enclosures.
An outside gap still does not prove legal state, credit, or Decode safety.

### Repetition and observer failures

The three independent Huginn N128 physical confirmation sets did not pass qualification.
Their Native A/A comparisons also failed at least one required five-percent field.
One diagnostic trace contained a 3263.690 ms D gap.
It is not accepted as Clean Idle Capacity.
Its specific cause remains NOT_IDENTIFIED.
Some clean references preceded their capture/repeat pair by about an hour.
This timing separation limits causal attribution of observer effects and system drift.

Ouro concurrent-kernel collection repeatedly changed Decode tails beyond the screen.
Short captures, larger preallocated buffers, host-pinned storage, and deferred termination did not resolve qualification.
Quiescent-only capture could observe selected epochs, but could not establish a complete D idle distribution.

Graph-level GPU enclosures offered a conservative alternative, not an exact kernel union.
The synthetic smoke test passed, but model captures contained missing graph timestamps.
The collector retained these records rather than treating them as empty graphs.
Synchronization and collector initialization before model construction did not resolve all missing timestamps.

Runtime API correlation located eight missing graph timestamps in the last graph-scope trial.
Three corresponding launches were provably after capture termination.
Five remained inside or near the captured interval.
Those unresolved records invalidated the bound.
NVIDIA defines zero graph timestamps as unavailable timing information, not absence of GPU work.
[NVIDIA ActivityGraphTrace documentation](https://docs.nvidia.com/cupti/12.8/api/structCUpti__ActivityGraphTrace.html)

The final alternative disabled D CUPTI during the proposed bound measurement.
It used padded multi-stream CUDA-event enclosures instead.
A separate diagnostic had to verify coverage of every actual D kernel first.
That coverage run returned one invalid D activity record and failed normal closure.
The bound, its confirmation repeats, and its planned Phase 2 entry were not run.
All 64 outputs existed, but successful outputs did not override the collector failure.

Clean Ouro physical D idle percentage and complete idle distribution remain NOT_MEASURED.
Clean repeated Huginn capacity also remains NOT_MEASURED.
Diagnostic numbers never become clean capacity by deleting failed records or changing the screen.

## Three marker-free Huginn N128 controls

These starts use Input3072, Output16, QPS1.1434257634805887, P2/D8, and legal pools 24064/24832.
Each start uses 128 distinct prompt IDs.
The arrivals differ by independent seed.
The table does not contain a Loop candidate or a speedup ratio.

| Arrival seed | TTFT p95 ms | E2E p95 ms | TPOT p95 ms | ITL p99 ms | Maximum ITL ms | Completed requests/s | Joint goodput requests/s | ITL exceedances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
'''
    controls=[]
    for seed in (2026111290,2026111291,2026111292):
        n=f'confirm_huginn_L3072_O16_P2_native_N128_a{seed}';z=rows[n]['metrics'];controls.append(rows[n])
        text+=f"| {seed} | {fmt(z['TTFT_p95_ms'])} | {fmt(z['E2E_p95_ms'])} | {fmt(z['TPOT_p95_ms'])} | {fmt(z['ITL_p99_ms'])} | {fmt(z['maximum_ITL_ms'])} | {fmt(z['completed_QPS'],6)} | {fmt(z['SLO_goodput'],6)} | {z['ITL_limit_misses']} |\n"
    text+='''
These finite-cohort drain rates do not establish sustainable online capacity.
The TTFT spread is substantial despite stable Decode percentiles.
The third start also lacks growing audited P pressure over its central host window.
Thus three completed Native starts do not establish three qualified joint-pressure starts.

No Candidate/Native confidence interval exists because the gated candidate was not executed.
Reporting an uncertainty interval around an invented zero improvement would be invalid.
The retained CSV reports every latency percentile, arrival rate, token rate, and attainment denominator.
Marker-free Service TTFT and exact per-request execution queue latency remain NOT_MEASURED.
Host admission or submission timestamps do not replace GPU execution start.

## Correctness and serial continuation costs

The current inventory contains 43 passed state qualification cells.
They include Native N32 checks for both models and new-length N8 Loop checks.
The initial Ouro 16384 Loop3 state attempt failed under an incompatible Atomic configuration.
A new namespace disabled that incompatible option and passed exact state checks.
The failed attempt remains preserved.

The new Huginn 3072 N8 checks cover Loop31 full-prompt and 128/256/512-token tiling.
They also cover full-prompt Loop28.
Ouro new-length Loop3 N8 checks passed at 16384 and 18432 tokens.
These checks include exact outputs, valid KV, Hidden State, recurrent accounting, wire state, ACK, and lifetime.
They do not replace N32 qualification of a new online migration policy.
No new online mixed-cut N32 or N128 migration success is claimed.

The following values are serial CUDA Event enclosures from eight requests in one process start.
They include waits and launch gaps.
They are not exclusive kernel time or a Serving effect.

| Model/input | Path | P enclosure p50 ms | D prompt enclosure p50 ms | D submissions/request | Gross P payload bytes/request | Observed host handoff p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
'''
    for name,label in (
        ('gapcost_huginn_L3072_zero_serial8','Huginn3072 Native'),
        ('gapcost_huginn_L3072_loop31_full_serial8','Huginn3072 Loop31 full'),
        ('gapcost_huginn_L3072_loop31_tile128_serial8','Huginn3072 Loop31 tile128'),
        ('gapcost_huginn_L3072_loop31_tile256_serial8','Huginn3072 Loop31 tile256'),
        ('gapcost_huginn_L3072_loop31_tile512_serial8','Huginn3072 Loop31 tile512'),
        ('gapcost_huginn_L3072_loop28_full_serial8','Huginn3072 Loop28 full'),
        ('gapcost_ouro_L18432_zero_serial8','Ouro18432 Native'),
        ('gapcost_ouro_L18432_loop3_full_serial8','Ouro18432 Loop3 full')):
        z=json.loads((ROOT/'raw'/name/'serial_cost_analysis.json').read_text())
        text+=f"| {label} | Serial diagnostic | {fmt(z['streams']['prefill']['p50'])} | {fmt(z['streams']['decode']['p50'])} | {z['D_prompt_submissions_per_request']} | {int(z['payload_bytes_per_request'])} | {fmt(z['handoff']['p50'])} |\n"
    text+='''
Native already transfers KV.
Gross Loop payload is not incremental overhead over Native.
Huginn Loop31 payload is 324382080 bytes lower than its Native payload in this calibration.
Observed host handoff spans coordinator Prefill completion to D receipt.
It is not wire-only NIXL time or an additive DMA measurement.
Component percentiles must not be added into TTFT p95.

For Huginn tile128, the median whole quantum is 5.1143200397491455 milliseconds.

Its median packed-core graph is 2.3679839372634888 milliseconds.

The longest whole quantum is 8.123711585998535 milliseconds.
All 192 measured graph selections hit the cache, with no fallback or recapture.
The surrounding Runtime cost therefore cannot be attributed to a graph-cache miss in that cell.
Core-only cost still does not include Coda, handoff, or whole-quantum scheduling.

## Fixed-trace Offline Oracle

The qualified Huginn pilot contains eight mapped R31-ready source events.
The replay places measured serial enclosures into the same Native physical gaps.
It assumes zero transfer, free reserved state, and unchanged future Decode behavior.
Those optimistic assumptions do not describe an implemented online Scheduler.

Full-prompt Loop31 requires 40.60753631591797 ms at enclosure p50.
Tile128 requires 24 completed tiles per full Loop.
Tile256 requires 12 tiles and tile512 requires six tiles.
No whole ModelRunner quantum fits the observed maximum gap.
The core-only tile128 optimism fits one tile slot, but completes zero full Loops.
Partial tiles do not count as migrated Loops.

The replay retains p50, p99, and maximum cost alternatives.
All reported completed full-Loop counts are zero under these specific cost models.
This is an OFFLINE_ORACLE budget screen, not a universal compute-only upper bound.
Serial enclosures can exceed exclusive kernel time.
The unchanged Native trace also omits changed admission, transfer overlap, and future batch composition.

Source event mapping does not bound nonlinear clock drift.
The Native source did not retain a committed export snapshot at every observed cut.
Request-feasible Decode-safe capacity is therefore NOT_MEASURED.
Predicted TTFT, E2E, and goodput improvements are NOT_MEASURED.
This short trace cannot support a 384-request or one-third-work capacity extrapolation.

## Figures and missing comparisons

'''
    for name,title in (
        ('figure_1_huginn_actual_kernel_timeline','1. Actual Huginn kernel timeline'),
        ('figure_2_same_window_P_pressure','2. Same-window P busy and audited queue'),
        ('figure_3_huginn_idle_distribution','3. Physical gap distribution'),
        ('figure_4_source_readiness_and_gaps','4. Mapped source readiness and D gaps'),
        ('figure_5_native_confirmation_and_gated_candidate','5. Native repeated controls and unrun candidate'),
        ('figure_6_fixed_trace_budget_screen','6. Serial costs and fixed-trace budget screen')):
        text+=f'- {link(title,"figures/"+name+".png")}\n'
    text+='''
The figure manifest records each source and evidence class.
No plot inserts a conceptual TO-BE as an actual GPU trace.
No unrun migration arm receives a zero latency or zero goodput value.
The migration-fraction versus TTFT–ITL Pareto frontier is NOT_RUN.
An actual P-to-D migration timeline for Phase 2 is also NOT_RUN.

## Final answers

1. Neither model has a fully qualified, repeated Prefill-compute-heavy and Decode-safe migration condition in this campaign.
2. Huginn3072/16 and Ouro16384/16 or 18432/16 remain discovery candidates, not qualified final operating points.
3. Huginn's accepted short pilot measures P kernel busy at 97.29148492037717%.
4. The same pilot measures D kernel inactivity at 14.018167396768423%, mostly very short gaps.
5. Verified request-feasible Decode-safe milliseconds remain NOT_MEASURED.
6. Phase 2 completed migration counts are NOT_RUN, not measured zero-work candidate results.
7. Phase 2 TTFT improvement is NOT_MEASURED.
8. Candidate-relative ITL, TPOT, and TBT non-inferiority are NOT_MEASURED.
9. The main remaining gates are repeated clean observation and useful source-ready, handoff-feasible execution windows.
10. These results do not justify Phase 3 or Phase 4 development.

The measurements do not establish NO_GO_PHYSICAL_HEADROOM for all workloads.
They also do not establish a Decode-safe net benefit.
Further large scheduler work is not supported by this campaign alone.
The separate queued analysis will examine architectural novelty without assuming a positive result.

## Evidence and reproducibility

'''
    for label,target in (
        ('Run status and all planned cells','analysis/campaign_run_status.csv'),
        ('Serving latency, token rates, attainment, and queue screens','analysis/serving_results.csv'),
        ('Structured inventory and qualification details','analysis/final_campaign_inventory.json'),
        ('Matched graph capacity reference repair','analysis/matched_graph_capacity_references.json'),
        ('Fixed-trace packing details','raw/'+physical+'/fixed_trace_enclosure_packing.json'),
        ('Figure sources and evidence classes','figures/figure_manifest.json'),
        ('Original preservation signatures','state/preservation.json'),
        ('GPU closure','state/CAMPAIGN_CLOSED.json'),
        ('All-eight Burner generation verification','state/final_burner_verification.json')):
        text+=f'- {link(label,target)}\n'
    text+='''
Large raw files are recoverable from the lossless evidence bundle and its SHA256 manifest.
The local original evidence remains unchanged.
Per-run source ZIP files preserve the exact submitted code.
The bundle excludes redundant extracted source trees, bytecode, locks, and the live Burner log.
It includes a frozen byte prefix of the Burner log.
The repository source contains the bundle verifier and extraction instructions through the inherited verifier API.

Publication branch: `flashvep/prefill-heavy-decode-safe-20261111`.
The publication receipt records the verified evidence commit.
The final local remote-verification receipt records the metadata child SHA.
The user-facing completion message supplies that final GitHub link.
'''
    report=REPO/'poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_final_summary.md'
    report.write_text(text)
    print(report)


if __name__=='__main__':main()
