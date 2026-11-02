"""Build auditable English reports from terminal evidence, with explicit gaps."""
import collections
import statistics
from .common import *
from .analyze import records,table

FINAL=REPO/'poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom_final_summary.md'


def md(headers,rows):
    out=['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
    for row in rows:
        out.append('| '+' | '.join(str(v).replace('|','/') if v is not None else 'NOT_MEASURED' for v in row)+' |')
    return '\n'.join(out)


def fmt(value,digits=3):
    return f'{value:.{digits}f}' if isinstance(value,(int,float)) and not isinstance(value,bool) else str(value) if value is not None else 'NOT_MEASURED'


def load(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    runs=load(ROOT/'serving_runs.json',[]);index={r['name']:r for r in runs}
    configs={p.stem:json.loads(p.read_text()) for p in (ROOT/'configs').glob('*.json')}
    configs={k:v for k,v in configs.items() if isinstance(v,dict) and 'name' in v}
    gate_rows=load(ROOT/'p_d_gate_results.json',[]);gindex={r['name']:r for r in gate_rows}
    confirmation=load(ROOT/'state/confirmation_plan.json',{'selection':[],'runs':[]})
    counts=collections.Counter();failures=[];unrun=[];correctness=[]
    for name,c in sorted(configs.items()):
        folder=ROOT/'raw'/name;terminal=load(folder/'dispatcher_result.json')
        status='COMPLETE' if terminal and terminal.get('complete') else 'FAILED' if terminal else 'UNFINISHED' if folder.exists() else 'NOT_RUN'
        counts[status]+=1
        if status=='FAILED':failures.append([name,c['model'],terminal['returncode'],load(folder/'watchdog.json',{}).get('reason','See preserved log')])
        if status in ('NOT_RUN','UNFINISHED'):unrun.append([name,c['model'],status,'No measurement or success is inferred'])
        check=load(folder/'recovery_gate.json')
        if check:
            correctness.append([name,c['model'],c['count'],check['passed'],json.dumps(check,separators=(',',':'))])
    profiles=[]
    for p in (ROOT/'raw').glob('*/profile_analysis.json'):
        value=json.loads(p.read_text())
        profiles.append([p.parent.name,value.get('status'),value.get('recorded_kernel_invocations'),
            fmt(100*value['kernel_busy_fraction']) if value.get('kernel_busy_fraction') is not None else 'NOT_MEASURED',
            fmt(100*value['matched_wall_difference_fraction']) if value.get('matched_wall_difference_fraction') is not None else 'NOT_MEASURED',value.get('eligible_for_matched_clean',False)])
    policy_rows=[]
    for r in runs:
        c=configs.get(r['name'],{})
        if c.get('measurement_epoch')!='WARM_GRAPH_DURABLE_V2' or c.get('phase') not in ('qps_pilot','aggregated_QPS_pilot','final_repeat'):continue
        policy_rows.append([r['name'],r['offered'],r['completed'],fmt(r['target_QPS']),fmt(r['requests_per_s']),
            fmt(r['TTFT_ms_p95']),fmt(r['request_TPOT_ms_p95']),fmt(r['token_ITL_ms_p99']),fmt(r['SLO_goodput'])])
    graph_rows=[]
    for name,c in configs.items():
        if c.get('phase')!='graph_warmup_control':continue
        folder=ROOT/'raw'/name;captures=[]
        for p in folder.glob('graph_captures_*.jsonl'):captures.extend(records(p))
        measured=[r for r in captures if r['measured_phase']]
        r=index.get(name,{})
        graph_rows.append(dict(name=name,model=c['model'],warmed=c['prime_all_graph_shapes'],async_evidence=c['async_evidence'],
            completed=r.get('completed'),measured_capture_calls=len(measured),measured_capture_host_seconds=sum(v['host_seconds'] for v in measured),
            ITL_p99_ms=r.get('token_ITL_ms_p99'),TPOT_p95_ms=r.get('request_TPOT_ms_p95'),TTFT_p95_ms=r.get('TTFT_ms_p95'),
            evidence='MEASURED_NATIVE_CAPTURE_HOST_SPANS_AND_REQUEST_TIMES',scope='Capture wall spans can include synchronization and garbage collection. They are not exclusive GPU kernel time.'))
    table('graph_warmup_controls.csv',graph_rows)
    verdict={}
    gate_table=[]
    for model in ('ouro','huginn'):
        candidates=[r for r in gate_rows if r['model']==model and r['name'].startswith(('profile_warm_','long_profile_'))]
        chosen=max(candidates,key=lambda r:(r.get('profile_perturbation_qualified',False),r.get('offered',0)>=128,
            r.get('P_compute_gate',False),r.get('D_SLO_gate',False)),default=None)
        selection=next((r for r in confirmation['selection'] if r['model']==model),{})
        decision='NOT_CONFIRMED_ON_TESTED_GRID'
        verdict[model]=dict(model=model,verdict=decision,Dynamic_A='NO_GO_EXPANSION_WITHOUT_SAME_START_JOINT_PROOF',
            P_ready='PASS' if chosen and chosen.get('P_ready_lower_bound_half_growth',0) and chosen['P_ready_lower_bound_half_growth']>0 else 'NOT_CONFIRMED',
            P_compute='PASS' if chosen and chosen.get('P_compute_gate') else 'NOT_CONFIRMED',
            D_credit='PASS' if chosen and chosen.get('D_credit_gate') else 'NOT_CONFIRMED',
            D_SLO='PASS' if chosen and chosen.get('D_SLO_gate') else 'NOT_CONFIRMED',
            D_borrowing='NOT_MEASURED',same_start_evidence=chosen,selected_pilot=selection,
            scope='Failure on the tested grid does not prove recurrence makes the target impossible. Missing kernels do not imply zero activity. No original-HF or downstream-accuracy claim.')
        if chosen:gate_table.append([model,chosen['name'],fmt(chosen.get('P_actual_kernel_busy_central')*100) if chosen.get('P_actual_kernel_busy_central') is not None else None,
            fmt(chosen.get('P_completion_rate_in_interval')),fmt(chosen.get('P_fraction_reference')),fmt(chosen.get('P_ready_lower_bound_half_growth')),
            fmt(chosen.get('D_request_TPOT_p95_ms')),fmt(chosen.get('D_token_ITL_p99_central_ms')),chosen.get('D_min_coordinator_free_blocks'),chosen.get('classification')])
    write(ROOT/'final_per_model_verdict.json',verdict)
    (ROOT/'correctness_and_parity.md').write_text('# Internal correctness and parity\n\nThese checks preserve fixed R4/R32, BF16, native KV values, and qualified numerical geometry.\nInternal parity does not establish original-HF fidelity or downstream accuracy.\nA byte gate and an output gate have different denominators.\n\n'+md(['Run','Model','Offered requests','All recorded checks pass','Exact check payload'],correctness)+'\n\nCommon-request comparisons reside in each `subset_output_parity.json`.\nThe `compared` field states their actual denominator.\nFinal-repeat work checks reside in `confirmation_work.json`.\nIndependent-start output checks reside in `state/cross_start_output_parity.json`.\nHeavy SHA readback runs are not clean timing results.\n')
    (ROOT/'failures_and_unrun.md').write_text('# Failures and unrun configurations\n\nNo failed namespace is overwritten.\nA completed response cohort can still fail profiler export or native shutdown.\nThose failures remain failures.\n\n'+md(['Terminal status','Configurations'],sorted(counts.items()))+'\n\n## Failed jobs\n\n'+md(['Run','Model','Return code','Reason'],failures)+'\n\n## Unrun or unfinished configurations\n\nOlder superseded plans remain visible.\nTheir absence is not a correctness failure or a performance measurement.\n\n'+md(['Run','Model','Status','Interpretation'],unrun)+'\n')
    (ROOT/'reproduction_commands.md').write_text('''# Reproduction commands

Run commands from `/home/esjung/MLLM-EP-github`.
Use `/home/esjung/.venvs/looped-pd-abc/bin/python` as `PYTHON` below.
Create a new run name before repeating an immutable completed configuration.
Do not delete an old namespace to force a rerun.

```bash
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.tests
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.run NEW_CONFIGURATION_NAME
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.analyze
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.capacity_analysis
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.queues
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.gates
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.accounting
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.paired_analysis
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.reporting
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.figures
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.timeline_figures
```

Each `configs/NAME.json` includes the complete input IDs and arrival offsets or names their pinned dataset.
Each `raw/NAME/source.zip` records the actual executed harness and private runtime diff.
`commands.jsonl` records the launch, config hash, source archive hash, and execution baseline commit.
`raw/NAME/isolation.json` records physical GPU IDs, UUIDs, driver, and other active processes.
`state/preservation.json` records the original Git index signature and private runtime diff signature.

Publication uses a separate results branch and temporary index.
It does not alter the active staged index.
Use `archive_evidence restore --destination EXPLICIT_EMPTY_DIRECTORY` to reconstruct oversized evidence from lossless archive parts.
''')
    text='''# KV admission recovery and Prefill-heavy PD headroom

## Decision and scope

This study does not establish the strict Prefill-heavy and SLO-safe Decode-spare hypothesis on the tested grid.
Admission recovery and profiler recovery are separate results.
Neither result alone proves safe Decode-side borrowing.
The report preserves negative controls and measurement failures.

The model depth remains Ouro R4 and Huginn R32.
The model weights, BF16 precision, and native attention semantics do not change.
The existing numerical patches remain enabled in all relevant comparisons.
Original-HF equivalence and downstream accuracy are not established here.

## Source and environment

The execution baseline commit is `f4ecea59f36f620c69c4fef09c590bf790802cd9`.
The private runtime HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Its preserved working diff SHA256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
Each run records its actual source archive hash.
The working branch is `flashvep/looped-pd-kv-profile-headroom-recovery`.
The publication branch is `flashvep/looped-pd-kv-profile-headroom-recovery-results`.
The final Git SHA appears in the verified publication receipt and final user response.
A commit cannot contain its own hash without changing that hash.

The environment uses Python 3.14.6, torch 2.11.0+cu129, CUDA 12.9, transformers 5.16.1, NIXL 1.4.1, and FA4 4.0.0b30.
Ouro revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Huginn revision is `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
NVML reports 85,520,809,984 physical bytes, or 79.6474609375 GiB, per H100.
CUDA reports 85,028,372,480 managed bytes, or 79.1888427734375 GiB.
Earlier percent_HBM columns use CUDA-visible capacity. The physical-memory audit supplies both explicit denominators.
Clean measurements use one isolated GPU pair while all other managed burners remain stopped.
The experiment does not assume independent pairs are free of shared-host interference.

## Data and frozen decisions

Prompts come from the pinned ShareGPT and LongBench public-text pool.
Length-matched prefixes are controlled workloads, not natural Azure requests.
Generation uses greedy ignore-EOS output caps.
This study does not report these outputs as natural response-length distributions.
The 24 pilot prompts and 128 final-repeat prompts are disjoint.
Final repeats use three arrival seeds and physical pairs 0/1, 4/5, and 2/3.
All paired PD and DP2 starts use identical prompts, generation seeds, and arrivals.
Calibration prompts are separate from both sets.
Huginn input plus output remains within 4096 tokens.

SLO limits freeze before held-out pilots.
The primary joint request SLO uses TTFT and request-average TPOT.
Decode-spare assessment also requires the separately frozen token-ITL tail limit.
The tighter per-metric Atomic/C2048 serial calibration determines each limit.
No candidate result changes these limits.

## Admission recovery

The original Ouro 8192/64 condition has a 66-GiB KV pool and about 6.047 GiB per reserved long request.
Ten reservations fit in 22,528 native blocks.
The next request can fail D reservation before P compute starts.
This is an admission limit, not evidence of P GPU compute saturation.

Supported controls reduce D workspace, enlarge the legal D pool, bound draining reservations, and tune wire credit.
The D pool increases to 25,088 blocks, or 73.5 GiB.
Native ACK and buffer-lifetime checks remain active.
No state releases before a verified native acknowledgment.
The P pool remains unchanged in these controls.
The exact before/after request metrics remain in `serving_runs.csv`.
Memory categories and percentages remain in `gpu_memory_inventory.csv` and `workload_capacity_envelope.csv`.
Raw refusal frequency is not exclusive waiting time.
Binding refusal and co-occurring refusal remain separate fields.

Huginn reserves 256 uniform planes while only 132 planes carry valid state.
The 48.4375% difference is structural accounting.
This study does not implement compact physical allocation or claim that difference as reclaimed HBM.
The native wire path already excludes unused boundary planes.
The 3500/32 control can fill all four D reservations exactly.
The lighter 2560/32 and 3072/32 cases test the capacity/compute trade-off separately.

## Profiling recovery

Historical broad Nsight starts stopped after 80/128 and 14/128 responses.
The exact first blocked operation remains unknown.
Selected-stage process startup and bounded cohorts restore measured progress without replacing model execution.
Several Nsight controls still exceed the predeclared 5% perturbation limit.
Those traces remain diagnostic only.

The native CUDA-only profiler passes a matched 24-request Huginn 3500/32 comparison.
It records 472,272 measured-window kernel invocations with exact 24/24 output parity.
The matched wall difference is +1.351%.
Its whole-window P kernel union is 63.349%.
Kernel union is not SM occupancy or an exclusive E2E component.
The attempted three-second activity toggle returns no kernels despite completed inference.
That empty collector result is not zero GPU activity.
Subsequent matched profiles use the checked full-cohort collector.

## Generic runtime controls

The original evidence writer synchronously fsyncs individual records on a scheduling thread.
The replacement reuses the project's durable background writer.
It preserves record contents and checks enqueued/written counts before accepting a terminal result.
This change does not disable evidence or change model arithmetic.

Native CUDA graphs capture new row geometries lazily.
The warm-up control exercises supported row shapes with real fixed-depth startup requests.
It preserves the original full-length warm-up first.
It applies the same startup principle to PD and Native Mixed DP2.
Additional warm-up requests do not enter measured latency or throughput denominators.
`graph_warmup_controls.csv` reports actual measured-phase capture counts and request tails.
Capture host spans include synchronization and garbage collection.
They are not pure GPU execution time.

Exact CPU descriptor packing is a generic transport control.
Its CPU replay reduces input descriptors without changing the final coalesced descriptors.
CPU replay timing is not an E2E speedup.
The warmed packed/unpacked GPU controls report actual request effects separately.
None of these generic controls is a new Looped-specific method.

## P/D capacity and causal gates

`independent_capacity.csv` contains measured finite P-only and prepared-D service results.
P-only service excludes D reservation and handoff.
Prepared D uses actual native valid KV imports before starting its Decode timer.
Neither finite rate proves sustainable online capacity.
Three repeats within one process are not three independent starts.

The P gate requires growing D-clear P-ready backlog, at least 80% kernel union, and at least 85% of the strongest matched finite P reference.
The ready-queue count excludes every submitted unretired request.
D-reservation waiting is not P compute-ready waiting.
The D gate requires legal free blocks and request TPOT p95 below 90% of its frozen limit.
The separate token-ITL p99 limit also applies.
All conditions must overlap within one server start.
Borrowing remains unproven without actual added D work with checked state and a no-regression comparison.

'''
    text+='## Execution counts\n\n'+md(['Terminal status','Configurations'],sorted(counts.items()))+'\n\n'
    text+='## Warmed serving results\n\nAll listed latency values are milliseconds.\nThese are finite arrival-to-drain results, not proven sustainable capacity.\n\n'+md(['Run','Offered','Completed','Target req/s','Completed req/s','TTFT p95','TPOT p95','ITL p99','Joint SLO goodput'],policy_rows)+'\n\n'
    text+='## Same-start kernel and SLO assessment\n\n'+md(['Model','Actual profiled run','P kernel union %','P completed/s','P/reference','Ready half-growth','D TPOT p95','Central D ITL p99','D free blocks min','Classification'],gate_table)+'\n\n'
    text+='## Confidence intervals and full metrics\n\n`paired_repetitions.csv` records each independent PD/DP2 pair.\n`paired_start_uncertainty.csv` reports two-sided Student-t 95% intervals across three starts.\nA missing third start produces no invented interval.\nThe raw table retains mean, P50/P95/P99 request metrics, output throughput, token tails, and peak HBM.\n`e2e_component_shares_summary.csv` reports duration-weighted exclusive host phase percentages.\nThose phases sum to the host E2E interval.\nThey do not identify exclusive dispatch, GEMM, or DMA percentages.\nCumulative overlapping transfer durations cannot substitute for those percentages.\n\n'
    text+='## Limits and next step\n\nThe tested grid does not prove that recurrence prevents Prefill-heavy serving.\nQualified fixed-row numerical geometry, host submission, KV layout, transport, and model depth can all affect these results.\nStructural row padding applies only to selected projections, not the entire model.\nThe study retains those qualified fixes rather than exchanging correctness for throughput.\nA full Dynamic A campaign is not justified without same-start compute saturation and safe borrowing evidence.\nRemaining unrun workloads and failed jobs remain listed in `failures_and_unrun.md`.\nThe figure manifest names every exact plotted source.\nThe reference-code audit states why LoopServing is not a drop-in runtime replacement.\nReproduction commands and immutable configuration/source archives remain available.\n'
    FINAL.write_text(text)
    write(ROOT/'state/reporting.json',dict(time=time.time(),counts=counts,final_report=str(FINAL),scope='Generated only from recorded artifacts. Review required before publication.'))
    print(json.dumps(dict(counts=counts,final_report=str(FINAL))))


if __name__=='__main__':main()
