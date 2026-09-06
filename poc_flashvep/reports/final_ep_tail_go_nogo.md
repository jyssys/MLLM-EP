# Final fixed-shape / history-dependent DeepEP tail gate

Date: 2026-09-06  
Branch: `flashvep/final-ep-tail-go-nogo`  
Model: Qwen3-VL-30B-A3B-Instruct (local snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`)  
Topology: TP2 / DP2 / EP4 / PP1, BF16, vLLM 0.20.0 V1, DeepEP `1.2.1+73b6ea4`, high-throughput, Triton unquantized MoE, eager, DBO off, prefix cache off.  
Devices: physical GPUs 1--4 only (`CUDA_VISIBLE_DEVICES=1,2,3,4`).

## Final decision

**FINAL DECISION: FINAL_NO_GO**

The earlier MoE-only/wave-critical perfect-oracle projection (16.11% at a
matched-p50 baseline and 23.83% at an optimistic p25 baseline) was not a
request-level result. A new live run with request context and scheduler IDs
provides the required conservative join: 84 requests, with a capped
request-level removable upper bound of **1.09% aggregate** (median 0.29%, p90
1.77%; 15.63% maximum single request). Removing the largest logical event
reduces this to 0.62%, and removing the top five to 0.38%. This is an upper
bound because co-batched invocation excess is assigned to every participating
request and then capped by that request's observed E2E. It cannot support the
required 12% actual E2E gate.

The causal mechanism is still real and dispatch-first, but the economic and
feasibility gates for a method are not met. No production optimization was
implemented.

## Tail definition and existing primary population

For each `(run, phase, layer, M)` group, normal is the matched group
distribution. Normal is `<=p95`, moderate is `p95--p99`, severe is
`p99--p99.9`, extreme is `>p99.9` and at least 10x the group median, and giant
is at least 100x or 100 ms. Thresholds 5/10/20/50/100 ms are secondary.

The existing three-run stock population has 89,664 logical invocations and
150,486.718 ms of logical MoE time. With a matched-p50 baseline, excess mass
is 28.14% for all positive excess, 12.10% for `>20 ms`, and 11.05% for
`>100 ms`. The optimistic p25 all-excess mass is 39.29%. These numbers are
valid MoE-level/wave-critical projections, not direct request latency.

## New direct live validation

`direct_stock_run4` is a fresh real-image vLLM serving run on physical GPUs
1--4 (12 waves, 84 output-bearing request records, 6,912 contextual logical
MoE invocations after TP deduplication). The run captured scheduler request
IDs in the child engine process and saved request metrics. The local join
maps the numeric scheduler-ID prefix to `RequestOutput.request_id`; for
co-batched events it caps assigned excess at each request's measured E2E.
No rank rows are summed and no cross-device CUDA timestamp subtraction is
performed.

| direct contextual set | n | p50 ms | p90 ms | p99 ms | p99.9 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 4,896 | 2.433 | 4.093 | 5.233 | 8.853 | 251.951 |
| decode | 2,016 | 2.282 | 2.932 | 3.084 | 3.771 | 239.909 |

The largest events remain DeepEP dispatch-dominant (for example 251.95 ms
dispatch with 0.44 ms expert and 0.05 ms combine). Thus the mechanism is
reconfirmed, while its directly attributable request-level mass is small in
the instrumented run.

### Direct request-level impact upper bound

Across 84 requests, observed E2E (TTFT plus the measured first-to-last-token
interval) is 156,643.356 ms. Capped assigned excess is 1,710.985 ms:

- mean share 1.09%, median 0.29%, p90 1.77%, p99 13.16%, maximum 15.63%;
- 10 requests have an assigned event over 10/20 ms and 4 over 100 ms;
- without the largest logical event: 0.62% aggregate;
- without the five largest: 0.38% aggregate.

This is intentionally an optimistic upper bound, not a claim that all of the
assigned excess lies on each request's unique critical path. The aggregate
is far below 12% even before correcting for overlap and policy overhead.

## Runtime generality

The current runtime is DeepEP V1 (`deep_ep` `1.2.1+73b6ea4`) and exposes only
the high-throughput and low-latency choices in this vLLM build. The low-
latency attempt was made without changing source and failed during engine
initialization at the documented `nvshmem_qp_depth >= 2*(max_dispatch+1)`
assertion; it produced no serving measurements. No DeepEP V2 package/build is
installed in the validated environment, and no isolated V2 build was
available without replacing the baseline runtime. Allgather/reduce-scatter
is not an available option in this vLLM configuration. Consequently the
current classification is **LEGACY_OR_CURRENT_HT_ONLY_UNRESOLVED**, not a
cross-backend result; this is itself a failed generality gate for a paper
claim.

The source path is `deepep_ht.py` -> `dbo_get_previous_event` ->
`Buffer.get_dispatch_layout/dispatch/combine(previous_event=...)` ->
`EventOverlap.current_stream_wait`; DBO is off and the Python previous-event
object is absent in this path. The prior Nsight trace nevertheless showed
`deep_ep::intranode::notify_dispatch` and `barrier` waits.

## Feasible oracle and interventions

The prior wait-aware study provides the only repeated policy comparison:
global synchronize was diagnostic only; selective stream drain and a simple
previous-dispatch policy were unstable, reduced throughput by more than the
allowed 2--3% in aggregate, and did not provide robust p99/extreme-tail
benefit. The new direct run's observed downstream wait proxy sums to only
13.2% of logical MoE span and underestimates the internal dispatch wait. No
direct timeline supports claiming more than the request-level 1.09% capped
upper bound as a feasible oracle.

| oracle/intervention | result |
|---|---|
| Perfect zero-cost (old wave projection) | 16.11% p50 / 23.83% p25 projection; not request-level |
| Existing overlap-slack evidence | no direct request gain; bounded by 1.09% in new join |
| Communication-priority oracle | not measured as a safe runtime action |
| Backpressure/pacing oracle | not measured; no request-level evidence |
| ALWAYS_SYNC / selective drain | prior repeated runs: unstable and throughput-costly |

The best realistic oracle is therefore **below the required 18% gate** by
direct request evidence. Moving a wait earlier can improve a stage counter
without removing work from the request critical path; the prior controls
demonstrate this failure mode.

## Hard gates

1. **Direct E2E gate:** failed. The direct request-level capped upper bound is
   1.09% aggregate, with only an isolated single-request maximum above 12%.
2. **Runtime generality gate:** failed/unresolved. Low-latency is blocked by
   an unmodified V1 NVSHMEM queue-depth requirement; V2 is unavailable in the
   current environment.
3. **Feasible-oracle gate:** failed. No safe selective intervention has
   demonstrated the required 18% direct E2E headroom.
4. **Method gate:** not run. It would be invalid to implement or headline a
   method without passing the preceding gates.

## Conclusion

The communication-debt causal phenomenon remains a useful engineering
observation: previous asynchronous DeepEP work can inflate the next
dispatch, while expert and combine stay ordinary. It is not, however,
economically large enough in the directly joined serving run to justify a
12%-E2E research method. Further engineering is unlikely to bridge the gap
without changing the workload, counting projections as E2E, or paying the
throughput cost that the acceptance rule forbids.

**DO_NOT_PURSUE:** a production spillover-prevention method for this exact
vLLM 0.20 / DeepEP HT path. Retain the instrumentation and raw traces as an
engineering regression test; revisit only with a newer runtime that exposes
request-critical communication debt and demonstrates a materially larger
direct E2E share.

## Reproducibility

- New live runs: `poc_flashvep/deepep_revalidation/results/final_ep_tail_go_nogo_20260906_173000/`.
- Direct analyzer: [`analyze_final_gate.py`](/home/esjung/MLLM-EP-github/poc_flashvep/final_ep_tail_go_nogo/analyze_final_gate.py).
- Existing root-cause report: `poc_flashvep/reports/fixed_shape_tail_root_cause_report.md`.
- Existing wait-aware report: `poc_flashvep/reports/deepep_wait_aware_tail_poc.md`.
