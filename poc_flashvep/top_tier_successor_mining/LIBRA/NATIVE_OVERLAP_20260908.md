# Native supplied-runtime overlap diagnostic

Source: `online/libra_native_profile_m2048_20260908/profile.sqlite` and
`analysis/libra_native_nsys_20260908/` under the result root.

Four physical H1004–7,real Qwen3-30B-A3B48-layer text prefill,M2048/source,
one four-source input group,three warmups and three randomized paired measured
executions. This is the author's native SGLang replication/local-remote path,
not vLLM DeepEP. All12 measured Libra first tokens agree with native vanilla.

## Attribution contract

24 rank observations represent six coupled four-rank executions,not24 independent
runs. Same-process CUDA activity is associated with the launch API by correlation
id and then the shortest enclosing CPU NVTX range on the launching thread.
Activity unions/intersections use one device at a time. No cross-GPU absolute
CUDA-event subtraction,rank-summed request time or CPU-range coincidence mapping.

| Median rank-observation diagnostic,ms | Native vanilla | Native Libra |
|---|---:|---:|
| Profiled host prefill span |115.572|275.141|
| Union of observed GPU kernel/copy activity |105.465|163.553|
| Host span with no observed GPU activity |9.847|115.500|
| Expert-stage kernel union |52.621|61.562|
| Memcpy union |1.181|24.175|
| NCCL kernel union |31.882|57.605|
| Copy/expert temporal intersection |0|4.476|
| NCCL/expert temporal intersection |0|6.191|

The median of a difference need not equal the difference of the medians. Nested
stage totals are not additive. Memcpy includes metadata/dispatch/buffer traffic,
not exclusively expert weights. `Prefetch` and `Copy Buffer` launch-associated
memcpy time medians are8.204 and10.214ms,respectively. Copy/compute and collective/
compute overlap **really exists**; the tested native path is not simply serial.

Native Libra CPU NVTX unions across layers include Dispatch41.091ms,
MoE Remote33.681ms,Prefetch32.560ms and MoE Local28.166ms. These are range elapsed
times including API calls/waits,not isolated pure-Cython processing costs. Only
570/52,224 Libra activities lack a narrower stage attribution; they remain labeled
UNATTRIBUTED rather than assigned a convenient cause.

## What this does and does not establish

The execution pays many small launch/control/copy costs alongside genuine overlap.
This is consistent with the original paper's acknowledged short-window limitation.
Nsight perturbs host execution,so these numbers must not replace clean prefill
measurements or become a115ms removable-E2E oracle. They do not prove that all
no-activity intervals are removable CPU-planner waste.

The larger M8192/source oracle-interleaved diagnostic failed functional parity
and is explicitly invalidated. A fresh no-oracle native control passes80/80
first-token comparisons,min cosine.999526;vanilla377.645ms versus Libra398.134ms
rank-critical medians,paired reduction−5.401%[−6.045,−5.214]%. Thus the gross
large-input output corruption is isolated to the diagnostic path,not established
as an original Libra error. Do not use the invalidated run's apparent1.62%
prediction benefit. The precise hook/native-state interaction is not claimed
to be diagnosed; it is outside the valid opportunity evidence.
No new MLLM-specific failure or cross-model-quality claim follows from this trace.
