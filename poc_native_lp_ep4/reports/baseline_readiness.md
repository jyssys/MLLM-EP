# Baseline readiness

Baseline model/config audit: PASS, frozen revision
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, d2048/m768,
128 experts, top8, 48 decoder layers. Raw checkpoint weights use input-major
gate/up `[128,2048,1536]` and down `[128,768,2048]`; the installed loader converts
these to its kernel layout. Physical owner map is linear, 32 experts per EP rank.

Actual runtime placement: EP0/TP0/DP0→physical0 owns experts0–31;
EP1/TP1/DP0→physical1 owns32–63; EP2/TP0/DP1→physical4 owns64–95;
EP3/TP1/DP1→physical5 owns96–127. TP groups are (EP0,EP1) and
(EP2,EP3); the EP group is all four ranks. EP=TP×DP=4, not sixteen GPUs.
Rich traces assert each loaded expert map rather than relying on the launch
flag. Each local bank is gate/up [32,1536,2048], down [32,2048,768] in
BF16, **9,437,184 bytes/expert**. Actual DeepEP prepare/finalize and sparse
Triton expert calls are recorded on all four ranks; no masked dense HF path
or four independent replicas is used. Logical prefill source rows are SP
partitioned across the source TP pair, not counted twice as TP copies.

Environment: vLLM **0.20.0+cu129**, torch **2.11.0+cu129**, CUDA **12.9**,
DeepEP **1.2.1+73b6ea4** high-throughput. This port pins eager
execution, FlashAttention, TP2/DP2/EP4/PP1, DBO/EPLB/prefix caching off and no
redundant experts. The prior experiment also used eager execution. All arms use
the same settings. GPU mask is 0,1,4,5; UUIDs, topology and external activity are
saved in `artifacts/gpu_audit/environment.json`.

The four authorized devices are H100 80GB HBM3 with NV18 connections in the
recorded NVSwitch topology. CUDA logical 0/1/2/3 map to physical 0/1/4/5:
`f217c8a0...`, `a77f3471...`, `6076e2f2...`, `a1a1cfcf...` respectively.
Full UUIDs and PCIe/NIC/CPU affinity are retained in the audit. No hardware
clock-setting changes or package upgrades were performed.

Runtime reservations are 4 GiB KV/rank, maximum context 20000, token capacity
20000 and five sequences per DP (four residents plus the one injection on DP0).
CP switches the real scheduler token budget to its chunk size plus four resident
rows; LP keeps full scheduled prompt rows and changes only its active layers.
The installed E32/N768 H100 sparse-Triton tuned config is absent; the same
installed default expert kernel/config is used for all arms. This is disclosed,
not silently replaced with another model/shape or calibration.

Historical quality evidence is reused from the Part1 completion run rather than
the older root bundle's NOT_RUN fields: full MMStar N1500, reference 71.985% /
EP4 72.220%; ChartQA N2500, reference 85.600% / EP4 85.400%. Public protocol
remains partial. No new quality benchmark run is needed for this lossless
scheduler experiment; native LP numerical correctness is tested separately.

The reused completion bundle is
`poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/frozen_baseline_bundle.json`,
not an older root bundle containing NOT_RUN entries. Public task evidence is
not relabeled as theoretical quality preservation.

Primary workload is the frozen prior 2029-token prompt (1980 image rows) and
2033-token matched text. Four frozen residents are duplicated as independent
requests across two DP schedulers, giving global B8. Request identities and
teacher-forced continuations are retained per DP; injection occurs on DP0 only.

Other GPUs carry workload and share the NVSwitch/host. They are outside the
authorized set and remain untouched. Paired/interleaved repeats therefore
control shared-environment drift; this is not an exclusive whole-server result.
