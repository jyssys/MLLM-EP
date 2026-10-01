# Baseline readiness

Baseline model/config audit: PASS, frozen revision
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, d2048/m768,
128 experts, top8, 48 decoder layers. Raw checkpoint weights use input-major
gate/up `[128,2048,1536]` and down `[128,768,2048]`; the installed loader converts
these to its kernel layout. Physical owner map is linear, 32 experts per EP rank.

Environment: vLLM 0.20.0, torch 2.11.0+cu129, DeepEP HT. This port pins eager
execution, FlashAttention, TP2/DP2/EP4/PP1, DBO/EPLB/prefix caching off and no
redundant experts. The prior experiment also used eager execution. All arms use
the same settings. GPU mask is 0,1,4,5; UUIDs, topology and external activity are
saved in `artifacts/gpu_audit/environment.json`.

Historical quality evidence is reused from the Part1 completion run rather than
the older root bundle's NOT_RUN fields: full MMStar N1500, reference 71.985% /
EP4 72.220%; ChartQA N2500, reference 85.600% / EP4 85.400%. Public protocol
remains partial. No new quality benchmark run is needed for this lossless
scheduler experiment; native LP numerical correctness is tested separately.

Primary workload is the frozen prior 2029-token prompt (1980 image rows) and
2033-token matched text. Four frozen residents are duplicated as independent
requests across two DP schedulers, giving global B8. Request identities and
teacher-forced continuations are retained per DP; injection occurs on DP0 only.

Other GPUs carry workload and share the NVSwitch/host. They are outside the
authorized set and remain untouched. Paired/interleaved repeats therefore
control shared-environment drift; this is not an exclusive whole-server result.
