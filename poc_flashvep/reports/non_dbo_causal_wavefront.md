# Non-DBO Causal Stage-Wavefront

`NON_DBO_CAUSAL_WAVEFRONT: NO-GO`

The final A/S measurement is the v3 run, captured with source SHA
`e63becaf7fa73efaf50f506287462233e6f9cd1c`. The earlier v2 run is retained
as an unreported plumbing retry and is not pooled with these numbers.

## Stage 0 gate

- A median: 94.6203 ms.
- S median: 181.4859 ms.
- Split overhead S/A: 1.9180×.
- Median zero-contention A→W oracle speedup: 0.6836×.
- Gate: **NO-GO** (W requires >=1.10×).
- DBO is disabled in both variants; S uses one host owner thread and one compute stream.

| Request | A ms | S ms | S/A | W oracle ms | A/W oracle |
|---|---:|---:|---:|---:|---:|
| coins | 91.0668 | 184.6181 | 2.0273× | 129.7418 | 0.7019× |
| histology | 92.3508 | 174.2815 | 1.8872× | 135.0909 | 0.6836× |
| method | 99.7432 | 181.8589 | 1.8233× | 148.6910 | 0.6708× |

## Decision

Even the zero-contention stage-wavefront oracle cannot provide 1.10× median headroom over stock. Per the preregistered stop condition, W is not implemented or executed.

- Correctness DP repeatability A/S: True/True.
- Stock-vs-split logits: max abs 0.578125, minimum cosine 0.999360919, greedy token equal `True`.
- A and S intentionally have zero cross-stage CUDA overlap; no concurrent DeepEP collective is issued.
- W actual useful overlap: not measured because the Stage 0 gate did not pass.

## Mean per-layer stage timing

| Variant | Segment | Stage | Mean ms | Median ms |
|---|---|---|---:|---:|
| A | full | attention | 0.7092 | 0.6694 |
| A | full | combine | 0.1652 | 0.1303 |
| A | full | decoder_layer | 2.2768 | 2.2136 |
| A | full | dispatch | 0.4234 | 0.3636 |
| A | full | expert | 0.5576 | 0.4526 |
| A | full | moe_total | 1.5388 | 1.4744 |
| S | prefix | attention | 0.6731 | 0.6412 |
| S | prefix | combine | 0.1579 | 0.1181 |
| S | prefix | decoder_layer | 2.1699 | 2.0977 |
| S | prefix | dispatch | 0.3929 | 0.3432 |
| S | prefix | expert | 0.5456 | 0.4387 |
| S | prefix | moe_total | 1.4544 | 1.3519 |
| S | tail | attention | 0.6575 | 0.6410 |
| S | tail | combine | 0.1249 | 0.1080 |
| S | tail | decoder_layer | 2.1350 | 2.0971 |
| S | tail | dispatch | 0.3664 | 0.3437 |
| S | tail | expert | 0.4246 | 0.4134 |
| S | tail | moe_total | 1.3954 | 1.3545 |

## Execution invariants

- Physical GPUs exposed: 1,2,3,4 only.
- DBO configured: false on every rank and variant.
- S host owners: one per worker; S compute stream: the original single compute stream.
- Cross-stage CUDA concurrency in A/S: 0 by construction; W was gated.
- DeepEP collective overlap: 0; prefix and tail MoE calls are sequential.
- Attention metadata has two isolated scopes for S, without enabling vLLM DBO execution.
- Kernel/collective call counts are doubled by S (one prefix and one tail call per layer); this is the dominant measured split cost. The split is not a second DBO execution.
- Actual CUDA kernel-concurrency tracing was not started after the preregistered gate failed; A/S cross-stage useful overlap is therefore exactly 0 by construction, and W is `NOT-RUN`.
- Control file reads: 42 per rank at wave boundaries, 0 inside model forward. The cached control object is used for the hot path.
- Result directory: `/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/non_dbo_causal_wavefront_20260826_stage0_v3`.
