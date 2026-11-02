# Atomic Prefill PD: Final Independent Capacity

Compiled at 2026-11-02T01:28:43.086700+00:00. Elapsed time: 21.426 hours.

Branch: `flashvep/looped-pd-atomic-prefill-qps-crossover`.

Compilation source commit: `147c4dc48215dd82cc85f77b044ff27325ea84fc`.

Each measured run records its own source archive, source commit, and configuration hash.
The final publication commit can differ from the measurement commit.
Use `git log -1 --format=%H` on this branch to obtain the publication SHA.

Actual analysis-tree SHA256: `a392ed79e603c437d00c3575d39166f104f681e2144fe757f6ef0524e7bc5c18`.

The compilation HEAD can precede uncommitted analysis changes.
See `state/analysis_source_manifest.json` for the exact file hashes.

Evidence type: MEASURED_FINITE_INDEPENDENT_SERVICE.
These synchronized host service boundaries are not pure GPU kernel durations.
All-ready D cohorts are optimistic relative to staggered online readiness.

## P service by chunk


| Model | Input | C | Selected P slots | Finite requests/s |
| --- | --- | --- | --- | --- |
| qwen8 | 3900 | 3900 | 1 | 10.3589 |
| qwen8 | 8000 | 256 | 4 | 4.0563 |
| qwen8 | 8000 | 1024 | 2 | 4.2816 |
| qwen8 | 8000 | 8000 | 1 | 4.5732 |
| ouro | 4096 | 4096 | 4 | 4.5748 |
| ouro | 8192 | 256 | 8 | 1.4130 |
| ouro | 8192 | 1024 | 8 | 2.0999 |
| ouro | 8192 | 2048 | 4 | 2.0821 |
| ouro | 8192 | 4096 | 4 | 2.1905 |
| ouro | 8192 | 8192 | 4 | 2.1331 |
| ouro | 16384 | 256 | 4 | 0.4864 |
| ouro | 16384 | 1024 | 4 | 0.8993 |
| ouro | 16384 | 2048 | 4 | 0.9798 |
| ouro | 16384 | 4096 | 2 | 0.9676 |
| ouro | 16384 | 16384 | 1 | 0.9970 |
| huginn | 3500 | 256 | 3 | 0.8316 |
| huginn | 3500 | 1024 | 1 | 0.8257 |
| huginn | 3500 | 2048 | 1 | 0.8473 |
| huginn | 3500 | 3500 | 1 | 0.8231 |
| huginn | 3900 | 256 | 2 | 0.6992 |
| huginn | 3900 | 1024 | 1 | 0.7536 |
| huginn | 3900 | 2048 | 1 | 0.7556 |
| huginn | 3900 | 3900 | 1 | 0.7362 |

Each selected P concurrency is the smallest tested batch within 95% of that C setting's best rate.
The strongest C remains a mandatory tuning control.

## Local prepared D


| Model | Input | Output | D batch | Finite requests/s | Strongest P / D |
| --- | --- | --- | --- | --- | --- |
| qwen8 | 3900 | 32 | 16 | 38.0846 | 0.2720 |
| qwen8 | 3900 | 64 | 16 | 18.2469 | 0.5677 |
| qwen8 | 3900 | 100 | 16 | 11.6238 | 0.8912 |
| qwen8 | 8000 | 100 | 32 | 13.4841 | 0.3392 |
| qwen8 | 8000 | 200 | 32 | 6.6554 | 0.6871 |
| ouro | 4096 | 64 | 8 | 4.0653 | 1.1253 |
| ouro | 4096 | 100 | 8 | 2.5845 | 1.7701 |
| ouro | 8192 | 32 | 8 | 6.4635 | 0.3389 |
| ouro | 8192 | 64 | 8 | 3.1815 | 0.6885 |
| ouro | 8192 | 100 | 8 | 2.0276 | 1.0803 |
| ouro | 8192 | 200 | 8 | 1.0059 | 2.1776 |
| ouro | 16384 | 100 | 5 | 1.0513 | 0.9483 |
| huginn | 3500 | 32 | 3 | 0.8864 | 0.9559 |
| huginn | 3500 | 64 | 3 | 0.4360 | 1.9433 |
| huginn | 3500 | 100 | 3 | 0.2772 | 3.0561 |
| huginn | 3900 | 32 | 3 | 0.8664 | 0.8721 |
| huginn | 3900 | 64 | 3 | 0.4268 | 1.7705 |
| huginn | 3900 | 100 | 3 | 0.2716 | 2.7824 |

RLT P timing excludes first Coda sampling.
RLT prepared D includes first Coda sampling.
Native AR P includes first sampling.

Its prepared D excludes preparation and first sampling.
These boundaries are not additive E2E components.

## Actual native prepared D


| Model | Input/output | Execution family | Effective async | Graph rows | Batch | D budget | D KV blocks | Independent starts | Requests/s | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | 3500/32 | sync_g4 | False | 4 | 4 | 128 | 28288 | 3 | 0.9977 | [0.9809755469454421,1.0085054922513141] |
| ouro | 16384/32 | P2_L16384_O32 | True | 8 | 4 | 128 | 25088 | 3 | 2.2773 | [2.2574822377054202,2.303679747599995] |
| ouro | 8192/32 | ORIGINAL | True | 8 | 8 | 128 | 25088 | 3 | 4.8017 | [4.5756507319094855,5.052604029863785] |
| huginn | 3900/32 | sync_l3900_g4 | False | 4 | 3 | 128 | 24576 | 3 | 0.7748 | [0.7661993461172523,0.7805905884970411] |
| huginn | 3500/32 | sync_g4 | False | 4 | 3 | 128 | 28288 | 3 | 0.7078 | [0.648876169431893,0.7509192510511276] |
| huginn | 3900/32 | IGNORED_ASYNC_REQUEST_async_l3900_g4 | False | 4 | 3 | 128 | 24576 | 3 | 0.7785 | [0.7721293318734116,0.7834571849515013] |
| huginn | 3500/32 | IGNORED_ASYNC_REQUEST_async_g4 | False | 4 | 3 | 128 | 28288 | 2 | 0.7727 | [0.7698062232341656,0.7756587030006564] |
| huginn | 3500/32 | ORIGINAL | False | 8 | 3 | 128 | 28288 | 3 | 0.7539 | [0.7336999663417688,0.7910496826101309] |
| huginn | 3500/32 | IGNORED_ASYNC_REQUEST_async_g4 | False | 4 | 4 | 128 | 28288 | 1 | 0.9933 | [null,null] |

These timers start after actual native NIXL import and valid-state synchronization.
They include first Coda and the native coordinator.
Three starts remain a small inferential sample.

All-ready native service is still not sustainable online capacity.

## Actual P-only device activity


| Model | Input | Batch | Within-start repeats | Clean P service mean ms | Profiled P kernel fraction mean | Matched profile pairs within 5% |
| --- | --- | --- | --- | --- | --- | --- |
| huginn | 3500 | 1 | 3 | 1247.5405 | 0.8198 | 3 |
| huginn | 3500 | 2 | 3 | 2499.4806 | 0.7842 | 3 |
| huginn | 3500 | 3 | 3 | 3680.9474 | 0.8141 | 3 |
| ouro | 8192 | 1 | 3 | 501.8168 | 0.6627 | 0 |
| ouro | 8192 | 4 | 3 | 1884.5464 | 0.7597 | 3 |
| ouro | 8192 | 8 | 3 | 3681.1340 | 0.7783 | 3 |
| ouro | 16384 | 1 | 3 | 1014.4710 | 0.7990 | 3 |
| ouro | 16384 | 2 | 3 | 2056.4719 | 0.7765 | 3 |
| ouro | 16384 | 4 | 3 | 4080.6032 | 0.7791 | 3 |

These local P cohorts have no D reservation or NIXL progress dependency.
The table separates synchronized host service from recorded CUDA kernel activity.
A non-kernel gap is not automatically CPU time or GPU idle.

This diagnostic does not prove a universal P compute ceiling or native Serving capacity.
See `p_duty_windows.csv` for individual exact-output and overhead checks.

## Reserved KV versus active Decode requests


| Run | Mean reserved D contexts | Mean pending D contexts | Mean active D contexts | Pending share of reserved context-time |
| --- | --- | --- | --- | --- |
| ext_final_huginn_l3500_c2048_r1_s0_v1 | 2.8549 | 0.9984 | 1.8565 | 0.3497 |
| ext_final_huginn_l3500_cfull_r1_s0_v1 | 3.9118 | 0.9965 | 2.9153 | 0.2547 |
| ext_final_ouro_l16384_c4096_r1_s0_v1 | 3.8644 | 2.9088 | 0.9556 | 0.7527 |
| ext_final_ouro_l16384_cfull_r1_s0_v1 | 3.7129 | 2.9970 | 0.7159 | 0.8072 |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s0_v1 | 11.2438 | 10.0937 | 1.1501 | 0.8977 |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s0_v1 | 11.5175 | 9.2875 | 2.2300 | 0.8064 |

These counts use native coordinator timestamps in the central arrival window.
A pending reservation holds D KV before D activation.
The pending interval can include P computation, transfer, and D admission.

Reserved context-time equals pending context-time plus active context-time.
These counts are not simultaneous GEMM rows, kernel busy time, or proof of GPU idle.
See `native_D_reservation_accounting.csv` for every completed native PD pilot and final run.

## KV accounting


| Model | Baseline KV bytes/GPU | Blocks | Allocated planes | Valid Attention planes |
| --- | --- | --- | --- | --- |
| Ouro R4 | 70866960384 | 22528 | 96 | 96 |
| Huginn R32 | 66437775360 | 24576 | 256 | 132 |

Evidence type: MEASURED_ALLOCATION plus STRUCTURAL_ACCOUNTING.
Huginn has 124 unused storage planes, or 48.4375% of the uniform allocation.
This is not a measured speedup from compact KV storage.

Actual native transfer uses valid state, not every reserved storage plane.

| Model | Physical Attention layers | Valid logical planes | Allocated logical planes | KV heads | Head dimension | Full-valid bytes/token | Uniform reserved bytes/token | Valid density / Qwen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | 24 | 96 | 96 | 16 | 128 | 786432 | 786432 | 5.3333 |
| huginn | 8 | 132 | 256 | 55 | 96 | 2787840 | 5406720 | 18.9062 |
| qwen8 | 36 | 36 | 36 | 8 | 128 | 147456 | 147456 | 1.0000 |

These equations use the pinned checkpoint configuration and observed runtime layout.
They count K and V at BF16. They exclude block padding, metadata, temporary buffers, and incomplete valid frontiers.
The Qwen checkpoint has 36 Attention layers and eight KV heads.

Ouro has 96 logical Attention planes and 16 KV heads.
Huginn has 132 valid logical planes and 55 KV heads. Its uniform runtime reserves 256 planes.
Both recurrent depth and KV-head width change this comparison. Neither factor is isolated here.

See `model_kv_density_accounting.csv` and `kv_pool_allocation_identity.csv` for equations and allocation checks.

| Model/input | Measured bytes/request | Fragment completions/request |
| --- | --- | --- |
| Ouro/8192 | 6442455040 | 97 |
| Huginn/3900 | 10872554880 | 163 |
| Huginn/3500 | 9757418880 | 146 |

The fourth Huginn 3500/32 reservation requires 28,288 blocks.
The larger measured KV pool occupies 76,472,647,680 bytes.
Its D128 memory probe peaks at 83,746,289,664 allocated bytes.

This tight allocation does not guarantee arbitrary Prefill-continuation workspace.

All batches and blocked memory cases remain in `capacity_samples.csv` and `capacity_summary.csv`.
