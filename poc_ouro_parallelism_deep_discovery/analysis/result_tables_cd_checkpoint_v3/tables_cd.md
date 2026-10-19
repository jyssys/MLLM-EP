# Current result panels: specification Tables C and D

This is an active-campaign synthesis. It is not the final campaign verdict.
The tables preserve separate evidence lanes. They do not certify the campaign duration.

## Table C1: original natural-generation calibration

These are actual greedy natural outputs from the original8 requests per domain.
Each runtime and TP setting remains a separate operating point.
No row supplies candidate quality or a matched architecture speedup.
Request-level95% bootstrap intervals describe this small calibration sample.
They do not establish1pp or2pp noninferiority. The original held-out remains untouched.

| Model / mode | Runtime | GPUs | Domain | Correct/N | Accuracy,95% CI | Parse failures | Truncated |
|---|---|---:|---|---:|---|---:|---:|
| Base1.4B / adaptive | staged_reference | 1 | gsm8k | 5/8 | 62.500%,[25.000,87.500]% | 0 | 0 |
| Base1.4B / adaptive | staged_reference | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 0 | 0 |
| Base1.4B / fixed | native_vllm | 1 | gsm8k | 3/8 | 37.500%,[12.500,75.000]% | 1 | 0 |
| Base1.4B / fixed | native_vllm | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 0 | 0 |
| Base1.4B / fixed | native_vllm_async | 1 | gsm8k | 3/8 | 37.500%,[12.500,75.000]% | 1 | 0 |
| Base1.4B / fixed | native_vllm_async | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 0 | 0 |
| Base1.4B / fixed | native_vllm_async_tp2baseline | 2 | gsm8k | 4/8 | 50.000%,[12.500,87.500]% | 0 | 0 |
| Base1.4B / fixed | native_vllm_async_tp2baseline | 2 | math500 | 6/8 | 75.000%,[37.500,100.000]% | 0 | 0 |
| Base1.4B / fixed | native_vllm_async_tp4baseline | 4 | gsm8k | 4/8 | 50.000%,[12.500,87.500]% | 0 | 0 |
| Base1.4B / fixed | native_vllm_async_tp4baseline | 4 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 2 | 2 |
| Base1.4B / fixed | staged_reference | 1 | gsm8k | 4/8 | 50.000%,[12.500,87.500]% | 0 | 0 |
| Base1.4B / fixed | staged_reference | 1 | math500 | 3/8 | 37.500%,[12.500,75.000]% | 1 | 1 |
| Base2.6B / adaptive | staged_reference | 1 | gsm8k | 6/8 | 75.000%,[50.000,100.000]% | 0 | 0 |
| Base2.6B / adaptive | staged_reference | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 3 | 3 |
| Base2.6B / fixed | native_vllm | 1 | gsm8k | 7/8 | 87.500%,[62.500,100.000]% | 0 | 0 |
| Base2.6B / fixed | native_vllm | 1 | math500 | 5/8 | 62.500%,[25.000,100.000]% | 2 | 3 |
| Base2.6B / fixed | native_vllm_async | 1 | gsm8k | 7/8 | 87.500%,[62.500,100.000]% | 0 | 0 |
| Base2.6B / fixed | native_vllm_async | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 2 | 3 |
| Base2.6B / fixed | native_vllm_async_recheck_on | 1 | gsm8k | 7/8 | 87.500%,[62.500,100.000]% | 0 | 0 |
| Base2.6B / fixed | native_vllm_async_recheck_on | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 2 | 3 |
| Base2.6B / fixed | native_vllm_async_tp2baseline | 2 | gsm8k | 7/8 | 87.500%,[62.500,100.000]% | 0 | 0 |
| Base2.6B / fixed | native_vllm_async_tp2baseline | 2 | math500 | 5/8 | 62.500%,[25.000,100.000]% | 2 | 2 |
| Base2.6B / fixed | native_vllm_async_tp4baseline | 4 | gsm8k | 7/8 | 87.500%,[62.500,100.000]% | 0 | 0 |
| Base2.6B / fixed | native_vllm_async_tp4baseline | 4 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 3 | 4 |
| Base2.6B / fixed | native_vllm_recheck_off | 1 | gsm8k | 7/8 | 87.500%,[62.500,100.000]% | 0 | 0 |
| Base2.6B / fixed | native_vllm_recheck_off | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 2 | 3 |
| Base2.6B / fixed | staged_reference | 1 | gsm8k | 7/8 | 87.500%,[62.500,100.000]% | 0 | 0 |
| Base2.6B / fixed | staged_reference | 1 | math500 | 4/8 | 50.000%,[12.500,87.500]% | 3 | 4 |

The table retains backend/TP output differences and length truncation.
It does not promote partial teacher-forced agreement into natural-output parity.

## Table C2: disjoint baseline extension

The frozen extension excludes the original calibration and held-out IDs.
It uses32 new requests per domain and the unchanged original gate.
Only complete, independently audited arms enter this panel.
The extension is baseline evidence. It is not candidate held-out confirmation.

| Model / mode | Runtime | GPUs | Domain | Correct/N | Accuracy,95% CI | Parse failures | Truncated |
|---|---|---:|---|---:|---|---:|---:|
| Base1.4B / fixed | staged_reference, graph-enabled B1 | 1 | gsm8k | 24/32 | 75.000%,[59.375,90.625]% | 1 | 0 |
| Base1.4B / fixed | staged_reference, graph-enabled B1 | 1 | math500 | 17/32 | 53.125%,[37.500,68.750]% | 1 | 4 |
| Base1.4B / adaptive | staged_reference, graph-enabled B1 | 1 | gsm8k | 24/32 | 75.000%,[59.375,90.625]% | 2 | 0 |
| Base1.4B / adaptive | staged_reference, graph-enabled B1 | 1 | math500 | 21/32 | 65.625%,[50.000,81.250]% | 3 | 4 |

The extension does not pool its requests with the earlier8/domain cohort.
Adaptive-minus-fixed comparisons describe different original operating points, not a new mechanism.

## Table C3: controlled native hardware workload

These native calls execute real pretrained weights and real physical TP collectives.
Each request emits128 controlled constant tokens. EOS does not shorten this hardware workload.
Each cell reports the median of five trials. This is not task accuracy or natural reasoning latency.
TP rows use the native API wall. DP rows use the parent release-to-completion wall.
The two endpoints are not interchangeable. DP speedups against a TP endpoint remain NOT_RUN.
TP1 ratios divide matching cohort/context medians. Sequential jobs do not establish randomized paired uncertainty.
No candidate earns a same-P speedup from these baseline rows.

| Model | Method | GPUs | C | Prefix | Median wall,ms | Output TPS | TP1 ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Base1.4B | TP1 | 1 | 1 | 1024 | 1089.762 | 117.457 | 1.000× |
| Base1.4B | TP1 | 1 | 1 | 4096 | 1218.461 | 105.051 | 1.000× |
| Base1.4B | TP1 | 1 | 2 | 1024 | 1166.763 | 219.410 | 1.000× |
| Base1.4B | TP1 | 1 | 2 | 4096 | 1449.465 | 176.617 | 1.000× |
| Base1.4B | TP1 | 1 | 4 | 1024 | 1265.089 | 404.715 | 1.000× |
| Base1.4B | TP1 | 1 | 4 | 4096 | 1907.486 | 268.416 | 1.000× |
| Base1.4B | TP1 | 1 | 8 | 1024 | 1493.105 | 685.819 | 1.000× |
| Base1.4B | TP1 | 1 | 8 | 4096 | 2781.507 | 368.146 | 1.000× |
| Base1.4B | TP2 | 2 | 1 | 1024 | 929.606 | 137.693 | 1.172× |
| Base1.4B | TP2 | 2 | 1 | 4096 | 1011.082 | 126.597 | 1.205× |
| Base1.4B | TP2 | 2 | 2 | 1024 | 1001.474 | 255.623 | 1.165× |
| Base1.4B | TP2 | 2 | 2 | 4096 | 1166.614 | 219.439 | 1.242× |
| Base1.4B | TP2 | 2 | 4 | 1024 | 1054.849 | 485.377 | 1.199× |
| Base1.4B | TP2 | 2 | 4 | 4096 | 1426.702 | 358.870 | 1.337× |
| Base1.4B | TP2 | 2 | 8 | 1024 | 1200.884 | 852.705 | 1.243× |
| Base1.4B | TP2 | 2 | 8 | 4096 | 1924.296 | 532.143 | 1.445× |
| Base1.4B | TP4 | 4 | 1 | 1024 | 813.429 | 157.358 | 1.340× |
| Base1.4B | TP4 | 4 | 1 | 4096 | 863.745 | 148.192 | 1.411× |
| Base1.4B | TP4 | 4 | 2 | 1024 | 899.904 | 284.475 | 1.297× |
| Base1.4B | TP4 | 4 | 2 | 4096 | 993.632 | 257.641 | 1.459× |
| Base1.4B | TP4 | 4 | 4 | 1024 | 945.336 | 541.606 | 1.338× |
| Base1.4B | TP4 | 4 | 4 | 4096 | 1144.338 | 447.420 | 1.667× |
| Base1.4B | TP4 | 4 | 8 | 1024 | 1014.667 | 1009.198 | 1.472× |
| Base1.4B | TP4 | 4 | 8 | 4096 | 1450.209 | 706.105 | 1.918× |
| Base1.4B | DP4 | 4 | 1 | 1024 | 1046.912 | 122.264 | NOT_RUN |
| Base1.4B | DP4 | 4 | 1 | 4096 | 1214.872 | 105.361 | NOT_RUN |
| Base1.4B | DP4 | 4 | 2 | 1024 | 1046.604 | 244.601 | NOT_RUN |
| Base1.4B | DP4 | 4 | 2 | 4096 | 1213.399 | 210.978 | NOT_RUN |
| Base1.4B | DP4 | 4 | 4 | 1024 | 1091.817 | 468.943 | NOT_RUN |
| Base1.4B | DP4 | 4 | 4 | 4096 | 1216.290 | 420.952 | NOT_RUN |
| Base1.4B | DP4 | 4 | 8 | 1024 | 1206.251 | 848.911 | NOT_RUN |
| Base1.4B | DP4 | 4 | 8 | 4096 | 1446.820 | 707.759 | NOT_RUN |
| Base2.6B | TP1 | 1 | 1 | 1024 | 2062.765 | 62.053 | 1.000× |
| Base2.6B | TP1 | 1 | 1 | 4096 | 2403.272 | 53.261 | 1.000× |
| Base2.6B | TP1 | 1 | 2 | 1024 | 2220.936 | 115.267 | 1.000× |
| Base2.6B | TP1 | 1 | 2 | 4096 | 2867.936 | 89.263 | 1.000× |
| Base2.6B | TP1 | 1 | 4 | 1024 | 2469.343 | 207.343 | 1.000× |
| Base2.6B | TP1 | 1 | 4 | 4096 | 3788.884 | 135.132 | 1.000× |
| Base2.6B | TP1 | 1 | 8 | 1024 | 2904.009 | 352.616 | 1.000× |
| Base2.6B | TP1 | 1 | 8 | 4096 | 5562.196 | 184.100 | 1.000× |
| Base2.6B | TP2 | 2 | 1 | 1024 | 1826.399 | 70.083 | 1.129× |
| Base2.6B | TP2 | 2 | 1 | 4096 | 1990.852 | 64.294 | 1.207× |
| Base2.6B | TP2 | 2 | 2 | 1024 | 1964.820 | 130.292 | 1.130× |
| Base2.6B | TP2 | 2 | 2 | 4096 | 2298.624 | 111.371 | 1.248× |
| Base2.6B | TP2 | 2 | 4 | 1024 | 2074.718 | 246.780 | 1.190× |
| Base2.6B | TP2 | 2 | 4 | 4096 | 2818.433 | 181.661 | 1.344× |
| Base2.6B | TP2 | 2 | 8 | 1024 | 2369.124 | 432.227 | 1.226× |
| Base2.6B | TP2 | 2 | 8 | 4096 | 3826.569 | 267.603 | 1.454× |
| Base2.6B | TP4 | 4 | 1 | 1024 | 1628.599 | 78.595 | 1.267× |
| Base2.6B | TP4 | 4 | 1 | 4096 | 1718.983 | 74.463 | 1.398× |
| Base2.6B | TP4 | 4 | 2 | 1024 | 1797.888 | 142.389 | 1.235× |
| Base2.6B | TP4 | 4 | 2 | 4096 | 1984.755 | 128.983 | 1.445× |
| Base2.6B | TP4 | 4 | 4 | 1024 | 1893.501 | 270.399 | 1.304× |
| Base2.6B | TP4 | 4 | 4 | 4096 | 2261.567 | 226.392 | 1.675× |
| Base2.6B | TP4 | 4 | 8 | 1024 | 1999.824 | 512.045 | 1.452× |
| Base2.6B | TP4 | 4 | 8 | 4096 | 2877.707 | 355.839 | 1.933× |
| Base2.6B | DP4 | 4 | 1 | 1024 | 2097.563 | 61.023 | NOT_RUN |
| Base2.6B | DP4 | 4 | 1 | 4096 | 2398.698 | 53.362 | NOT_RUN |
| Base2.6B | DP4 | 4 | 2 | 1024 | 2097.841 | 122.030 | NOT_RUN |
| Base2.6B | DP4 | 4 | 2 | 4096 | 2397.719 | 106.768 | NOT_RUN |
| Base2.6B | DP4 | 4 | 4 | 1024 | 2120.216 | 241.485 | NOT_RUN |
| Base2.6B | DP4 | 4 | 4 | 4096 | 2401.673 | 213.185 | NOT_RUN |
| Base2.6B | DP4 | 4 | 8 | 1024 | 2304.716 | 444.306 | NOT_RUN |
| Base2.6B | DP4 | 4 | 8 | 4096 | 2868.595 | 356.969 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 1 | 1024 | 994.266 | 128.738 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 1 | 4096 | 1073.953 | 119.186 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 2 | 1024 | 968.941 | 264.206 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 2 | 4096 | 1121.203 | 228.326 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 4 | 1024 | 1040.467 | 492.087 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 4 | 4096 | 1208.624 | 423.622 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 8 | 1024 | 1134.365 | 902.708 | NOT_RUN |
| Base1.4B | TP2_DP2 | 4 | 8 | 4096 | 1445.441 | 708.434 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 1 | 1024 | 1871.042 | 68.411 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 1 | 4096 | 2030.435 | 63.041 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 2 | 1024 | 1884.316 | 135.858 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 2 | 4096 | 2019.527 | 126.762 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 4 | 1024 | 2035.081 | 251.587 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 4 | 4096 | 2304.985 | 222.127 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 8 | 1024 | 2166.237 | 472.709 | NOT_RUN |
| Base2.6B | TP2_DP2 | 4 | 8 | 4096 | 2903.111 | 352.725 | NOT_RUN |

All four-GPU configurations reserve208GiB aggregate KV,52GiB per physical rank.
TP4, TP2×DP2, and DP4 differ in weight multiplicity and KV ownership.
TP1 and TP2 use fewer physical ranks. These reservations do not measure HBM reads.
The separate32K native protocol reserves52GiB aggregate KV. Its rows must not be merged here.

## Table C4: candidate performance and quality

| Candidate set | Both Base sizes | Fixed / adaptive | GSM8K / MATH | Live architecture latency | Both speedups |
|---|---|---|---|---|---|
| C01–C19 | Target scope only | Separate unimplemented paths | NOT_RUN | NOT_RUN | NOT_RUN |

Component probes, numerical diagnostics, and source results do not fill these cells.

## Table D: charged headroom and missing measurements

UNRESOLVED means that no valid complete critical-path value exists.
A scoped premise failure is not a universal physical impossibility claim.

| Candidate | Baseline exposed time | Removed/overlapped cost | Added compute | Added communication | Added synchronization/state | Critical-path estimate | Live validation/error |
|---|---|---|---|---|---|---|---|
| C01 | UNRESOLVED | Proposed reuse of a ready operator frontier | All useful rows remain mandatory | Owner dispatch and return remain charged | Packet readiness, gate and KV publication | No distinct complete model | NOT_RUN. Generic ready unions are legal |
| C02 | Isolated MLP component only:local≈0.037ms,TP≈0.027–0.030ms at one row | Proposed local attention with remote full MLP | One complete MLP at the owner | Both measured directions:remote total≈0.062–0.067ms | Source waits for the returned down projection | Tested round trip loses at1–512 rows. Separate AG/RS hybrid is unmeasured | Component BF16 words match. No model E2E. TP component uses NCCL,not native custom AR |
| C03 | Terminal copy address fixtures only | Skipped-depth terminal copies under selected adaptive KV policy | Address lookup and publication | Any remote owner access remains charged | Terminal selection,reclamation and stale-reference prevention | No complete online estimate | Direct ragged-cache prior. No fixed-R4 terminal-copy saving |
| C04 | UNRESOLVED | Proposed layout conversion,bytes,startup or exposed reduction | Consumer-compatible nonlinear arithmetic remains mandatory | Deferred reductions/gathers remain charged | Residual,norm and gate layout boundaries | Specific partial-sum nonlinear rewrites fail. Other layouts are open | CPU algebra only. No complete native numerical or live layout result |
| C05 | Native TPOT is only a coarse model input | Hypothesized depth-chain span through parallel correction | Observed exact recovery:warm0 plus4 rounds or warm1 plus3 | Unmeasured and omitted in the optimistic screen | Validation,current-token KV repair,correction and allocation omitted | Explicit optimistic same-P model does not support tested exact recovery. Not a physical lower bound | Actual current-token numerical diagnostic. Distributed timing/free-running quality NOT_RUN |
| C06 | UNRESOLVED | Proposed retained base plus sparse or structured fresh work | Dense changed operands still require fresh contraction | Indices,changed values and exact base ownership | Accumulation order and exact delta representation | 2448 adjacent row pairs show no unchanged sampled input tile. No kernel-time estimate | Two prompts and three selected layers/model. No adaptive-decode or down-input coverage |
| C07 | Zero accepted service trials | Hypothesized useful verified branch span | Draft,verify and rejected ancestors remain charged | Branch-owner state,verification and commit | Rollback,private KV and gate-policy matching | UNRESOLVED after first P1 fixture parity failure | 1889/2048 output words differ. Cause unresolved. No branch E2E or architecture failure conclusion |
| C08 | 4096-byte P2 primitive:broadcast≈0.01035ms | Proposed exact encoded message reduction | Unfused transform≈0.00908ms before entropy codec | Compressed packet,metadata and inverse transform | Exact base lifetime and publication | Encoded size is not transfer latency. No full fused codec model | CPU exact-word coding and GPU transform only. No compressed serving collective |
| C09 | UNRESOLVED | Proposed early certified final decision | Certified bounds,verification and fallback | Any global decision/reduction | Actual final-hidden enclosure and original sampler | Toy certificates do not supply native hit rates or error bounds | No checkpoint useful-decision hits or live result |
| C10 | UNRESOLVED | Proposed avoided KV capacity or exposed movement | Exact restoration or rematerialization | Real missing-state transfers only | Historical hidden,RoPE,append and authoritative fallback | No named validated complete cost advantage | Generic restoration overlap. Known-address prefetch does not remove a future query dependency |
| C11 | No qualified removable native tail | Proposed coded straggler work or cheaper verification | Redundancy,decoder and prover remain charged | Parity data,checks and rank coordination | Native BF16 operation semantics | A naive BF16 checksum adaptation fails. Other proof systems remain open | CPU controls only. No native straggler or latency result |
| C12 | UNRESOLVED | Proposed exact or qualified query-summary work | Summary,query,certificate and fallback | Distributed summary/partial outputs | Fresh query,full KV lifetime and numerical envelope | A fixed-feature rank obstruction is not a universal nonlinear-index lower bound | Selected-key and toy positive controls. Native complete query algorithm NOT_RUN |
| C13 | INVALID_FOR_ATTRIBUTION | Proposed original-gate wait overlap | Exact gate and legally independent useful work | Publication and any original required data movement | Rounded hazards,selected hidden,terminal KV and original exits | All4 observer triplets fail frozen validity guards. No time share or Amdahl gain | Functional/depth/graph parity passes. That does not validate timing |
| C14 | Separate fixed-native128-output reference | Hypothesized faster prefill with later ownership transition | Pack/scatter and unchanged decode in this screen | Designed2.6B32K one-way cluster remote KV:36GiB | Paged layout,poison check,bitwise round trip and transition | F1/4 interpolation:7/8 modeled10% budgets negative. Sole positive2.138ms. Not a bound or measured PP | Transpose replay DEFERRED. No full candidate or adaptive result |
| C15 | UNRESOLVED | Proposed static projection-input storage saving | Position-correct RoPE/head contraction | Any distributed retained representation | Retained K,V,X and scratch | Static query absorption fails the position control. Valid retained-K variants lose the stated capacity premise | CPU mathematical control. No native latency or universal reassociation rejection |
| C16 | UNRESOLVED | Proposed changed ready-frontier ownership cost | Same useful rows under tested readiness assumptions | Packet,AG/RS,padding and owner returns | Self-consistent future online readiness and gate/KV semantics | Equal ready sets do not imply equal physical costs. Cross-owner cost remains open | Generic union/source controls. No zero-benefit physical proof |
| C17 | UNRESOLVED | Proposed component-ready norm/projection work | Deferred scaling,retention and complete fresh contractions | Required reductions and consumer publication | Native accumulation and tensor lifetime | Generic FlashNorm identity. Simple native BF16 word-exact rewrite fails | No native tail-certificate hits or retention cost measurement |
| C18 | UNRESOLVED | Proposed repeated exact transition computation | Read-set lookup,identity validation and side effects | Any shared result movement | History,position,gate,KV and sampler identity | No observed whole-row hit in bounded actual-input samples | Numerical identity screen only. Correct side effects remain an obligation,not a candidate PASS |
| C19 | No actual HBM traffic or exposed read interval | Proposed exact larger-history capacity/read saving | Complete decoder,append and metadata | Native TP has no remote historical-KV transfer to delete | Legal base/page lifetime and full K+V | CPU encoded capacity plus explicitly uncached read model. Not measured latency | 32K native succeeds in named reservation. Other capacity regimes remain open |

## Reproduction and scope

The structured summary records all input hashes and full timing endpoints.
The headroom supplement retains a source path for every candidate row.
The frozen card labels remain unchanged. Later scope corrections control the interpretation.
The CDB observer failed validity guards. It supplies no removable-time share or Amdahl percentage.
No actual HBM-counter result exists. No P8 hardware measurement exists.
The primary-claim re-audit controls source scope. Source-reported speedups remain separate from our measurements.
