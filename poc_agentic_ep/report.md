# Agentic serving × physical MoE EP: cheap discovery

**Verdict: NO-GO**. No runtime migration, learned predictor, scheduler or new kernel was implemented.

There is statistically detectable cross-turn locality after shared-prefix removal, but it does not yield an advantage over coding-domain global popularity. This is **not** a claim that adjacent turns have no locality or that all agentic EP opportunities are impossible. The tested previous-turn preparation premise fails its decisive generic-control comparison.

## Protocol and validity

Measured Qwen3-30B-A3B revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, BF16,48 layers/d2048/m768/E128/K8. TRUE TP2-DP2-EP4-PP1 /DeepEP HT/Triton sparse experts,linear32 ownership verified from loaded runtime tensors. Only physical2,3,6,7 (logical0,1,2,3), H10080GB/NV18. TP2/DP2 is the existing supported family, not spec-recommended optional TP1/DP1 rewritten into a new topology. One active user request on DP0; DP1 dummy collectives are not user routes.

SWE-smith XML-format public action/observation histories preserve semantic content, reconstructed with the frozen native Qwen3 template (thinking=False). Source revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`. Metadata-only seeded selection from100 public rows;12 instance-disjoint trajectories,9 turns each,4 calibration +8 held-out,64 held-out adjacent pairs. Outcome/resolution/patch/routing never selected samples. Input context is preserved, not cropped; maximum16384 tokens. No live tools or dataset commands were executed.

Source: [pinned SWE-smith dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/08e109b4a59eaeebf80e4675cd125d42e7ac99a4), [official SWE-agent trajectory semantics](https://swe-agent.com/latest/usage/trajectories/). This is replay under Qwen3, not a claim about agent task success.

**Prefix caching is actually enabled.** Primary routing retains only positions after both the actual KV-cache query start and the frozen exclusion frontier. The frontier is max(exact previous-prompt LCP, previous completed-context LCP, current-rendering new observation boundary). Native Qwen3 may rewrite prior assistant empty-think formatting, so plain LCP alone can accidentally count repeated assistant content. A sentinel rendered before routing gives the actual new observation span, conservatively excluding all historical semantic messages. Originalv1 smoke/cohort are preserved; v2 changes only exclusion metadata, not selected instances/input IDs/answers. Two calibration smoke turns were seen before this repair; no held-out routes were inspected. All new-only raw tensors pass position/coverage/owner/path-count checks.

Primary=new prefill suffix. Secondary=first32 recorded assistant tokens teacher-forced in a causal packed forward; not natural AR output, not decode timing, not future features. The native one-token sample after this suffix is discarded. Global prior uses only four fully completed calibration trajectories; previous-turn predictor uses only the prior completed turn. Oracle and shuffled future diagnostic controls are explicitly nondeployable.

All confidence intervals resample8 entire held-out trajectories (2000 draws). Layer events and64 pairs are not independent samples. Calibration uncertainty is not included in these conditional intervals. Controls and practical thresholds are frozen before held-out analysis; the first-turn exclusion sensitivity is separately recorded.

## Direct answers and primary held-out results

| Comparison | Expert cosine delta,95% CI | Coverage@16 delta pp,95% CI |
|---|---:|---:|
| Adjacent/previous minus same_shuffled | 0.068 [0.047, 0.091] | 4.163 [2.790, 5.720] |
| Adjacent/previous minus cross_matched | 0.042 [0.010, 0.073] | 2.321 [0.316, 4.367] |
| Adjacent/previous minus global | -0.062 [-0.085, -0.040] | -3.712 [-5.113, -2.359] |

1. **Adjacent agent turns routing-local?** Yes, both shuffled/matched expert-cosine contrast CIs exclude zero. The frozen practical Gate1=False: matched-control point gain0.042 is below0.05, and coverage gain2.321pp is below5pp. Statistical locality is not absent.
2. **Survives shared-prefix removal?** Only new observation suffix routes enter the primary contrasts. Full accumulated-prompt similarity is never used.
3. **Stronger than shuffled/matched?** Statistically yes against both expert-cosine controls, but the matched-control increment falls below the frozen practical threshold. Gate1=False; do not confuse that practical gate failure with an absence of locality.
4. **Previous beats global popularity?** Coverage@16 delta=-3.712 [-5.113, -2.359]pp. Gate2=False.
   The predeclared exclude-initial-predictor sensitivity (56 pairs) remains negative: -1.592pp,95% CI[-3.147,-0.093]. The main failure does not depend solely on initial task/instruction turns.
5. **Actual EP4 rank-level manifestation?** See rank deltas below, on actual contiguous32 ownership. Hottest-rank constancy is not automatically useful signal.

| Adjacent minus control | Rank cosine delta | Rank L1 delta (lower better) | Hottest-rank persistence delta pp |
|---|---:|---:|---:|
| same_shuffled | 0.006 [0.003, 0.008] | -0.026 [-0.037, -0.018] | 6.490 [4.463, 8.727] |
| cross_matched | 0.004 [0.001, 0.006] | -0.017 [-0.026, -0.006] | 3.006 [-0.043, 5.675] |
| global | -0.007 [-0.009, -0.005] | 0.025 [0.018, 0.033] | -2.214 [-4.264, -0.033] |

## Structural oracle preparation and tool-gap feasibility

Fresh loaded tensor arithmetic: **9,437,184 bytes/expert/layer** (9.000MiB), w1[32,1536,2048]+w2[32,2048,768]. Every replica budget counts all48 layers. One extra expert/layer/rank costs 0.421875GiB/rank, not merely9MiB for the entire model.

Replicas are modeled at the source rank; original expert IDs/weights/routes remain semantically unchanged. All policies use the same top-frequency selection rule. Oracle uses actual next-turn per-source counts and exactly maximizes removed assignments under independent slots, but does not prove a globally byte-optimal destination-closure solution. Useful communication bytes are deduplicated per token/destination. Removing one remote expert path does not eliminate a hidden-row transfer if another path still needs that destination.

Only active source ranks prepare positive-score replicas; idle DP sources do not invent traffic or useful replicas. Capacity is per rank, and actual total prepared bytes/counts/edge traffic are saved. Feasibility checks loaded-model+8GiB-KV free memory. Extra compute/load may move onto the source and worsen rank imbalance; those quantities are reported, not assumed improved.

| Slots/layer/rank | Capacity GiB/rank | Global remote bytes reduction % | Previous reduction % | Oracle reduction % | Previous/oracle recovery % | Previous serial-copy estimate ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.422 | 1.568 [1.470, 1.673] | 1.441 [1.340, 1.546] | 2.577 [2.466, 2.696] | 57.428 [52.992, 61.985] | 3.719 [3.625, 3.814] |
| 2 | 0.844 | 3.621 [3.449, 3.787] | 3.158 [2.941, 3.401] | 5.822 [5.683, 5.978] | 56.685 [52.919, 60.586] | 7.438 [7.251, 7.627] |
| 4 | 1.688 | 7.402 [7.152, 7.656] | 6.550 [6.058, 7.062] | 12.513 [12.059, 13.010] | 55.607 [52.126, 59.099] | 14.879 [14.505, 15.254] |
| 9 | 3.797 | 17.707 [17.188, 18.242] | 15.834 [14.716, 17.051] | 29.866 [28.688, 31.101] | 56.220 [52.976, 59.808] | 33.490 [32.651, 34.330] |

6. **Oracle EP headroom?** Gate3=True; table shows feasible next-truth replication-policy logical byte reductions, not measured latency gains.
7. **Previous-turn recovers what fraction?** Recovery is computed request-wise vs oracle logical byte reduction, then trajectory-bootstrapped. Gate4=True. Global comparison is mandatory even if recovery is high.
8. **Required expert memory?** Table counts full48-layer capacity; preparation_budget.csv also saves actual total/per-rank bytes. 0.5/1/2/4GiB maps to slots {"0.5": 1, "1": 2, "2": 4, "4": 9}.
9. **Can tool gap hide preparation?** Gate5=True. Actual12-direction torch P2P benchmark has3 warmups+10 repeats at9MiB and64MiB. The64MiB effective peer-copy service bandwidth spans **346.64–361.20GB/s**, not a pure transport-only bandwidth measurement. Serial-copy and bulk-serial extrapolations are feasibility estimates only; no concurrent replication/packing/allocator/admission overhead is measured. 50/100/250/500/1000ms status is saved per policy/request/budget. These are hypothetical gaps, not measured SWE-agent wall gaps.
10. **Overall verdict? NO-GO.** No runtime migration/scheduler. Close this tested agent-locality premise if G1/G2 fail; do not rescue with learned predictors.

At4 replica slots, previous removes17.95% remote expert paths but only6.55% deduplicated remote hidden-row bytes; global removes7.40% bytes and oracle12.51%. At9 slots, previous remains worse than global by-1.873pp byte reduction (95% CI[-2.973,-0.792]). The next-truth replica oracle also increases max-rank assignment load by16.62% at4 slots and41.33% at9 slots. Moving work toward the two active source ranks can create a compute hotspot even while communication shrinks; this is not net critical-path/E2E headroom. Thus high recovered-oracle percentage does not establish an agent-specific advantage.

The replica simulation transfers prior per-source expert frequencies, while the expert-locality/global-coverage comparison aggregates source ranks. Source-SP partition changes across cached turns are a limitation of the preparation model, not the sole cause of its verdict: source-aggregated previous-turn coverage is already below global popularity. The cohort is instance-disjoint, not guaranteed repository-disjoint; no cross-repository or production-agent generalization is claimed.

## Evidence boundaries and reproduction

Routing and P2P copy service are measured facts. Replica routing/byte/load gains are structural simulation. Preparation time is a measured-copy-based estimate. Expert/MoE trace timings are diagnostic instrumentation-only, with one cumulative-critical rank per invocation. They are not a validated latency predictor. No E2E/TPOT/real-migration speedup is claimed.

After the decisive global-control failure, the requested budget/tool-gap questions were completed only as cheap secondary characterization using already captured traces and Stage0 P2P measurements. They do not rescue the failed gate or authorize runtime promotion. Five automated validity tests pass; the offline audit verifies140 evidence hashes,108 semantic traces,10368 layer/phase count records and1536 frozen policy decisions. The decisive global comparison and trajectory-bootstrap CI reproduce from published evidence without a GPU.

Raw retained traces store top8 IDs, original FP32 routing weights, true source ranks and original token positions for new-prefill / teacher-forced decode only. No model weights or hidden states are uploaded. Full local failed/smoke/runtime logs remain append-only; compact semantic event/ownership/cache evidence is exported separately. Original dirty worktree is unchanged.

Reproduction commands are in README.md; tests cover prefix identity, phase mapping, destination dedup, memory arithmetic, idle ranks and trajectory bootstrap. No full benchmark quality rerun, new kernel, learned predictor, production scheduler, quantization or real dynamic migration.

```text
VERDICT: NO-GO
GATES: {'G1_LOCALITY': False, 'G2_PREVIOUS_OVER_GLOBAL': False, 'G3_STRUCTURAL_ORACLE_HEADROOM': True, 'G4_CAUSAL_RECOVERY': True, 'G5_TOOL_GAP_ESTIMATE': True}
GPU_MASK: 2,3,6,7
TOPOLOGY: TRUE TP2-DP2-EP4-PP1
BRANCH: flashvep/agentic-moe-ep-locality-poc
COMMIT: git log -1 --format=%H on this evidence branch; exact SHA in final handoff
MEASURED_RUNTIME_SPEEDUP: NOT_RUN
NEXT_ACTION: No runtime migration/scheduler. Close this tested agent-locality premise if G1/G2 fail; do not rescue with learned predictors.
```
