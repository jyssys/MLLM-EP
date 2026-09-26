# Part 1 baseline foundation summary

## Outcome

The Qwen3-VL model contract, image pipeline, physical sparse EP4 path, full
model clean timing anchors, real-route EP4/EP8 structural replay, and BF16
1/2GPU semantic fallbacks were implemented and executed.

Two promotion requirements remain unresolved:

1. Full MMStar and ChartQA public reproduction was not run, so official-quality
   matching and paired EP non-inferiority are unproven.
2. The latency simulator failed held-out gates (median APE 15.705%, P90
   46.817%) on a cohort smaller than the required request split.

Accordingly, this is durable Part 1 infrastructure but only
`READY_FOR_PART2=DIAGNOSTIC_ONLY`. No resolution/top-k optimization was begun.

## Required status matrix

| Item | Verdict |
|---|---|
| MODEL_CONTRACT | PASS |
| IMAGE_PIPELINE | PASS |
| PUBLIC_REPRODUCTION | PARTIAL_PROTOCOL (full score NOT_RUN) |
| REFERENCE_VS_EP_QUALITY | UNRESOLVED |
| PHYSICAL_EP4_PROOF | PASS |
| VIRTUAL_SEMANTIC_PARITY | PARTIAL / quality unresolved |
| SIM_EP4_VALIDATION | FAIL (`SIM_UNVALIDATED`) |
| SIM_EP8 | UNVALIDATED_EXTRAPOLATION |
| RESOURCE_FALLBACK | TESTED_4_2_1 |
| READY_FOR_PART2 | DIAGNOSTIC_ONLY |

## Final Required Summary

```text
PHYSICAL_GPU_SET / AVAILABLE_GPU_COUNT: 4,5,6,7 / 4
MODEL_REVISION / ACTUAL_EXPERT_SHAPES: 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c / d=2048,m=768,E=128,k=8,48 layers, gate_up=[128,2048,1536], down=[128,768,2048]
ACTUAL_TP_DP_EP_PP_AND_SOURCE_MAPPING: TP2/DP2/EP4/PP1, SP1; EP=TPxDP; linear 32 experts/rank; TP/DP source shards preserved in trace
REFERENCE_RUNTIME / EP_BACKEND: vLLM 0.20.0+cu129 stock TP4 / DeepEP high-throughput + sparse Triton experts
OFFICIAL_GENERATION_LANE: do_sample=true,seed=3407,temp=.7,top_p=.8,top_k=20,repetition=1.0,presence=1.5,max_new_tokens=32768,EOS; executed 4-request run capped at 128 and labeled deviation
SAMPLING_TOP_K / EXPERT_TOP_K: 20 / 8
DATASET_PROTOCOL_AND_COUNTS: manual image-plumbing 12 paired deterministic + 4 bounded official; MMStar/ChartQA full public protocols NOT_RUN
MMSTAR_REPORTED_REFERENCE / LOCAL_REFERENCE / ACTUAL_EP4: 72.1 / NOT_RUN / NOT_RUN
CHARTQA_REPORTED_REFERENCE / LOCAL_REFERENCE / ACTUAL_EP4: 86.8 / NOT_RUN / NOT_RUN
OTHER_TASKS_AND_METRICS: RealWorldQA 73.7, MMMU 74.2, OCRBench 903 raw, DocVQA test 95.0 registry only; local NOT_RUN
PUBLIC_REPRODUCTION_STATUS: PARTIAL_PROTOCOL (reported registry and official sampler pinned; full anchors NOT_RUN)
EP_VS_REFERENCE_PAIRED_DELTA_AND_CI: UNAVAILABLE; 12/12 paired sanity, 2/12 exact token sequence, zero empty/leak; no task-score CI
BASELINE_TTFT / NATURAL_E2E / FIXED32 / FIXED128: batch1 2863.995 ms / NOT_ESTABLISHED / 5792.639 ms / 14826.975 ms; batch4 3069.171 / NOT_ESTABLISHED / 5995.280 / 15001.826 ms
ROUTED_STAGE_E2E_SHARE / PROFILER_OVERHEAD: NOT_VALIDATED / weighted 13.550% (diagnostic component run rejected for gate use)
PHYSICAL_EP4_PROOF: PASS
VIRTUAL_EP4_SEMANTIC_AND_WORKLOAD_PARITY: WORKLOAD_REPLAY_PASS; SEMANTIC_QUALITY_UNRESOLVED
SIM_EP4_MEDIAN_APE / P90_APE / RANKING_AND_SIGN: 15.705% / 46.817% / NOT_AVAILABLE; SIM_UNVALIDATED
SIM_EP8_EVIDENCE_LABEL: UNVALIDATED_EXTRAPOLATION
ONE_GPU_MODE / TWO_GPU_MODE / FALLBACK_TEST: BF16 TP1 in-process PASS / BF16 TP2 PASS / TESTED_4_2_1
MODEL_MEMORY_LIMITS: TP1 fits one H100 80GB in-process (~58.18 GiB model load); multiprocess startup stalled; no quantization/model/resolution substitution
FROZEN_BASELINE_BUNDLE_PATH: /home/esjung/MLLM-EP-github/poc_mllm_ep_foundation/artifacts/frozen_baseline_bundle.json
READY_FOR_PART2: DIAGNOSTIC_ONLY
PRIMARY_BLOCKER / NEXT_ACTION: full MMStar+ChartQA reproduction and >=16/8 real request calibration/heldout set / collect those anchors before formal method promotion
NEW_OPTIMIZATION_IMPLEMENTED: NO (Part 1)
```
