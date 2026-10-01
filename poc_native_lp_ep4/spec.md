# PoC Specification: Native Layered Prefill on TRUE EP4 for MLLM MoE Serving

**Authoritative server path:**  
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/native_layered_prefill_ep4_discovery_spec.md`

**Allowed GPUs:** `0,1,4,5` only  
**Primary model:** `Qwen/Qwen3-VL-30B-A3B-Instruct`  
**PoC type:** Native Layered-Prefill-on-EP discovery with CP comparison

---

## 0. Goal

This PoC must answer a narrow question:

> When Chunked Prefill (CP) and Layered Prefill (LP) are both implemented correctly on TRUE EP4, is there any residual problem that is common to both because of Expert Parallelism or multimodal/visual-heavy input?

Two discovery classes matter.

### EP-common residual

A physical EP cost remains large in both CP and LP, such as:

- Dispatch/Combine A2A dominating victim iterations;
- a single active prefill layer/group still causing excessive critical-rank delay;
- rank-tail effects that remain after tuning CP/LP granularity.

### Modality-common residual

At matched token counts, visual-heavy MLLM input causes a consistent extra cost in both CP and LP, such as:

- larger Dispatch/Combine latency;
- larger victim ITL/TBT;
- stronger rank fanout/tail;
- different scheduling granularity requirement.

Do not assume a gap exists. This PoC should try to falsify it.

---

## 1. Native LP requirement

The previous `LAYERED_EP_PHYSICAL_REPLAY` is not sufficient.

This PoC must execute actual Layered Prefill semantics:

- prefill requests maintain stage/layer-group progress;
- intermediate prefill hidden state is preserved across serving iterations;
- only the active prefill layer group advances in each iteration;
- active decode requests still execute their full decode forward and produce a token every iteration;
- prefill advances to the next layer group in later iterations;
- TRUE EP4 DeepEP Dispatch/Expert/Combine physically executes in every active MoE layer.

Results must be labeled:

- `NATIVE_LP_EP4_SERVING`
- `CONTROLLED_NATIVE_LP_EP4_HARNESS`
- `REPLAY_ONLY`

Only the first two count as successful implementation.

---

## 2. Upstream semantic reference

Use the public Layered Prefill implementation as the semantic reference:

- repo: `scale-snu/layered-prefill`
- pinned commit: `053f80e5201a7c0ab56e468e3d578907a2ca9cc3`

Preserve these semantics:

1. Prefill progresses across contiguous layer groups.
2. Decode is scheduled every iteration.
3. Prefill stage state is stored per sequence.
4. The model runner executes prefill only on its active layer subset.
5. Stage count may depend on prompt size.

Port only the scheduling/model-execution semantics needed for the existing Qwen3-VL + TRUE EP4 + DeepEP runtime.

Before coding, document:

- how upstream stage queues map to the existing harness;
- how active layer ranges are chosen;
- how intermediate prefill hidden states are stored;
- how all EP ranks preserve identical collective order.

---

## 3. Hardware/runtime

Use only:

`CUDA_VISIBLE_DEVICES=0,1,4,5`

Record:

- GPU UUID/model/HBM
- `nvidia-smi topo -m`
- NVLink/PCIe topology
- foreign GPU load
- git SHA
- model revision
- PyTorch/CUDA/vLLM/DeepEP versions
- Layered Prefill upstream commit

Primary topology:

`TP2-DP2-EP4-PP1`

Reuse prior validated settings:

- BF16
- 128 experts
- top-k 8
- linear placement
- EPLB off
- redundant experts 0
- DBO off
- prefix caching off
- same DeepEP backend across compared arms
- deterministic decode

---

## 4. Primary workload

Use the prior controlled P/D-interference setup if available.

### Resident decode

- global B=8
- deterministic / forced decode
- fixed output length, preferably 64 or 128
- stable decode-only baseline before prefill injection

### Injected MLLM prefill

Reuse the high-resolution frozen request:

- ~2,029 total prompt tokens
- ~1,980 visual tokens

Primary experiment:

> B8 resident decode + one injected high-resolution MLLM prefill.

This is intentionally a low-request controlled serving scenario.

---

## 5. Mandatory modality control

Use the same Qwen3-VL model with a text-only prompt matched to the visual request:

- approximately 2,029 tokens
- reuse the prior ~2,033-token text request if available

Mandatory comparisons:

- CP visual vs CP text
- LP visual vs LP text

Optional after primary:

- low/medium visual requests
- matched low/medium text requests

---

## 6. Scheduling arms

### A. Decode-only

B8 decode without injected prefill.

### B. Full

Ordinary unchunked mixed prefill/decode.

### C. Chunked Prefill

Primary:

- CP512

Secondary:

- CP1024
- CP256 only if a Pareto curve is needed

Use physical CP, not virtual replay.

### D. Native Layered Prefill

Mandatory:

- `LP_AUTO`: reproduce upstream stage-selection semantics as closely as possible.

Controlled sensitivity:

- `LP_FIXED_G4`
- `LP_FIXED_G8`
- `LP_FIXED_G16`

where G is the number of contiguous layer groups across 48 layers.

If the bounded implementation prefers upstream-supported group counts, use:

- G4
- G12
- G16
- G24

but record exact layer partitions.

If LP16 still misses the decode-latency target, optionally test:

- LP24
- LP48 (one prefill layer per stage)

Only after correctness is established.

---

## 7. Native LP execution contract

For G=4, conceptual execution should be equivalent to:

```text
Iteration 1:
  Prefill: L1-L12
  Decode:  L1-L48 -> decode token completes

Iteration 2:
  Prefill: L13-L24
  Decode:  L1-L48 -> next token completes

Iteration 3:
  Prefill: L25-L36
  Decode:  L1-L48 -> next token completes

Iteration 4:
  Prefill: L37-L48
  Decode:  L1-L48 -> next token completes
  Prefill completes
```

Required:

- decode emits one token per serving iteration;
- prefill hidden state after each group is preserved exactly;
- the next group consumes the previous group's output;
- attention/KV semantics are correct;
- EP collectives occur only on the active prefill layer set;
- all ranks execute collectives in the same order.

---

## 8. Correctness gate

Before timing, compare native LP against ordinary full-prefill execution.

Check:

- same prompt token IDs
- same visual preprocessing/embeddings
- same router IDs/weights where semantics require equality
- same final prefill hidden state within BF16 tolerance
- same first-token logits within tolerance
- same deterministic first token
- same deterministic continuation for several decode tokens

Report:

- max absolute error
- max relative error
- cosine similarity
- first-token equality
- N-token continuation equality

Correctness failure =&gt; `NATIVE_LP_INVALID`.

---

## 9. RQ1 — Does native LP protect decode under TRUE EP4?

For every LP group count measure:

- TTFT
- mean ITL/TBT
- P95/P99 ITL when sample count allows
- peak VSEL
- response-window peak ITL
- same relative SLO proxy used previously

Compare:

- Decode-only
- Full
- CP512
- LP Auto / fixed groups

Key question:

> Does increasing LP group count reduce decode interference under real EP4 as Layered Prefill intends?

If yes, retire the old claim that LP leaves decode stall unresolved.

---

## 10. RQ2 — Does native LP preserve MoE efficiency?

Measure:

- ExpertReloadAmp
- cumulative expert execution/request
- per-layer expert time
- active experts/layer
- estimated expert-weight traffic proxy
- prefill wall
- TTFT

Expected:

- LP remains close to ~1x reload relative to Full;
- CP512 may show repeated expert execution/reload.

Unexpected LP fragmentation should first be treated as an implementation bug.

---

## 11. RQ3 — Is there an EP-specific residual common to CP and LP?

For CP512 and every viable LP configuration measure:

- Dispatch ms/layer
- Combine ms/layer
- Expert ms/layer
- MoE total ms/layer
- logical Dispatch/Combine bytes
- per-rank send/recv bytes
- critical-rank timing
- maximum Dispatch quantum
- rank max/mean skew
- victim-iteration accumulated Dispatch excess

The question is not whether Dispatch exists.

The question is:

> After each scheduler is tuned to a reasonable TBT point, does EP communication still dominate or set the lower bound on victim latency or prefill efficiency?

A candidate EP-common issue must appear in both CP and LP.

---

## 12. RQ4 — Is there a modality-specific residual common to CP and LP?

Within each scheduling mode compare visual-heavy ~2K vs token-matched text ~2K.

For CP and LP separately measure:

- Dispatch
- Combine
- Expert time
- peak VSEL
- ITL/TBT
- TTFT
- rank fanout
- per-rank bytes
- route/rank skew

Then compare:

`visual penalty under CP` vs `visual penalty under LP`

Strong modality-common signal requires:

- visual &gt; text under CP;
- visual &gt; text under LP;
- the same physical component explains both.

If only one scheduling method shows the effect, do not call it modality-common.

---

## 13. RQ5 — Does an existing CP/LP configuration already close the Pareto gap?

Construct a Pareto plot.

x-axis:

- prefill efficiency / TTFT / prefill wall

y-axis:

- peak ITL or P99 TBT

Annotate:

- CP256/512/1024
- LP G4/G8(or G12)/G16/(G24 if needed)

Also annotate:

- ExpertReloadAmp
- Dispatch share

If any existing configuration already gives:

- acceptable decode TBT;
- low MoE overhead / reload;
- acceptable prefill latency;

then there may be no remaining research gap.

Do not force a method.

---

## 14. EP-common gap definition

Call a problem `EP_COMMON` only if:

1. it appears in both CP and native LP under TRUE EP4;
2. it remains after reasonable chunk/group tuning;
3. it is attributable to an EP-specific component such as:
  - Dispatch
  - Combine
  - critical-rank synchronization
  - rank fanout/tail
4. it materially affects victim ITL/TBT or prefill efficiency.

Not sufficient:

- CP expert reload only;
- LP research-harness overhead only;
- a tiny Dispatch component that does not affect serving latency.

---

## 15. Modality-common gap definition

Call a problem `MODALITY_COMMON` only if:

1. total token count is approximately matched;
2. visual &gt; text under CP;
3. visual &gt; text under LP;
4. the same underlying physical metric changes in both.

Do not call “visual has more tokens” a modality-specific mechanism.

---

## 16. Optional non-EP control

Only after primary TRUE EP4 experiments.

If cheap, run an architecture-matched non-EP control (for example TP-only on the same GPUs).

Purpose:

- estimate how much of the CP/LP penalty is specifically due to EP.

This is optional because Dispatch/Combine decomposition already exposes EP cost directly.

Do not delay the primary PoC for this.

---

## 17. Instrumentation

### Clean lane

For:

- TTFT
- ITL/TBT
- VSEL
- E2E
- prefill wall

Requirements:

- minimal instrumentation
- no Nsight
- no per-layer host synchronization

### Trace lane

For:

- Dispatch
- Combine
- Expert
- route bytes
- rank skew
- active layer group
- LP stage progress

Use:

- CUDA events
- existing DeepEP hooks
- lightweight route accounting

Do not use trace-lane wall times as headline serving latency.

---

## 18. Repetitions

Correctness:

- > =3 deterministic repeats

Discovery:

- 3 repeats/configuration

Final primary comparisons:

- > =10 paired repeats

Use request-level/bootstrap statistics where appropriate.

Do not treat 48 layers as independent serving samples.

---

## 19. Native LP validation checklist

Before accepting timing, verify:

- [ ] stage/layer-group state changes across iterations
- [ ] decode token emitted every iteration
- [ ] prefill advances only through intended layer group
- [ ] intermediate hidden state survives between iterations
- [ ] all 48 layers execute exactly once for prefill
- [ ] EP work occurs only in active prefill layers
- [ ] decode executes all 48 layers each iteration
- [ ] collective order identical across EP ranks
- [ ] no post-hoc arithmetic reconstruction
- [ ] correctness passes

Failure =&gt; `NATIVE_LP_INVALID`.

---

## 20. Required tables

### Serving latency


| Method      | TTFT | Mean ITL | P95/P99 ITL | Peak VSEL | Prefill wall |
| ----------- | ----: | --------: | -----------: | ---------: | ------------: |
| Decode-only |      |          |             |           |              |
| Full        |      |          |             |           |              |
| CP512       |      |          |             |           |              |
| LP Auto     |      |          |             |           |              |
| LP G4       |      |          |             |           |              |
| LP G8/G12   |      |          |             |           |              |
| LP G16      |      |          |             |           |              |


### MoE/EP decomposition


| Method   | Dispatch | Expert | Combine | ExpertReloadAmp | Max Dispatch Quantum |
| -------- | --------: | ------: | -------: | ---------------: | --------------------: |
| Full     |          |        |         |                 |                      |
| CP512    |          |        |         |                 |                      |
| LP Auto  |          |        |         |                 |                      |
| LP tuned |          |        |         |                 |                      |


### Modality comparison


| Method   | Workload | Dispatch | Combine | Peak VSEL | Rank Fanout | Rank Skew |
| -------- | -------- | --------: | -------: | ---------: | -----------: | ---------: |
| CP512    | Visual   |          |         |           |             |           |
| CP512    | Text     |          |         |           |             |           |
| LP tuned | Visual   |          |         |           |             |           |
| LP tuned | Text     |          |         |           |             |           |


---

## 21. Verdict taxonomy

### `LP_EP4_WORKS_NO_COMMON_GAP`

Native LP is correct, protects decode with reasonable group count, preserves expert efficiency, and no material common EP/modality residual remains.

Interpretation:

- retire the previous CP→LP→Ours gap story.

### `EP_COMMON_RESIDUAL_FOUND`

Both CP and tuned native LP show the same material EP communication/tail bottleneck that survives granularity tuning.

This is the strongest desired EP discovery.

### `MODALITY_COMMON_RESIDUAL_FOUND`

Visual-heavy input causes a matched-token penalty in both CP and LP and the same physical component explains it.

### `EP_AND_MODALITY_COMMON_RESIDUAL_FOUND`

Both categories are supported.

### `CP_SPECIFIC_ONLY`

The issue exists in CP but disappears in proper LP.

Interpretation:

- LP already solves the relevant issue.

### `LP_IMPLEMENTATION_OVERHEAD_ONLY`

Poor LP results are dominated by integration overhead rather than intrinsic EP behavior.

### `NATIVE_LP_INVALID`

Correctness or scheduling semantics fail.

---

## 22. Promotion heuristics

A candidate common EP gap should preferably show:

- > =15–20% meaningful victim/prefill penalty in both CP and tuned LP, or
- EP communication dominates the remaining latency gap;
- reproducible with &gt;=10 paired repeats;
- not removed simply by moving LP from coarse to finer groups;
- same physical cause in traces.

For modality-common promotion:

- visual vs text delta &gt;=15% under both CP and LP;
- CI excludes negligible effect;
- same metric/component moves in both.

Small/unstable ~5% effects should be classified weak/no-signal.

---

## 23. Artifact layout

```text
poc_native_lp_ep4/
  artifacts/
    manifests/
    gpu_audit/
    correctness/
    decode_only/
    full/
    cp512/
    cp1024/
    lp_auto/
    lp_g4/
    lp_g8_or_g12/
    lp_g16/
    modality_controls/
    final_metrics.json
    final_verdict.json
  reports/
    upstream_semantics_audit.md
    native_lp_implementation.md
    correctness.md
    serving_latency.md
    ep_component_analysis.md
    modality_analysis.md
    pareto_analysis.md
    final_summary.md
    reproduction.md
  figures/
  scripts/
  hooks/
  tests/
```

---

## 24. Required final summary block

```text
GPU_SET: 0,1,4,5
MODEL:
TOPOLOGY:
DEEPEP_BACKEND:
UPSTREAM_LAYERED_COMMIT:
LP_IMPLEMENTATION_SCOPE:

NATIVE_LP_CORRECTNESS:
DECODE_CONTINUES_EVERY_ITERATION:
PREFILL_LAYER_GROUP_EXECUTION_VERIFIED:

CP512_TTFT_MS:
CP512_PEAK_VSEL_MS:
CP512_ITL_P99_MS:
CP512_EXPERT_RELOAD_AMP:
CP512_DISPATCH_MS_PER_LAYER:

LP_AUTO_STAGE_COUNT:
LP_AUTO_TTFT_MS:
LP_AUTO_PEAK_VSEL_MS:
LP_AUTO_ITL_P99_MS:
LP_AUTO_EXPERT_RELOAD_AMP:
LP_AUTO_DISPATCH_MS_PER_ACTIVE_LAYER:

LP_BEST_GROUP_COUNT:
LP_BEST_TTFT_MS:
LP_BEST_PEAK_VSEL_MS:
LP_BEST_ITL_P99_MS:
LP_BEST_EXPERT_RELOAD_AMP:
LP_BEST_DISPATCH_MS_PER_ACTIVE_LAYER:

CP_VISUAL_MINUS_TEXT_DISPATCH_PCT:
LP_VISUAL_MINUS_TEXT_DISPATCH_PCT:
CP_VISUAL_MINUS_TEXT_VSEL_PCT:
LP_VISUAL_MINUS_TEXT_VSEL_PCT:

COMMON_EP_COMPONENT:
COMMON_MODALITY_COMPONENT:
PARETO_GAP_EXISTS:
EP_COMMON_RESIDUAL:
MODALITY_COMMON_RESIDUAL:

OVERALL_VERDICT:
NEXT_ACTION:
BRANCH:
COMMIT:
```

---

## 25. Interpretation rule

The most important rule:

> **Do not try to prove that a gap exists. Try to falsify it.**

If properly implemented Layered Prefill on TRUE EP4 already achieves:

- stable TBT;
- low expert reload;
- acceptable TTFT;
- no significant modality-specific residual;

then the previous motivation is not a valid research gap.

If a common EP or modality bottleneck survives both CP and tuned LP, that surviving bottleneck becomes the correct starting point for the next method.

Do not design a new method until this PoC finishes.
