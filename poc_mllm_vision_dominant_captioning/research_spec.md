# Vision-Dominant Detailed Captioning Regime Discovery
## Multi-Domain Image Captioning + Secondary Video Description under TRUE EP2 / TRUE EP4 + Structural EP8

**Status:** bounded discovery PoC  
**Model:** Qwen3-VL-30B-A3B-Instruct  
**Primary goal:** test whether a vision-dominant generation regime—where the textual instruction is identical across requests and the visual input determines most of the decode content—reveals substantially stronger MLLM-specific decode expert/EP signals than the previous question-conditioned long-reasoning regime.

---

## 1. Motivation

Previous project results established that image context causally changes later decode-time MoE routing, that this effect persists through long decode, and that expert-level changes become more visible at rank level as EP granularity increases. However, the practical value of vision-separated prefill signals remained weak: `P_separated` did not outperform `P_text` / `P_full` for future decode workload prediction, vision-aware co-scheduling had insufficient oracle locality headroom, cross-expert substitution was functionally unsafe, temporal reuse was unsafe, and vision-conditioned skip-damage estimation did not improve over text-prefill control.

A plausible explanation is that the previous workload was too text-conditioned. It used request-specific natural-language questions, answer options, and an explicit reasoning instruction. During long reasoning, the generated textual trajectory may become a strong compact representation of the task and progressively dominate decode behavior.

This PoC intentionally changes the regime:

> **Use the exact same textual prompt for every request and let the image/video itself determine the generated content.**

The primary image task is long-form detailed image description. A secondary, clearly separated video lane tests whether temporally rich visual input strengthens the same phenomenon further.

---

## 2. Main Research Questions

1. **Vision dominance:** with byte-for-byte identical text prompts, does the visual input produce a stronger request-specific decode MoE routing signal than in question-conditioned reasoning?
2. **Text vs vision predictiveness:** does `P_vision` or `P_separated` now outperform `P_text` and `P_full` for future decode expert behavior?
3. **Persistence:** does the vision-specific effect remain strong through long caption generation rather than only at early decode?
4. **EP scaling:** does the visual expert signal translate more strongly into rank-level behavior from EP2 to EP4 to structural EP8?
5. **Batch relevance:** under larger decode batches, do heterogeneous image domains create meaningful expert/rank heterogeneity, locality, or load-organization headroom?
6. **Video extension:** does detailed video description create an even stronger visual/temporal decode signature?

---

## 3. Scope

This is a **regime-discovery PoC**, not yet an optimization-method implementation.

Do **not** implement expert skipping, rerouting, reuse, mixed precision, replication, or production scheduling. First determine whether the task regime makes the MLLM-specific visual signal strong enough to justify another optimization PoC.

---

## 4. Formal Baseline

Use:

- Model: `Qwen3-VL-30B-A3B-Instruct`
- Frozen revision: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- BF16
- 48 sparse MoE layers
- 128 routed experts
- top-k = 8
- hidden size = 2048
- MoE intermediate size = 768

Reuse validated tracing and DeepEP instrumentation. Do not train model weights.

---

## 5. EP Topologies

### TRUE EP2

`TP1-DP2-EP2`, GPUs `4,5`.

Verify 64 experts/rank and real DeepEP sparse dispatch/combine.

### TRUE EP4

`TP2-DP2-EP4-PP1`, GPUs `4,5,6,7`.

Verify 32 experts/rank and real DeepEP sparse dispatch/combine.

### Mandatory EP8 structural projection

Physical EP8 is unavailable. Project actual expert IDs into 128 experts / 16 experts per rank.

Allowed EP8 outputs:

- changed-rank fraction;
- rank histograms and distribution divergence;
- rank max/mean;
- active experts/rank;
- per-domain rank prototypes;
- remote-destination fraction;
- structural batching/locality metrics.

Forbidden:

- EP8 latency;
- TPOT;
- E2E speedup;
- simulator-derived latency claims.

Label every EP8 result:

`STRUCTURAL_EP8_PROJECTION_ONLY`

---

## 6. Primary Image Task: Detailed Image Captioning

Every primary image request must use the **same exact text prompt**:

`Describe this image in detail. Mention the main visible entities, their attributes, spatial relationships, visible text when relevant, and notable actions or context. Do not add facts that are not visually supported.`

The prompt must be byte-for-byte identical across all requests.

Do not include dataset name, domain label, image category, question, answer options, target caption, or request-specific hint.

The only request-specific semantic content before generation should come from the visual input.

---

## 7. Multi-Domain Image Cohort

Construct at least four and preferably five semantically distinct visual domains. Do not treat dataset names as domains unless the visual semantics differ.

### D1 — Fine-Grained Everyday Scene Description: DOCCI

Primary source: **DOCCI**.

Purpose:

- long detailed natural-scene description;
- fine-grained entities and attributes;
- spatial relations, orientation, lighting, counting;
- visually related/contrasting scenes.

Target: `N >= 40`, preferably 64.

### D2 — Text-Rich / OCR-Centric Captioning: TextCaps

Primary source: **TextCaps**.

Purpose:

- image-contained text;
- relationship between visible text and surrounding objects;
- OCR-heavy visual semantics.

Use the universal caption prompt, not an OCR-specific prompt.

Target: `N >= 40`.

### D3 — Open-World / Novel Object Captioning: nocaps

Primary source: **nocaps**.

Purpose:

- uncommon / novel concepts;
- Open Images visual diversity;
- in-domain, near-domain, out-of-domain object regimes.

Stratify across nocaps subsets where practical.

Target: `N >= 40`.

### D4 — Dense Multi-Object / Relational Description

Preferred source: **Image Paragraph Captioning** / Visual Genome-MSCOCO paragraph-description assets.

Purpose:

- multiple entities;
- object-object relations;
- coherent paragraph-level scene description;
- dense relational semantics.

If the exact paragraph dataset is not straightforward to acquire, use a clearly documented Localized Narratives or Visual Genome-derived substitute with dense relational content.

Target: `N >= 40`.

### D5 — Scene / Layout / Spatial Description

Preferred source: **Localized Narratives over ADE20K** or a similarly scene-centric public subset.

Purpose:

- scene structure;
- layout;
- indoor/outdoor organization;
- spatial relationships.

Target: `N >= 40`.

### Preferred total size

- 5 domains × 40 = 200 image requests.
- Minimum acceptable: 4 domains × 40 = 160.

Keep domain counts balanced. Do not let DOCCI dominate the sample count.

If one source cannot be acquired, replace it with a semantically similar public source and document the deviation.

---

## 8. Frozen Selection Rules

Freeze selection before inspecting routing.

Requirements:

- image-hash disjoint;
- no duplicate image;
- no routing-based selection;
- no model-answer/caption-score cherry-picking.

Within each domain include low/mid/high image resolution, visual-token count, and scene complexity.

Record source dataset, source ID, image dimensions, processed visual-token count, reference description length where available, and generated caption length.

---

## 9. Decode-Length Contract

Primary natural generation:

- deterministic / temperature 0;
- EOS enabled;
- `max_new_tokens = 256`.

Prefer outputs >=64 tokens for the main routing/decode analysis.

Do not force a reasoning prefix or a `Final answer:` line.

If one domain naturally produces shorter descriptions, report that fact rather than adding dataset-specific instructions.

Analyze windows:

- 1–32;
- 33–64;
- 65–96;
- 97–128;
- 129–192;
- 193–256 when available.

---

## 10. Previous Reasoning Regime Control

Reuse a bounded subset from the previous question-conditioned long-reasoning experiment as `R_reasoning`.

The new captioning regime is `R_caption`.

Do not rerun the full old experiment.

Primary comparison:

> Does identical-prompt detailed captioning increase visual decode routing and prediction signal relative to request-specific reasoning?

---

## 11. Prefill Signatures

Build comparable signatures:

- `P_text`: text-prefill only;
- `P_vision`: vision-prefill only;
- `P_full`: modality-agnostic full-prefill;
- `P_separated`: `concat(P_vision, P_text)`.

Reuse the previous layerwise histogram / routing-weight normalization logic whenever possible.

---

## 12. Mandatory Text-Signature Leakage Audit

Because the text prompt is identical, `P_text` should carry very little request-specific semantic identity.

Report:

- pairwise P_text cosine distribution;
- P_text variance across requests;
- same-domain nearest-neighbor rate;
- domain centroid distances.

If P_text remains request-specific, investigate whether the apparent text signature includes image-placeholder, vision-grid, position, processor, or special-token effects.

Do not call such leakage “text semantics.”

This audit is mandatory.

---

## 13. Causal Image-Swap Experiment

Mandatory matched intervention:

- same exact text prompt;
- same teacher-forced decode tokens;
- same decode positions;
- image A vs image B.

Measure:

- expert assignment change;
- routing-weight divergence;
- EP2 changed-rank fraction;
- EP4 changed-rank fraction;
- EP8 structural changed-rank fraction.

Use both within-domain and cross-domain swaps.

Also repeat identical image runs to establish the noise floor.

---

## 14. Future Decode Prediction

Using only prefill signatures, predict future decode expert behavior.

Use simple, matched predictor families:

- nearest neighbor;
- kNN;
- centroid/prototype;
- simple regularized linear mapping if needed.

Targets:

- expert histogram cosine;
- top-k expert recall;
- active-expert-set recall;
- expert-frequency MAE;
- rank-distribution cosine;
- rank-load MAE;
- hottest-rank accuracy.

Report per decode window and aggregate.

Critical comparisons:

- `P_vision - P_text`;
- `P_separated - P_text`;
- `P_separated - P_full`;
- `R_caption - R_reasoning`.

Use paired bootstrap confidence intervals.

---

## 15. Vision-Dominance Success Criteria

Evidence is stronger if multiple conditions hold:

1. image swap causes materially larger expert change than in reasoning;
2. P_vision strongly outperforms P_text;
3. P_separated outperforms P_full, or at minimum matches P_vision while P_text collapses;
4. the visual increment persists into late decode;
5. rank-level divergence increases EP2→EP4→EP8 structural;
6. visual domains form distinct decode workload profiles.

---

## 16. Domain Separation

For P_vision, P_full, P_separated, decode truth, and P_text control, report:

- within-domain vs between-domain cosine distance;
- same-domain nearest-neighbor rate;
- silhouette score;
- centroid distance matrix;
- layerwise separation.

The key question is whether domains separate because of image content rather than request-specific question text.

---

## 17. Batch Design

### Signal discovery: B1

Use B1 for:

- image-swap causality;
- per-request signatures;
- decode persistence;
- clean modality analysis.

### Physical serving characterization: global B8 primary

Use B8 on TRUE EP4 if stable.

Reason:

- enough concurrency to expose expert-frequency concentration and cross-domain heterogeneity;
- still feasible on the current four-GPU machine;
- avoids immediately saturating memory/runtime with B16/B32.

Fallback: B4 only if B8 repeatedly fails bounded smoke tests.

### Structural sweep

Replay frozen traces for:

`B = {1,2,4,8,16}`

and B32 only if enough requests exist.

Measure:

- active expert union;
- paths/active expert;
- expert-frequency concentration;
- per-rank load;
- max/mean;
- remote destinations;
- cross-domain vs same-domain grouping effect.

Do not infer latency from structural replay.

---

## 18. Mixed-Domain Locality Revisit

Required but secondary.

Compare:

- FIFO mixed-domain;
- same-domain grouping;
- P_full grouping;
- P_vision grouping;
- P_separated grouping;
- future-truth oracle.

Primary batch: B8.

Secondary: B4 and structural B16.

Metrics:

- expert-union reduction;
- per-rank union reduction;
- expert-frequency concentration;
- rank max/mean;
- remote destination fraction.

If oracle headroom remains <10%, do not implement a runtime scheduler.

---

## 19. Physical EP2 / EP4 Evidence

Use representative frozen subsets under both physical topologies.

Measure:

- expert-level image-swap change;
- rank-level image-swap change;
- P_text/P_vision/P_full/P_separated prediction quality;
- domain separation at rank level;
- B8 workload characteristics.

Do not assume identical expert traces across topologies.

---

## 20. Mandatory EP8 Structural Evidence

Project actual expert IDs to EP8 and report:

### Image swap
- EP2 changed-rank fraction;
- EP4 changed-rank fraction;
- EP8 structural changed-rank fraction.

### Domain separation
- EP2 rank-domain separation;
- EP4 rank-domain separation;
- EP8 structural rank-domain separation.

### Prediction
- P_text / P_vision / P_full / P_separated rank prediction.

### Batching
- B8/B16 mixed-domain rank working sets;
- same-domain vs cross-domain grouping;
- future-truth oracle headroom.

Hypothesis:

> Finer expert-to-rank granularity may expose visual workload differences washed out when 32–64 experts share one rank.

---

## 21. Secondary Video Description Lane

Qwen3-VL natively supports video input. The video lane is secondary and must not contaminate the primary image conclusion.

Run only after the image infrastructure is stable.

### Universal video prompt

Use the exact same text for every video:

`Describe this video in detail, including the main events, entities, actions, and important temporal changes. Do not add facts that are not visually supported.`

No request-specific question.

### Preferred video datasets

#### V1 — ActivityNet Captions

Purpose:

- diverse real-world activities;
- temporally localized events;
- multi-event videos;
- dense event description.

#### V2 — YouCook2

Purpose:

- instructional/procedural video;
- temporally ordered actions;
- object/action transitions.

Use at least one; use both if acquisition/preprocessing is straightforward.

### Video cohort

Bounded discovery only:

- 16–24 videos per dataset;
- total <=48 video requests.

### Frozen preprocessing

Record and freeze:

- sampling FPS;
- max frames;
- max visual tokens;
- resolution policy;
- clip duration.

### Video batch

Primary B1.

Secondary structural B2/B4 if memory permits.

Do not force video B8.

### Video metrics

Use the same signature/routing metrics:

- P_text;
- P_vision/video;
- P_full;
- P_separated;
- video-swap causal change;
- late-decode persistence;
- EP2/EP4 if practical;
- EP8 structural projection.

Report video separately from image results.

---

## 22. Image vs Video Interpretation

Do not conclude that video is stronger merely because it contains more visual tokens.

Stratify or normalize where practical by:

- processed visual-token count;
- decode length;
- total multimodal input length.

Report raw results and token-matched subsets when feasible.

If video produces a stronger signal, discuss whether the difference could arise from token count, temporal structure, or context length.

---

## 23. Optional Functional Skip-Damage Revisit

Only if the caption regime shows a clear positive vision increment.

Do not repeat the entire previous damage-skipping PoC by default.

On a small bounded subset, reuse the exact drop-one machinery and compare:

- text-only damage predictor;
- vision-conditioned damage predictor.

Run this only if:

- P_vision/P_separated clearly beats P_text in captioning; and
- image-swap effects are materially larger than the previous reasoning regime.

Otherwise skip it.

---

## 24. Quality / Generation Sanity

This is primarily a routing/systems-signal PoC, but outputs must be legitimate detailed descriptions.

Where reference descriptions make evaluation straightforward, report lightweight caption-quality sanity such as CIDEr / BLEU / SPICE. Also report:

- words/tokens generated;
- repetition;
- malformed output;
- obvious refusal/failure counts.

Do not tune prompts per dataset.

---

## 25. Execution Budget

Primary image full-model requests: `<=320`.

Suggested allocation:

- acquisition/screening <=40 model runs;
- B1 signature/causal capture <=180;
- EP2/EP4 validation subset <=60;
- B8 serving characterization <=40.

Secondary video: `<=48`.

Suggested total ceiling: `<=368` full-model requests.

Never exceed 400 without stopping and reporting.

Generation guardrails:

- image max_new_tokens <=256;
- video max_new_tokens <=256;
- EOS enabled;
- hard timeout;
- no unbounded generation;
- no 32768-token cap.

---

## 26. GPU Hygiene

EP2: GPUs `4,5`.

EP4: GPUs `4,5,6,7`.

Before clean physical measurements:

- stop burners;
- check competing processes;
- warm up.

After all success/failure paths restore the existing burner policy.

Use cleanup/trap logic.

---

## 27. Stage Gates

### Gate A — Prompt/modality isolation

Require identical prompt hash and successful P_text leakage audit.

### Gate B — Vision-dominant decode signal

Strong positive if P_vision/P_separated materially outperforms P_text, image-swap expert change exceeds reasoning, and late-decode vision effect persists.

### Gate C — MLLM-specific increment

Require at least one:

- P_vision/P_separated > P_full with supported CI;
- stronger rank-level prediction than generic controls;
- vision-specific domain separation predicts decode workload beyond generic full-prefill.

### Gate D — EP relevance

Positive if changed-rank fraction or rank-level domain/workload separation increases EP2→EP4→EP8 structural, or large-batch mixed-domain rank heterogeneity becomes materially stronger.

### Gate E — Video extension

Promote video only if it adds signal beyond image captioning after controlling for visual-token count / length as feasible.

---

## 28. Verdict States

### `VISION_DOMINANT_CAPTION_REGIME_SUPPORTED_STRONG`

Identical-prompt captioning clearly shifts predictive power from text toward vision, the signal persists through long decode, and EP4/EP8 show stronger rank-level manifestation.

### `VISION_DOMINANT_CAPTION_REGIME_SUPPORTED_PARTIAL`

P_vision > P_text and causal image effects strengthen, but P_separated does not beat P_full or EP-rank increment remains weak.

### `GENERIC_FULL_PREFILL_STILL_SUFFICIENT`

Captioning increases visual effects but P_full remains as good as vision-specific/separated features.

### `TEXT_SIGNAL_STILL_DOMINANT`

P_text remains equally or more predictive even after leakage audit under identical-prompt captioning.

### `NO_REGIME_AMPLIFICATION`

Captioning does not materially strengthen visual decode effects relative to reasoning.

### `VIDEO_EXTENSION_SUPPORTED`

Secondary label when video provides a stronger, distinct visual/temporal decode signal.

### `VIDEO_TOKEN_COUNT_ONLY`

Video increase is explained by visual-token count rather than temporal/video structure.

### `RUNTIME_BLOCKED`

Technical failure only.

---

## 29. Required Reports

Create:

`poc_mllm_vision_dominant_captioning/`

Required reports:

- `reports/baseline_readiness.md`
- `reports/image_domain_cohort.md`
- `reports/universal_prompt_audit.md`
- `reports/text_signature_leakage_audit.md`
- `reports/image_swap_causal_effect.md`
- `reports/caption_vs_reasoning_regime.md`
- `reports/signature_predictor_comparison.md`
- `reports/domain_separation.md`
- `reports/batch_sensitivity.md`
- `reports/mixed_domain_locality_headroom.md`
- `reports/ep2_ep4_scaling.md`
- `reports/ep8_structural_projection.md`
- `reports/video_extension.md`
- `reports/final_summary.md`

Machine-readable artifacts:

- frozen image manifests;
- dataset/domain metadata;
- universal prompt hash;
- visual-token counts;
- generated lengths;
- prefill signatures;
- decode-truth summaries;
- causal-swap manifests;
- predictor outputs;
- bootstrap CIs;
- B1/B4/B8/B16 structural summaries;
- EP2/EP4 physical summaries;
- EP8 structural summaries;
- video preprocessing manifest;
- final verdict JSON.

---

## 30. Required Final Summary Fields

```text
BASELINE_STATUS:
IMAGE_COHORT_STATUS:
UNIVERSAL_PROMPT_STATUS:
TEXT_SIGNATURE_LEAKAGE_STATUS:
DOCCI_RESULT:
TEXTCAPS_RESULT:
NOCAPS_RESULT:
DENSE_RELATION_RESULT:
SCENE_LAYOUT_RESULT:
CAPTION_OUTPUT_LENGTH:
P_TEXT_RESULT:
P_VISION_RESULT:
P_FULL_RESULT:
P_SEPARATED_RESULT:
VISION_INCREMENT_VS_TEXT:
VISION_INCREMENT_VS_FULL:
IMAGE_SWAP_EXPERT_CHANGE:
REASONING_REGIME_IMAGE_SWAP_CHANGE:
LATE_DECODE_VISION_PERSISTENCE:
B8_MIXED_DOMAIN_ORACLE_HEADROOM:
EP2_CHANGED_RANK:
EP4_CHANGED_RANK:
EP8_STRUCTURAL_CHANGED_RANK:
EP_SCALING_INTERPRETATION:
VIDEO_STATUS:
VIDEO_VISION_INCREMENT:
VIDEO_EP8_STRUCTURAL_RESULT:
VERDICT:
NEXT_ACTION:
```

---

## 31. Final Interpretation Rule

The claim under test is not the trivial statement that captioning uses images.

The actual claim is:

> **When request-specific textual task information is intentionally removed, does the visual input become a substantially stronger determinant of long decode MoE expert/EP behavior, creating a more favorable regime for MLLM-specific serving optimization?**

Only if this is supported should the project return to vision-conditioned skipping, scheduling, placement, or another optimization mechanism.
