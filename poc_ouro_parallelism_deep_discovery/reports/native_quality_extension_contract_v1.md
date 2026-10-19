# Native TP4 baseline-quality extension

## Reason and boundary

The original native TP4 quality pilot contains only eight requests per domain.
Its free-running outputs can differ from the staged reference despite matching a short top-1 probe.
The completed staged extension now supplies 32 frozen requests per domain for both Base sizes.
This experiment evaluates the unchanged native TP4 baseline on those same IDs.

This is a baseline robustness check, not a new architecture.
The requests already have staged fixed/adaptive outcomes.
They are not a new held-out cohort or an independent candidate confirmation.
The original candidate held-out remains untouched.
No outcome selects a policy, prompt, tolerance, or new token cap.

## Frozen scope

- Models: the unchanged pinned Base Ouro-1.4B and Base Ouro-2.6B.
- Precision: BF16. Depth: fixed R4.
- Physical GPUs: ordered 4/5/6/7. Native TP4 only.
- Existing optimized vLLM path with async scheduling enabled.
- Full prefill, prefix caching off, chunked prefill off, and no EP.
- Existing depth-indexed KV and four local Q/KV heads per rank.
- Graph capture sizes: 1/2/4/8. One request generates at a time.
- Greedy natural generation, EOS allowed, and the unchanged 8192-token cap.
- The same frozen prompts and stop strings as the staged extension.
- Request timeout: 1200 seconds. Each model uses a separate guarded job.

The script reuses the configuration and generation path of `native_quality_pilot.py`.
It changes the cohort source and records extra provenance.
It changes no model code, attention, gate, sampler, or collective implementation.
It uses the read-only ownership extension directly, without the earlier four-model module's import-time controls.
It rejects inherited model-control variables before native imports.
Both allocator aliases remain job-local and disable expandable segments before Torch imports.
The documented process-title option remains job-local.

## Analysis contract

Require successful native status and outer-guard completion for each complete 64-request arm.
Check pinned revisions, frozen request identities, prompt IDs, raw output IDs, and configuration hashes.
Retain native completion and stop fields.
Reject unsupported native completion reasons.
The canonical evaluation text is the pinned tokenizer's complete decode of raw output IDs.
The native API text also remains in each raw record.
This distinction avoids silently treating text trimming as a token-parity result.
Native string-substring stopping and staged token-suffix stopping remain different backend mechanisms.
Matching stop strings alone do not establish identical terminal tokens.

Use the unchanged parser and completed-answer rule from the staged extension.
Length-capped answers remain incomplete.
Keep request-level 4000-resample bootstrap intervals and the existing bootstrap seed.
Compare native TP4 versus staged fixed R4 within each model and domain.
Retain exact raw output equality, first divergence, EOS/stop metadata, and correctness transitions.

Do not pool this lane with the earlier eight-request pilot.
Do not compare adaptive and fixed latency as an architecture speedup.
The recorded native request wall remains a quality-run diagnostic.
No candidate E2E row, task-quality PASS, or noninferiority claim follows.

## Admission and stopping

The source-hash contract must exist before launch.
Use fresh guarded output directories.
No protected factorial job or foreign GPU process can stop.
If runtime bring-up or provenance fails, preserve the failure rather than rebuilding an engine.
The run does not change the current zero-candidate shortlist.
