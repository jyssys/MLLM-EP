# Experiment log

## 2026-09-09 — initialization and source audit

- Created branch `flashvep/cp-composition-successor-poc` from commit `77c090a846bff30f169f4363d60504a5819af364`.
- Verified physical GPUs 4--7 were empty; physical GPU 0 had an unrelated user process and was untouched.
- Audited vLLM current main at `5acd95906b7c7a54dde89396d2bb06fe28ebeed0` and all required issues.
- Identified current-main binary dependency gap versus the stable local vLLM 0.20 environment.
- Established that Qwen3-VL cannot support faithful PCP or DCP on the requested four-GPU current path; selected cached DeepSeek-V2-Lite-Chat MLA MoE as the bounded faithful CP candidate.

## 2026-09-09 — live screening

- Installed an isolated CUDA-12.9-compatible vLLM 0.26 environment after current vLLM 0.29/current-main CUDA-13 binaries proved incompatible with the host driver. The baseline environment was not modified.
- Track A: ran vanilla, DCP4, n-gram speculation, and DCP4+n-gram. The combined path was 37.99% faster than DCP-only and 14.46% faster than spec-only by median request E2E, but 9.11% slower than vanilla. Cross-DCP greedy agreement was only 2/8 (no-spec) and 3/8 (spec); the performance signal is therefore excluded from positive evidence.
- Track B: PCP-only and DCP-only exact 16K prefill plus 32-token decode runs agreed exactly and showed zero clean first-decode transition excess. PCP4+DCP4 made no progress during model warmup in both a long run and a 180-second 8K bounded smoke. Current source already writes PCP-gathered tokens to canonical global DCP-aware slot mappings.
- Track C: screened 8K/16K/32K and concurrency 1/4/16 over PCP1/2/4 fixed fleets, then ran an exact-token content-matched c=2 control over natural/code/math/repetitive inputs. The raw per-regime oracle was 15.42% over best static and 12.49% over length-only, but the obvious rule `PCP4 at c=1, PCP1 otherwise` recovered 100% of that oracle. At c=2 every content class selected PCP2 with a 28.65--38.94% margin, so content/MoE geometry did not change the decision.

## Decision

No track met the paper-level promotion gate. Tracks A and C have decisive correctness/triviality kills. Track B's intended zero-transition operation is already present upstream, while its combined execution remains environment-blocked; the overall status is `PARTIAL_ENVIRONMENT_BLOCKED` rather than claiming all three were fully executed.
