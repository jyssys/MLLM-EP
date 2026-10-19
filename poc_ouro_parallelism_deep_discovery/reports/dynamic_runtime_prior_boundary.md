# Dynamic-runtime prior boundary

Status: source and public-code audit complete for this branch. Worker runtime and quality tests are `NOT_RUN`.

## Outcome

Enabling is a direct prior for C03's terminal-page aliases. Its inference-only path retains the checkpoint gate. Its page table implements the same historical KV value rule as last-exited copying. It is a stronger baseline lead, not a result that this campaign defeated.

This does not make Ragged KV exact relative to dense full-R4 history. It can instead preserve the campaign's specified last-exited policy. The distinction is between preserving a chosen adaptive computation and preserving full-depth model outputs.

CDB remains the public executable original-gate baseline. Its published timing protocol and its public online-gate path are different evidence. The paper's replay measurements cannot establish free-running original-gate quality.

## Versions and code status

| Work | Latest version observed | Public-code status |
|---|---|---|
| [Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching](https://arxiv.org/html/2608.09444v2) | 2608.09444v2, 2026-09-25 | Official [continuous-depth-batching](https://github.com/kschwethelm/continuous-depth-batching/tree/abdfae0adbdd12be8c6123ccb34eee98a38afb39), revision `abdfae0adbdd12be8c6123ccb34eee98a38afb39`. |
| [Enabling Dynamic Computation in Looped LMs](https://arxiv.org/html/2610.09013v1) | 2610.09013v1, 2026-10-06 | `OFFICIAL_CODE_NOT_LOCATED` after the bounded paper, web, and repository searches. This is not proof of absence. |
| [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) | 2510.25741v6, 2026-10-11 | This worker verified metadata only. The coordinator inspected PDF v6. |

Ouro v4 is not the latest version. The coordinator's v6 inspection covers Sections 3.1–3.2, Table 2, Section 5.4.2, and Section 7.3. This report does not substitute a v4 body for that inspection. Enabling predates v6, so its description of an earlier Ouro release cannot establish what v6 omits.

The [registry](../artifacts/dynamic_runtime_sources.jsonl) preserves authors, timestamps, search queries, code locators, and inspection scope. It separates worker inspection from coordinator runtime evidence.

## CDB: three separate gate paths

The CDB paper schedules recurrent and boundary stages separately, with refill or no-refill. Its last-exited cache copies final KV into skipped slots. Its shared cache instead overwrites one slot across recurrent depths. These are different computations.

Appendix E.2 trains lookahead and preloop heads by distillation on frozen Ouro. Those are new trained components. The original checkpoint head remains a separate baseline.

Section 6.1 and Appendix C replay reference outputs and previously recorded exits, rather than active gate-driven free-running responses. Dynamic-arrival serving also replays a workload. The authors measure GPU executions under replay, not only simulated FLOP counts. All paper experiments use one H100-80GB, with Ouro-1.4B and Huginn-3.5B. [CDB, Sections 4 and 6, Appendices C and E](https://arxiv.org/html/2608.09444v2)

The official README also names Ouro-2.6B support. This is implementation scope, not a published two-size performance result. It describes a research runtime and currently greedy decoding, not a full serving engine. [Pinned README](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/README.md)

### Public code preserves the original-head option

`configure_ouro_exit_gate` rejects an untrained lookahead/preloop head unless the caller supplies trained weights or the checkpoint supplies that gate. The adapter selects `early_exit_gate` for the original path. It applies the gate after loop-boundary normalization. It declares that signal as same-step, rather than lookahead. See lines 54–113 and 249–299. [Pinned adapter](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/adapters/ouro.py#L54)

However, the generic config defaults to asynchronous execution, delayed gate consumption, and `kv_policy="single"`. These defaults do not define the campaign baseline. The exit-policy code promotes even a same-step signal to an offset-one decision when delayed asynchronous transport is active. Its immediate original policy updates cumulative probability and survival, then checks the threshold after the minimum depth. See lines 83–95 and 150–168. [Pinned config](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/config.py#L74), [pinned exit policy](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/exit_policy.py#L83)

The runtime flushes terminal KV copies on the compute stream before deeper reads. It assumes at most one recurrent token per request in flight. Replay consumes a staged signal readback but discards its value. Charging readback cost does not turn replay into live gate decisions. See lines 1190–1230 and 1696–1709. [Pinned runtime](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/continuous_api.py#L1190)

## Enabling: inference-only aliases and separate training changes

Ragged KV uses the original gates of the four released Ouro Base/Thinking checkpoints without new training. Dense-Ragged keeps full-depth prompt KV and changes decode. ETE-Ragged also changes prefill.

Equation 7 reads historical token j at depth `min(d,e_j)`. Appendix A.1 implements this with one-token pages. It aliases deeper logical entries to the terminal page, reclaims unused pages, and suppresses writes through aliases with `PAD_SLOT_ID`.

Appendix A.2 reports eager H100 runs. Its nested model runner still executes some exited lanes and is not an ideal work-conserving queue scheduler. The source does not establish true P2/P4 execution. Sections 4–5 separately change prior enforcement and use Ragged-aware training on new 160M models. Those trained models are not frozen Base checkpoints. The worker located no official code release. [Enabling, Sections 3–5 and Appendix A](https://arxiv.org/html/2610.09013v1)

The paper describes gate-driven generation, not CDB's recorded-exit protocol. This worker did not reproduce it or copy old specification scores. Public code remains unavailable to this audit.

## What remains exact?

The following comparison is an `ANALYTIC_IDENTITY`, not a model execution result.

Let `e_j` be a historical token's committed exit depth. Last-exited copying defines the logical value:

```text
read_copy(j,d) = KV(j,min(d,e_j)).
```

A page alias can define the same read without copying the tensor:

```text
address(j,d) = page(j,min(d,e_j))
read_alias(j,d) = contents(address(j,d)).
```

These expressions agree if the terminal page remains immutable and every alias points to the correct request, token, physical layer, and depth. Current-token appends need private writable pages. Reference counts or equivalent ownership rules must prevent premature release. Attention must use the same visibility and masking rules.

This identity does not assert `KV(j,e_j) == KV(j,d)` for an uncomputed deeper depth. It preserves terminal-fill semantics, not the dense full-R4 trajectory. Fixed R4 has no skipped depths, so it provides no terminal-fill copies to remove. Unequal computed depth states cannot share one page.

Even value-equivalent page addressing does not establish bitwise equality after a different attention kernel or BF16 reduction order. Validate the actual kernel before claiming matching logits or tokens.

## Stronger baseline and C03 boundary

The campaign's [load_engine](/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/runtime.py:23) source selects the original head and checks its checkpoint weight and bias. Adaptive mode sets threshold 0.5, minimum depth 2, and maximum depth 4. It explicitly disables delayed consumption, asynchronous batching, and synthetic exit replay. It selects `last_exited` and reserve-mode cache pressure.

The coordinator reports full-R4 prefill and tiny both-size checks for layer counts and terminal copies. Free-running calibration pilots are separate `QUALITY_ONLY` evidence. This worker inspected the loader source but reran none of those checks. The campaign does not implement page aliases at this cutoff.

A stronger same-policy baseline would retain these choices and replace copies only with a value-equivalent page table. That is exactly why Enabling constrains C03. Its storage mechanism must enter the nearest-prior comparison before C03 can claim novelty or performance advantage.

A complete comparison must preserve:

1. The checkpoint gate, threshold, minimum depth, and decision timing.
2. Full-R4 prefill and the same decode KV value policy.
3. Request-local terminal ownership and private writes.
4. The same useful branch/request work and concurrency.
5. The costs of table updates, allocation, padding, kernels, and synchronization.

New lookahead training, global-prior training, shared-slot overwrite, and ETE-Ragged prefill belong in separate changed-policy experiments. They cannot silently strengthen the unchanged-gate baseline.

The direct C03 contribution cannot be terminal aliases alone. A remaining claim needs a distinct mixed-depth runtime or distributed mechanism, plus measured benefit against the alias baseline. Code unavailability does not restore algorithmic novelty. It limits immediate reproducibility.

## Remaining scope and time

No inspected result closes both Base sizes, both depth modes, and true H100 P2/P4 serving. The current evidence supports a baseline lead and a semantic boundary. It does not establish a defeated prior or an architecture speedup.

The recorded worker segment ran from 2026-10-19T10:44:29Z to 2026-10-19T10:52:19Z. Its observed duration is 470 seconds, or 7 minutes 50 seconds. This is the accounting cutoff. These minutes overlap the coordinator's elapsed time. They do not add campaign elapsed hours. No GPU work, model change, installation, or deliberate wait occurred. Final artifact stamping after the cutoff is outside this duration.

