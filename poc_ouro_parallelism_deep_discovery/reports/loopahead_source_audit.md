# LoopAhead: source and changed-assumption audit

Revision 1. Decision: stronger C07 prior, but no permitted new candidate.
No model experiment or performance measurement occurs in this audit.

## Verified source

The primary version is arXiv:2610.21594v1, submitted at 2026-10-15 09:38:11 UTC.
The abstract, HTML, and 25-page PDF are available. [Primary record](https://arxiv.org/abs/2610.21594)

| Primary locator | Bounded finding |
|---|---|
| §3.1, App. A.5 | Frozen target. Separately trained two-layer, width-512 LAD. |
| §3.2 | Mixed-depth branches require matching-depth ancestor KV. Suspension retains states. Verification reroots and prunes. |
| App. D, Algorithm 1 | A missing selected child starts at depth zero. Native proposals use acceptance ratios and positive residual correction. LAD uses argmax drafts. |
| App. A.1 | Ouro-1.4B/2.6B use fixed R4. The experiment disables adaptive exit. |
| App. A.5 | Four 48-GB A40s run independent shards, one request per GPU. BF16 target weights/caches, FP32 linear accumulation. |
| App. A.5, Eq. 11 | Warmed CUDA graphs. Synchronized wall-clock decode timing includes drafting, scheduling, and verification. It excludes prefill, setup, and the first output token. Throughput is total tokens/total decode seconds. |
| §4.2, App. A.5 | Greedy outputs match AR. Sampling uses T=1, top-p=0.9, and three seeds. The 7.82× maximum concerns Raven-TinyLlama/HumanEval, not Ouro. |
| Table 3, App. A.4 | Scheduling-only controls show no clear gain without LAD. |

These are author-reported findings, not independently reproduced results. [Versioned body](https://arxiv.org/html/2610.21594v1)

## Official code status

Status: `NOT_LOCATED`, not proof of nonexistence.
The inspected HTML links no LoopAhead repository.
The PDF's GitHub matches concern reference datasets, not a LoopAhead implementation.
Four web queries and two GitHub repository queries locate no official code.
No implementation revision or source-level function receives a verification claim.
`AdvanceBatch`, `NativeVerify`, and `Reroot` are paper pseudocode names, not inspected Python functions.
Third-party search summaries and their code buttons do not establish a public implementation.

The local PDF-text command failed because `pdftotext` is absent.
The browser's PDF reader supplied the fallback.
No package installation followed.
The source and search records preserve these limits.

## KV and correctness boundary

The following conclusions apply the campaign's existing contracts to the inspected mechanism.
They are deductions, not additional experimental findings.

Retention and repair are different operations.
A suspended node can preserve valid work only while its conditioning lineage remains valid.
A selected token does not validate descendants conditioned on a competing token.
Those descendants need pruning or recomputation, not a KV-address rename.
The paper's missing-child path therefore does not authorize shallow-state substitution.

A distributed C07 implementation still needs an explicit state owner for each branch version.
It must deliver any missing tentative ancestor KV before a dependent attention read.
It must publish accepted state and release losing state only after outstanding readers finish.
Local tree retention does not implement these remote dependencies.
Four independent experiment shards do not establish a distributed single-request protocol.

Distribution preservation also differs from identical seeded output.
For one proposal distribution q and target distribution p, accepted probability mass is `min(q(v), p(v))`.
The remaining mass uses the normalized positive part of `p-q`.
This identity requires the actual conditional distributions and a correct rejection implementation.
It does not validate acceptance based only on a drafter's argmax or confidence.
Batch-dependent BF16 differences and RNG ownership need implementation tests.
The unavailable code prevents that source-level check here.

The original adaptive target remains a separate target.
Its original gate, minimum depth, hazard CDF, and terminal `last_exited` KV semantics must remain intact.
Full-R4 verification alone does not certify that adaptive distribution.
No gate replacement, gate training, or delayed-authority shortcut follows from this source.

## C07 novelty and frontier decision

| Question | Decision |
|---|---|
| Does useful mixed-depth branch execution create a new C07 axis? | No. This is another direct mechanism prior, alongside the existing LoopSpec/WaveFront evidence. |
| Is retaining suspended KV a new distributed ownership protocol? | No. Local retention leaves remote state placement and publication unresolved. |
| Can this campaign adopt LAD? | No. A pretrained target remaining frozen does not make its separately trained drafter training-free. |
| Does removing LAD preserve the headline result? | The evidence does not support this inference. The ablation does not establish that result. |
| Can training-free policy ideas remain relevant prior art? | Yes. Training requirements do not erase the source's scheduling and lifecycle mechanisms. |
| Does absent distributed code establish novelty? | No. A distinct ownership transformation and a matched cost advantage remain necessary. |

The committed one-token graph retains its recurrent dependency chain.
Speculation adds provisional future-token nodes with their own lineage and state versions.
It does not produce the next pass's complete input before its producer finishes.
Thus this source changes the available proposal workload, not the original graph's data dependencies.

For a stride-one matched DAG, let S count batch steps and A count candidate-loop advances.
Then `A/S` is average branch width, not a speedup.
A legal timing model must charge:

`T_decode = sum_s T_target(batch_s, context_s, layout_s) + T_draft + T_schedule + exposed_state_and_commit_cost`.

Avoided rejected advances do not alone prove lower latency.
Wider batches, kernel shapes, cache traffic, and scheduler overhead can change each step's cost.
The same physical weights already support a co-batched branch baseline.
Replica placement cannot count that existing weight reuse again as a new saving.

The mandatory comparison remains co-batching, ordinary TP of the same branch batch, and split replicas.
Each comparison needs the same target, proposals, accepted-prefix rule, and branch work.
If scheduling changes the work, record that change separately from the ownership effect.
Charge remote tentative state, accepted-state publication, rejected work, and replicated capacity.
Native TP retains stationary local KV. Do not invent per-token KV transfers for that baseline.

The campaign's C07 architectural verdict therefore remains `GENERIC_ONLY`.
Distributed feasibility and performance remain `UNTESTED`.
Original-gate adaptive branch verification remains `UNVALIDATED`.
This source supplies no result that this campaign defeated.

## Audit limits and time

The audit reads §§3–4, limitations, Apps. A.1/A.3–A.5, selected B diagnostics, C.1, and Algorithm 1 in App. D.
It also reads the existing C07 card and branch-ownership contract.
The exact-frontier contract provides a read-only comparison.
No extra bibliography or neighboring model replaces the campaign target.
No weights, GPU jobs, trained components, engines, or environment changes occur.

Actual segment start: 2026-10-19 13:05:40 UTC.
Validated evidence cutoff: 2026-10-19 13:10:00 UTC. Recorded interval: 260 seconds.
Only final metadata publication follows this cutoff.
This worker interval is not additional campaign elapsed time.
