# Ouro Parallelism Deep Discovery
## A 10+ Hour, Training-Free Research Campaign for a Recurrence-Specific Distributed Inference Architecture

**Version:** 1.0  
**Prepared:** 2026-10-19  
**Audience:** Codex Astra research/coding agent operating in the user's server environment  
**Language:** English  
**Required server specification path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_parallelism_deep_discovery_spec.md`  
**Suggested isolated workspace:** `/home/esjung/MLLM-EP-github/poc_ouro_parallelism_deep_discovery/`  
**Campaign duration:** at least 10 hours of substantive research activity; plan for approximately 12 hours across resumable sessions  
**Hardware scope:** H100, P=2 and P=4 now when actually available; P=8 is a future target, not current authorization  
**Current authorized physical GPU pool:** 4, 5, 6, 7 only; another experiment is already using this pool  
**Status of this document:** research instructions, not a claim that any new architecture or speedup has been demonstrated

---

## 0. Executive mandate

Discover whether the **particular computational structure of the released dense Ouro looped Transformers** can support a new, useful, training-free distributed inference architecture.

The preferred outcome is a new parallel execution axis or execution abstraction. A materially new recurrence-aware composition or variant of TP, PP, DP, or CP is the second preference. A generic scheduler, a relabeled standard parallelism, an application of CDB to several replicas, or a single-GPU kernel optimization alone does not satisfy the primary objective.

The desired research character is analogous to the following examples, not a demand to copy their mechanisms:

- **DistriFusion:** model-specific structure makes a nontrivial distributed execution possible.
- **AsyncDiff:** a carefully defined dependency relaxation creates useful multi-device overlap.
- **LoPA:** an inference-algorithm opportunity and an accompanying distributed runtime are designed together.

For Ouro, the agent must independently identify the corresponding opportunity. Neither stale activations nor independent branches should be assumed to exist merely because they were useful in diffusion models.

**Core question:**

> Which property of Ouro's repeated shared-weight computation, depth-indexed state, and causal execution permits a legal or quality-controlled execution transformation that improves inference beyond a competent same-resource conventional baseline?

The agent is not required to invent a successful architecture by force. It is required to perform a serious, sustained search, formulate falsifiable alternatives, test the smallest decisive facts, and return a defensible shortlist or a well-supported negative result.

### 0.1 The outcome is a research dossier, not just a literature survey

Produce:

1. A verified source and implementation map, including nearest competing methods.
2. An audited description of the actual Ouro execution semantics.
3. A diverse candidate register, including ideas not suggested in this document.
4. Dependency, state-ownership, memory, and critical-path analyses for finalists.
5. Measured or explicitly modeled headroom with uncertainty and quality status.
6. Concrete recommended next PoCs, or explicit reasons no candidate currently deserves implementation.

Do not finish with only “try attention,” “use weight sharing,” or “investigate speculative decoding.” Every surviving candidate needs a mechanism, a legal schedule, a cost argument, a comparison, and a falsification test.

---

## 1. Non-negotiable research constraints

### 1.1 Model family

- The primary models are the released **dense, non-MoE Ouro-1.4B and Ouro-2.6B** checkpoints.
- Both sizes must be addressed. Success on only one size is a partial result, not completion of the user's full target.
- Base checkpoints are the initial common target for fixed and adaptive evaluation. Thinking checkpoints may be additional fixed-depth quality checks, but do not assume their adaptive exit behavior is interchangeable with Base.
- Do not pivot to Huginn, Raven, a dLLM, an image diffusion model, or a looped MoE as the empirical target. Such work may be read for transferable ideas and counterexamples.
- Do not convert Ouro into an MoE, retrain its architecture, or rely on a different model's convergence properties.
- Read the exact loaded config. The publicly inspected Base configs have 24 physical layers for 1.4B and 48 for 2.6B, with four recurrent passes. Do not reuse the old “24 layers divided by four” assumption for 2.6B. [S01–S03]

### 1.2 Fixed and dynamic depth

The sought mechanism should have useful headroom in **both**:

- Fixed R4 inference, without using early-exit heterogeneity as its only source of benefit.
- Adaptive inference with the established original gate and an explicitly pinned runtime/KV policy.

Compatibility and speedup are different claims. “The implementation does not crash in fixed R4” is not evidence of fixed-depth headroom.

Do not require uniform improvement at every context, batch, and GPU count. Instead, define representative operating points in advance and seek positive evidence for both model sizes and both depth modes. A candidate beneficial only in one mode must be labeled `MODE_SPECIFIC`, rather than silently relaxing the target.

Changing q, minimum depth, maximum depth, output length, or KV semantics only for the proposed method is not a systems speedup comparison.

### 1.3 Training-free means no newly trained component

Allowed:

- Unchanged released model weights and the original already-trained Ouro exit gate.
- Runtime scheduling, exact algebraic reorganization, placement, communication protocols, caching of genuinely reusable values, and kernel implementation changes.
- Hardware autotuning and latency measurement on a separate calibration workload.
- Analytically defined or non-learned inference approximations, but only under the explicit approximate-method track and its quality tests.
- Numerical solvers operating during inference, provided all computation is charged and no learned surrogate is introduced.

Not allowed:

- RL, SFT, distillation, gate training, LoRA/adapters, learned hidden-state predictors, trained draft heads, or a learned coarse propagator.
- A new trained gate borrowed from a paper presented as though it were the original Ouro gate.
- Parameter updates hidden behind “calibration.” A coefficient fit to model outputs is not automatically exempt from the no-training rule.
- Quantization or reduced-precision shortcuts as the main source of improvement. Use BF16 model execution consistently; match any higher-precision accumulations across comparisons.

Existing additional trained components in a prior work can be documented as prior art, but a proposed solution that requires them is outside this campaign's training-free contract.

### 1.4 Architecture, not only policy

The ongoing CDB refill × hindsight factorial study is a separate experiment. Its eventual evidence can constrain this campaign, but:

- Do not rerun or replace it.
- Do not turn this campaign into another request-placement or RL project.
- Do not treat its unfinished results as known.
- A plain CDB+DP extension or a better request-depth predictor is not the requested new architecture.

### 1.5 Hardware, latency regimes, and baseline strength

- Evaluate P=2 and P=4 on actual authorized H100 devices when available.
- Analyze P=8 feasibility and topology sensitivity, but mark unmeasured P=8 cells as projected or not run.
- Never emulate eight ranks on four GPUs and label the result “8GPU performance.”
- Prioritize low-concurrency inference, particularly system-wide C=1, 2, 4, and 8. Include moderate C=16/32 only where the candidate's mechanism needs it.
- A multi-request-only candidate may remain on the shortlist, but cannot be described as accelerating a single request.
- A modest but reproducible, structurally interesting gain can be valuable. There is no automatic rejection simply for failing to reach 1.20×.
- Beating a deliberately weak naive baseline is not sufficient. Report both 1GPU-normalized speedup and incremental improvement over the strongest relevant same-P baseline.

---

## 2. Campaign execution and the 10+ hour requirement

### 2.1 What the duration requirement actually means

The user requests a long-running investigation, not a quick brainstorm. Budget at least **10 hours of substantive campaign activity**, aiming for about 12 hours.

Count primary-source reading, implementation inspection, candidate derivation, mathematical or executable DAG tests, cost modeling, approved microbenchmarks, analysis, and report synthesis. Do not count arbitrary sleep, a countdown, repeated identical searches, an unattended stalled command, or simply keeping a process alive.

Record actual UTC start/end times, active segments, tool or experiment identifiers, and outputs. Do not manufacture time logs. Concurrent subagents do not multiply elapsed campaign time; report elapsed and summed worker time separately when measurable.

A prompt cannot override the external agent's session, tool, quota, or permission limits. If the host cannot sustain the entire campaign, checkpoint and report `PAUSED_RUNTIME_LIMIT` or another precise blocker. Resume from saved state in a later session. Never claim ten hours of work when fewer were performed.

### 2.2 Twelve-hour working plan

| Phase | Budget | Required product |
|---|---:|---|
| 0. Existing evidence, environment, and constraints | 0.75 h | Prior-evidence ledger; safe resource plan |
| 1. Primary literature and code inspection | 2.00 h | Verified source map; nearest-prior matrix |
| 2. Independent candidate generation | 1.50 h | Diverse initial candidates; at least three unseeded ideas |
| 3. Dependency and ownership red-team | 1.25 h | Candidate DAGs; first pruning with explicit reasons |
| 4. Cost and resource feasibility | 2.00 h | Critical-path accounting; minimal measured/modelled tests |
| 5. Minimal execution/quality falsification | 2.00 h | Small decisive probes for the best candidates |
| 6. Both-size/both-mode robustness | 1.50 h | 1.4B/2.6B × fixed/adaptive × P matrix |
| 7. Synthesis, adversarial review, next-PoC plans | 1.00 h | Final ranked dossier and reproducibility package |
| **Planned total** | **12.00 h** | **At least 10 substantive hours before normal completion** |

Budgets are reallocatable. If a GPU is occupied, perform source inspection, DAG derivation, code reading, and analytic work rather than competing with the ongoing experiment. Report the empirical gap honestly.

Do not stop the entire campaign because the first candidate fails. Reject that candidate and explore a materially different family. Conversely, do not spend the entire budget rescuing one speculative idea.

### 2.3 Breadth and depth targets

These are research-completeness targets, not permission to pad the dossier:

- Screen roughly 25–40 distinct primary works or relevant official implementation components.
- Deep-read approximately 8–12 nearest works, including algorithm, limitations, experiment protocol, and code where available.
- Formulate at least 12 materially distinct initial candidate cards spanning at least five mechanism families.
- Include at least three candidates invented by the agent outside the seeded areas below.
- During candidate generation, spend at least half of the effort on potentially new execution axes/abstractions rather than only tuning familiar hybrids; seek at least four such initial hypotheses, without pretending they survive verification.
- Produce dependency and cost screens for at least six candidates.
- Seek small decisive probes for about three or four candidates, subject to feasibility and GPU access.
- Return zero to three serious finalists. Do not fabricate survivors to meet a count.

If a numerical target would require irrelevant sources or duplicate ideas, explain the shortfall and preserve rigor.

### 2.4 Checkpoints and resume

After every substantial work block, and approximately every 45–60 minutes when the session remains active, update:

- `research_state.json`
- `reports/progress_log.md`
- `reports/next_actions.md`
- `artifacts/time_ledger.jsonl`

Save candidate status, evidence paths, rejected hypotheses, pinned versions, running commands/PIDs owned by this campaign, pending questions, and the next concrete action. Save research conclusions and reproducible derivations, not a purported private reasoning transcript.

Use a single coordinator for final claims. If the runtime genuinely supports subagents, parallelize literature, semantics, architecture generation, and skeptical review. Otherwise perform these as separate sequential roles and label them accurately.

---

## 3. Read existing work before proposing another architecture

### 3.1 Existing local specifications to inspect, read-only

Look for the following under `/home/esjung/MLLM-EP-github/poc_flashvep/reports/`, and inspect any actual report/worktree locations referenced inside them:

- `depth_cdb_factorial_hindsight_spec.md` — ongoing, must not be interfered with.
- `decode_first_depth_aware_dp_spec.md`
- `decode_first_depth_aware_dp_spec_v3.md`, if present.
- `recurrent_memory_compute_asymmetry_discovery_spec.md`, if present.
- `async_recurrent_pp_poc_spec.md`, if present.

A specification is not an experiment result. Source code and logs are needed before attributing exact implementation behavior.

User-reported prior locations, to locate rather than assume:

| Study | Report suffix / location hint | Reported branch or commit |
|---|---|---|
| Predictive async recurrent PP | `MLLM-EP-async-recurrent-pp/poc_async_recurrent_pp/reports/final_report.md` | `flashvep/async-recurrent-pp-poc`; `458c1fe9b43f08863308d3b661a4163a854cb9aa` |
| Memory–compute asymmetry | `MLLM-EP-recurrent-asymmetry/poc_recurrent_asymmetry/reports/final_report.md` | `flashvep/recurrent-memory-compute-asymmetry-poc`; `eb7d6a400f794d16952cb15889e8d567a5eb0a62` |
| Decode-depth DP v3 | `MLLM-EP-decode-first-depth-dp-v3/poc_decode_first_depth_aware_dp_v3/reports/final_report.md` | `flashvep/decode-first-depth-aware-dp-v3`; `0eb9c04d1eef6af33c30a0ca2c0244fb26e41950` |

Use targeted repository/worktree searches, not unrestricted recursive scans of unrelated users' directories. Missing access is `UNAVAILABLE`, not proof that a result or repository does not exist.

### 3.2 Historical evidence ledger

The following are **user-reported observations from previous campaigns**, not newly verified results of this campaign:

| Prior direction | Observation | Consequence for this search |
|---|---|---|
| Predictive async PP, Ouro-1.4B-Thinking, BF16, R4 | 16/domain; exact GSM8K 15/16, MATH 14/16; W1 stale 8/16 and 3/16; W1 delta 7/16 and 3/16; W0 initialization 6/16 and 1/16 | Simple stale-state/delta replacement failed quality badly. Do not repeat it under a new name. |
| Memory–compute asymmetry | Shared weights 2.672 GiB; fixed-depth B1 KV about 0.75 GiB at 1K and 12 GiB at 16K | Parameter sharing does not imply depth-state sharing or cheap migration. These are footprints, not measured HBM traffic. |
| Distributed-attention track | Separate attention-core fraction about 7.3%; ideal four-way component model implied about 1.058× end-to-end speedup; tested CP operator slower than native | Generic CP is not a fresh candidate without a changed operating regime or distinct mechanism. Component share is condition-specific. |
| Prefill wavefront track | Tested exact DAG had ordinary causal-PP structure | Chunking plus loop wrap-around alone is not enough for novelty. |
| Depth-aware admission/hindsight DP | 128 requests; C64 model estimate about 1.146×; live approximately 1.046× with interval about [0.923×, 1.154×], assuming reported percentages meant latency reduction | Imbalance exists, but the tested placement heuristic did not establish robust useful incremental gain. It is not a mathematical optimum. |
| Baseline and numerical gaps | Native TP/small chunks had unresolved tolerance issues; HBM counters unavailable; batch-output equivalence unresolved in v3 | Treat numerical and traffic claims carefully; do not invent exactness or counters. |
| Current five-arm ablation | Running separately; outcomes not yet supplied | Read only stable completed artifacts; do not infer its verdict. |

The original adaptive-vs-fixed quality warning is retained: Base 1.4B q=.5, R2–R4, `last_exited`, 32/domain gave adaptive 21/32 GSM8K and 19/32 MATH versus fixed 22/32 and 24/32. That is a property of the existing operating point, not evidence that a new scheduler or architecture caused the gap.

### 3.3 Reopening a rejected family

A reopening requires a written **changed-assumption statement**:

1. Precisely which previous mechanism failed?
2. Which new fact, representation, cost reduction, runtime feature, or operating regime changes that failure?
3. What is the smallest test that could refute the new version?
4. Why is this not just adding more tuning to an already failed variant?
5. What is the time budget before abandoning the reopening?

The old finding is neither erased nor treated as a theorem about all future variants. In particular, the old MoE speculative-decoding result is not an Ouro impossibility proof, but a generic speculative method with an Ouro name is not automatically novel either.

---

## 4. Source-verification and literature-search protocol

### 4.1 Verify, do not inherit citations uncritically

Search the live literature at campaign start. Pin paper versions and access times. Earlier chat summaries, model memories, search snippets, and existing specs are **leads**, not authoritative evidence for external claims.

For every important claim capture:

- Exact title, authors, identifier, version, publication/access dates.
- Primary URL and relevant section/page/algorithm/table or code file/function.
- What the source actually supports, in a short paraphrase.
- Whether the result is trained, training-free, exact, approximate, replayed, simulated, single-GPU, or multi-GPU.
- The condition under which it was measured and any important limitation.
- Official authors' code vs independent reimplementation, with provenance and revision.

Do not call a work “SOTA” or “latest” without a defined comparison and an up-to-date search. Do not make up an arXiv number or repository. In particular, disambiguate the 2024 asynchronous-denoising AsyncDiff from unrelated papers sharing its name.

Use primary sources for technical facts: papers, official project/checkpoint pages, author repositories, and vendor documentation. Aggregators may help locate a source but are not substitutes for it. Read figure/table pixels when extracting visual values; do not invent exact plotted values.

### 4.2 Search families

Use multiple rounds, changing queries as candidate mechanisms evolve:

| Family | Example queries to adapt |
|---|---|
| Direct Ouro serving | `Ouro looped transformer distributed inference`, `Ouro recurrent depth parallelism`, `Ouro inference KV cache training free` |
| Recurrence execution | `reentrant pipeline inference tied weights`, `cyclic pipeline shared parameters`, `weight stationary recurrent transformer multi GPU` |
| State/compute ownership | `attention FFN disaggregation inference KV owner`, `remote compute stationary KV cache`, `distributed weight shared inference` |
| Communication | `communication avoiding tensor parallel transformer`, `persistent distributed inference recurrent`, `layout propagation tensor context parallel inference` |
| Numerical parallel time | `DEER nonlinear recurrence parallel evaluation`, `parallel in time recurrent inference`, `Picard Parareal pretrained inference correction cost` |
| Depth/token speculation | `Ouro self speculative decoding`, `LoopSpec`, `WaveFront decoding looped`, `training free recurrent branch verification` |
| Reuse and residuals | `cross loop residual attention Ouro`, `looped KV compression training free`, `exact reuse recurrent linear projection` |
| Conventional nearest baselines | `token level pipeline TeraPipe`, `continuous depth batching`, `TP PP CP hybrid serving low latency` |

For each plausible candidate, search for its *mechanism* without the words Ouro or looped. This detects generic prior art hidden behind a different model name.

Search backward references and forward citations for the nearest works. Record explicit negative searches carefully: “not found in these sources and queries,” not “no prior work exists.”

### 4.3 Mechanism-transfer worksheet

For DistriFusion, AsyncDiff, and LoPA, write four short entries:

1. Original bottleneck.
2. Model property that made the transformation possible.
3. Extra work, state, or approximation introduced.
4. Which corresponding property is present, absent, or unknown in Ouro.

For LoPA, inspect its token-filling-order exploration and distributed winner-cache protocol. Do not interpret recurrent depths of one Ouro token as independent branches by analogy. For AsyncDiff/DistriFusion, treat stale-feature quality as a measured property of those models, not a transferable theorem. [S04–S06]

---

## 5. Audit the actual Ouro computation before changing it

### 5.1 Four identities must be kept distinct

For every task and tensor, record:

- Request and token position.
- Logical recurrent depth r.
- Physical layer l and operator type.
- Physical device, tensor layout, and KV ownership.

A useful node label is `op(request, token_or_chunk, recurrent_depth, physical_layer, operator)`.

The fact that W_l is the same across r does not make the inputs or outputs the same. Model identity is not data readiness.

### 5.2 Build a faithful dependency graph

Start from the pinned implementation, including:

- Token embeddings and any prelude.
- Attention projections, rotary embeddings, query/key scaling, masks, and KV writes.
- Attention score and output computation, output projection, residual connections, and every normalization.
- MLP gate/up/down projections and nonlinearities.
- Per-loop normalization, exit gate, output selection, and LM head.
- Loop wrap-around to the next application of the same physical stack.
- Autoregressive sampling/argmax and availability of the next input token.
- Cache-policy-specific dependencies for missing-depth states in adaptive mode.

The inspected official HF forward loops through the shared layer stack and applies a normalization at each loop boundary. An overly simple equation `h_r = F(h_{r-1})` is useful notation only after F includes all required operations and state. [S03]

A schematic exact dependency is:

`h[r,l,t] = Block_l(h[r,l-1,t], KV[r,l,<=t], mask, position)`

with the last-layer output, required loop-boundary normalization, and exit policy connecting to the next r. The actual cache object is stateful: evaluating a branch must not silently overwrite the authoritative cache.

### 5.3 Prefill and decode are not interchangeable

- Prefill: all prompt token IDs are known, but intermediate representations and causal KVs are not all ready at once.
- Decode: the current input token is known after its predecessor is generated; later tokens are not freely available.
- The first output token may be produced by prefill; use one consistent definition in timing and depth accounting.
- Independent DP replicas do not have an inherent global decode-step barrier. Do not insert one into the baseline and claim to remove it.
- Online adaptive gate decisions are not known before the computations that produce them, unless a proposed method adds a separately justified inference procedure.

Audit the actual fixed and adaptive runtime paths. The original checkpoint's exit gate and the runtime's physical early skipping are separate facts. Record logical exits, executed loops, and allocated KV slots independently.

### 5.4 Ownership ledger

For every candidate, fill:

| State | Creation point | Owner | Readers | Mutable? | Version/lifetime | Bytes | Transfer or recompute cost |
|---|---|---|---|---|---|---:|---:|
| Weight shard/tile | Load/setup | TBD | TBD | No | Model revision | TBD | TBD |
| Historical KV | Past exact or selected adaptive execution | TBD | TBD | Normally immutable after commit | Request/layer/depth/position | TBD | TBD |
| Current-token KV | Current attention projection | TBD | TBD | Until committed | Candidate/version | TBD | TBD |
| Hidden activation | Predecessor operator | TBD | TBD | No after publication | Node ID | TBD | TBD |
| Gate/logit state | Gate/head computation | TBD | TBD | Per output step | Request/token | TBD | TBD |
| Candidate scratch | Branch/solver start | TBD | TBD | Yes | Discard/commit | TBD | TBD |

Never solve a migration problem by forgetting cache reads, nor a branch problem by committing losing-branch KV.

---

## 6. Search Area A — Shared-weight execution with recurrence-aware residency

**Priority:** high. **Category:** potentially new execution topology; also potentially only a PP/DP variant.

### Question

Can the same physical weights be kept stationary while multiple **already ready** tasks from different recurrent depths use them in a way that conventional model partitioning cannot realize at comparable memory and communication cost?

### Candidate ingredients to examine

- Persistent stage or operator workers keyed by physical layer rather than logical unrolled depth.
- A legal recurrent task frontier that groups shared-weight GEMMs across depth without changing per-token dependencies.
- Nonuniform replication of selected physical layers/operators where tied reuse changes the memory economics.
- Tile-level reuse across independent tokens/chunks/requests combined with cross-device ownership.
- A cyclic or reentrant execution layout that has a material advantage beyond generic causal PP.

### Required argument

Identify the saved cost: repeated HBM reads, redundant weight replicas, exposed dispatch latency, or a concrete pipeline bubble. Show why the saved cost is significant and why ordinary DP+CDB, grouped GEMM, graph capture, or tuned PP does not already obtain it.

Do not assume the whole model fits in L2. Do not assume a layer's weight tile survives until its next loop use after all intervening layers. Compute cache capacity and reuse distance; use counters only if actually available.

### Smallest probes

- A ready-task trace with dependencies and physical-layer identities.
- Shared-pointer vs separate-copy execution with identical weight values.
- Per-operator batching and residency timing at relevant shapes.
- A tiny executable scheduler that proves the proposed overlap actually has ready work.

### Failure signatures

- Requires R2 of a token before R1 completes.
- Reuse appears only in a synthetic ordering impossible for Ouro.
- Matches CDB/ordinary grouped batching exactly with no distributed benefit.
- Saves weight memory but does not change useful latency, throughput, or feasible capacity.
- Benefits vanish against a tied-weight-aware conventional baseline.

---

## 7. Search Area B — Decouple KV ownership from shared compute ownership

**Priority:** high. **Category:** state/compute distributed architecture.

### Question

Can an exact execution keep large historical KV state stationary while moving only small ready activations to reusable shared-weight compute workers, exploiting Ouro's repeated operators without repeatedly migrating whole requests?

Potential organizations include attention-local / MLP-remote execution, operator pools, replicated hot projection stages, and a loop-aware combination with head/tensor partitioning. These are hypotheses, not asserted improvements.

### What would make it Ouro-specific?

The repeated physical operator set might permit a compute pool to serve several logical depths without hosting the weights of an unrolled dense network. The agent must show that this changes the memory/communication/performance optimum, rather than merely restating generic attention–FFN disaggregation.

### Mandatory checks

- Follow every residual, normalization, and Q/K/V/output dependency.
- Measure both directions of activation transfer and metadata costs.
- Include the cost of moving or reducing results back to the authoritative state owner.
- Bound any serial path that remains even if other GPUs are idle.
- Test C=1 separately. Cross-request pooling does not automatically accelerate one request.
- Compare against conventional TP and generic disaggregation using the same worker resources.

The old attention-core fraction of about 7.3% does not forbid this area: it may target the larger projection/MLP path instead. But it cannot be used to claim attention itself is now a large bottleneck without a new measurement.

### Decisive probe

Use actual Ouro hidden dimensions, kernel shapes, and topology to measure local attention plus remote compute round-trip versus the matched native block. If the round-trip cannot beat native even under ideal ready-work supply, prune that layout before building an engine.

---

## 8. Search Area C — Recurrence-aware layouts and communication avoidance

**Priority:** high/medium. **Category:** novel composition or transformation of conventional parallelism.

### Question

Can a distributed tensor layout or partial result remain useful across the loop boundary, avoiding redistribution or synchronization repeated by a conventional TP/PP/CP composition?

Search for invariants in layouts, residual ownership, head/feature partitioning, and identical operator sequences. Consider phase-dependent layouts and a fixed schedule that reuses communication resources across r.

### Required derivation

For each claimed eliminated collective, state:

- Which baseline operation communicates what data?
- Which consumer originally needs the result?
- In what layout can that consumer operate instead?
- Where, if anywhere, does the deferred communication reappear?
- Why do normalization, attention softmax, and MLP nonlinearities remain correct?

Repeated weights do **not** justify reusing a previous loop's all-reduce output. Current activations differ. Carrying an unreduced partial result through a nonlinear operator is generally not equivalent; derive any exception rather than asserting it.

### Baselines and probes

Compare with ordinary layout propagation, communication–compute overlap, generic TP/CP hybrids, and vendor-supported collective fusion. An improvement caused solely by fixing an inefficient baseline layout is useful engineering but not yet the requested recurrence-specific architecture.

A hand-audited two-layer/two-loop distributed toy graph followed by a real Ouro block microbenchmark is preferable to a diagram without algebra.

---

## 9. Search Area D — Useful independent frontiers and verified branch execution

**Priority:** medium, high novelty risk. **Category:** possible new parallel work, but close prior art exists.

### Question

Does Ouro expose **useful work that can be started independently**, then committed or corrected cheaply, because of its shared recurrent states and repeated operator structure?

Potential directions include recurrence-aware candidate continuation, branch-state prefix sharing, using otherwise unavoidable verification to do useful next-step work, and branch/cache ownership protocols. Do not decide in advance that any of these are viable.

### Strict boundaries

- R1, R2, R3, and R4 of the same token are a nested dependency chain, not four independent branches.
- Running shallow and deep copies from the same input may just duplicate a prefix.
- Best-of-N reasoning improves search quality or aggregate samples; it does not by itself reduce latency for the original inference task.
- Ordinary speculative decoding with a new name is insufficient.
- LoopSpec and WaveFront Decoding are mandatory nearest-prior checks. Their existence must be established from primary sources and their actual device/runtime scope inspected. [S08–S09]
- The current target remains Ouro, not the dLLM token-filling order used by LoPA.

### Required branch contract

Specify branch construction, available inputs, shared prefix, immutable versus speculative KV, ownership, validation, correction, and commit. For an exact proposal, state precisely what is preserved: greedy output, sampling distribution, or only a conditional guarantee.

Measure accepted useful work per critical-path second, not raw speculative token count. Charge failed branches, verification, rollback, duplicate prefixes, and winner-cache synchronization. Compare against the strongest applicable existing Ouro self-speculative method, as well as native TP/DP.

A candidate may be reconsidered despite earlier generic/MoE speculation results only if it has a materially different recurrence-specific mechanism and measured cost argument.

---

## 10. Search Area E — Parallel-in-time numerical formulations

**Priority:** bounded high-risk exploration. **Category:** potentially new parallel axis.

### Question

Can the finite recurrent trajectory be solved or corrected in parallel, without learning a new model and without leaving the expensive exact computation on the critical path?

Read DEER/nonlinear sequential evaluation, scalable variants, ParaDiGMS, and related coarse/fine or fixed-point methods as mathematical analogies. They demonstrate formulations to inspect, not that Ouro R4 can be sped up. [S13–S15]

### What must be derived for Ouro

- The exact finite-trajectory equations, including per-depth historical KV and loop-boundary normalization.
- Whether the solver state also needs current-token KV or other mutable state.
- Initialization using only causal information; no hidden access to the sequential solution.
- The number of refinement rounds required for a meaningful accuracy or output criterion.
- Cost of Jacobians, Jacobian–vector products, linear solves, coarse propagation, and synchronization.
- What happens when the adaptive gate stops at R2 or R3.

### R4 break-even test

Four original recurrent passes leave little room to amortize initialization and correction. If the effective parallel procedure requires four expensive rounds plus overhead, concurrency alone is not a speedup.

Use a critical-path inequality, not a claim based on asymptotic sequence-length scaling:

`T_initialize + K * T_parallel_round + T_validate + T_commit < T_best_matched_baseline`.

The round cost includes all stage or recurrent evaluations performed. If using warm-up, preserve the earlier preference of zero or at most one **per-token exact recurrent warm-up**. Distinguish this from ordinary runtime kernel/graph warm-up, which is still required for benchmarking.

### Scope discipline

Start with a short mathematical and shape/cost screen. Do not spend hours implementing full Jacobians when their dimensions and lower-bound cost already defeat R4. Conversely, do not dismiss a novel low-cost formulation solely because generic stale-state prediction failed before.

A finite residual tolerance is not automatically proof of exact token outputs or preserved sampling. State the gap explicitly.

---

## 11. Search Area F — Structured component/residual work instead of whole hidden-state replacement

**Priority:** medium/low; must pass quality early. **Category:** an approximation-controlled distributed representation.

### Question

Is there a smaller, structurally constrained component of recurrent computation that can be reused, decomposed, or independently computed while preserving task quality, unlike the previously failed full-stage stale/delta replacement?

Possible objects to inspect include projection residuals, per-head attention contributions, exact base-plus-residual linear algebra, or a boundary representation with a provable error/control path. The object is not presumed redundant simply because cosine similarity is high.

### Non-negotiable qualifications

- `W(h + delta) = Wh + W delta` is exact for a linear W, but offers no speedup if W delta is another equally expensive dense multiplication.
- Softmax and normalization invalidate many naive base-plus-residual shortcuts.
- A full-rank residual may leave both compute and communication unchanged.
- Truncation, selective recomputation, or low-rank compression is approximate unless proved otherwise.
- No quantization or learned predictor is allowed in this campaign.
- FlashLoop and cross-loop KV/latent-attention work are mandatory nearby checks; verify whether their specific implementations require training. [S10–S11]

### Smallest useful test

Inject the proposed component transformation in a one-GPU numerical harness, preserve its real causal/KV consequences, and measure both internal error and free-running task quality before building a distributed implementation.

A surviving candidate must explain why distributing this representation changes the systems design. Single-GPU caching by itself is not enough.

---

## 12. Search Area G — Agent-originated alternatives and combinations

Reserve at least approximately one quarter of idea-generation time for alternatives not suggested above. The seed areas are a starting map, not an exhaustive menu or a mandate to rescue a favored idea.

Generate candidates from a structured design space:

| Axis | Options/questions |
|---|---|
| Work unit | Request, token, chunk, recurrent pass, physical layer, attention head, feature tile, branch, correction task |
| Shared property | Weight identity, known historical state, repeated operator sequence, shared prefix, representation relation |
| State placement | Request-local, depth-local, layer-local, head/sequence-sharded, selectively replicated, versioned scratch |
| Compute placement | Fixed worker, pooled worker, split operator, hybrid group, persistent kernel |
| Scheduling | Event-driven, recurrence-aware frontier, pipeline, bounded lookahead, explicit correction |
| Exactness | Exact graph, equivalent numerics, distribution-preserving validation, controlled approximation |
| Regime | One request, several ready tokens, long prefill, long decode, mixed arrival, memory-constrained |

Combine mechanisms only when they remove **different exposed costs**. Do not multiply their isolated speedups or count the same saved time twice.

For each unseeded candidate, write the best skeptical explanation of why it might be generic, impossible, quality-damaging, or too expensive. Then identify the smallest discriminating test.

---

## 13. What qualifies as a new parallelism claim?

### 13.1 Three categories, explicitly labeled

1. **Potentially new execution axis:** a concrete useful-work dimension or dependency transformation not obtained by ordinary partitioning of the existing graph.
2. **Recurrence-specific hybrid/transformation:** familiar parallelism primitives, but a distinct state/weight ownership or execution organization whose advantage depends materially on repeated shared operators.
3. **Generic optimization or parameter tuning:** an existing scheduling/layout/kernel technique with no demonstrated recurrence-specific mechanism.

Category 2 is acceptable as the user's second preference. Category 3 may be recorded as useful engineering, but should not be renamed “Loop Parallelism.”

Do not let the easiest-to-benchmark TP/PP variant crowd out the first-priority search. Search priority is not a claim that a risky new axis is more likely to succeed; preserve both a creative track and a practical hybrid track until their evidence distinguishes them.

### 13.2 Novelty and loop-specificity must be falsifiable

For every finalist answer:

- What is the exact concurrent unit of useful work?
- Which dependency is preserved, reorganized, deferred, or approximated?
- Why is the mechanism valuable in an R4 model with shared weights?
- What is the nearest generic method and nearest looped-model method?
- What would remain if the recurrence-specific property were removed?
- Could the same claimed gain be obtained by turning on a standard implementation option?
- Is the benefit actual critical-path reduction, memory capacity, throughput, or only reduced aggregate work?

### 13.3 Counterfactual controls

Use as appropriate, with limitations:

- **R=1 control:** same physical core without recurrent reuse; a systems ablation, not a quality-equivalent model comparison.
- **Logical unrolling with aliased weights:** same function, exposes whether the proposed scheduler merely recognizes a standard graph.
- **Physical copies with identical values:** duplicate W_l for each logical depth, retaining identical functions, to test the importance of physical sharing/residency. Ensure the backend does not silently deduplicate them.
- **Sharing-disabled implementation path:** same inputs/operator schedule with sharing-related optimization disabled.
- **CDB on/off or refill/no-refill:** use the separate factorial study when applicable; do not duplicate it unnecessarily.
- **Generic architecture matched to the proposed ownership/layout:** tests whether the gain comes from recurrence or generic engineering.

Physical copies alter memory pressure; report both peak memory and traffic. A materialized control does not prove global novelty. If a conventional unrolled runtime can exploit the same aliases and schedule, the claim needs narrowing.

A different independently trained dense model is not a clean causal control for weight-sharing effects.

---

## 14. Mandatory candidate card

Every serious candidate must contain the following fields, using `templates/candidate_card.md`:

1. ID, title, search family, and revision.
2. One-sentence mechanism without a marketing name.
3. Relevant Ouro property and its source or measured evidence.
4. Target model sizes, modes, stages, C, context, and P.
5. Exact baseline and the exposed cost being removed.
6. AS-IS operation sequence and TO-BE sequence.
7. Legal DAG and state-ownership table.
8. Source of concurrency; already-ready vs predicted/speculative work.
9. Exactness class and free-running quality plan.
10. Weight, activation, historical/current KV, and scratch memory.
11. Communication messages, bytes, dependencies, and topology.
12. Critical-path model and all added costs.
13. Nearest prior works and substantive distinction.
14. Relation to prior project NO-GO evidence.
15. Smallest falsification probe and stop criterion for this candidate.
16. Results, uncertainty, and current evidence class.
17. Both-size/both-mode/P=2/4/8 compatibility matrix.
18. Counterfactual/ablation demonstrating loop-specificity.
19. Implementation complexity and next-PoC estimate.
20. Verdict, unresolved risks, and the single next most informative action.

The agent may use Mermaid, ASCII, or another editable diagram format inside reports. Every arrow must correspond to a real dependency/message. A visually parallel diagram is not feasibility evidence.

---

## 15. Execution-DAG validation requirements

Before substantial GPU implementation, build a small validator for candidate schedules where practical.

Each node should have:

`node_id, request_id, token_or_chunk_id, r, l, operator, predecessors, weight_id, kv_reads, kv_writes, owner, layout, scratch_version, estimated_cost, resource_type`.

Validate:

- All predecessor outputs exist before use.
- Each required weight shard is resident or its transfer completes.
- Historical and current KV reads refer to the correct request/layer/depth/position/version.
- Losing or provisional branches cannot update committed KV.
- Per-device compute/copy/communication limits are respected.
- Adaptive exit handling cannot use an uncomputed gate decision as though it were online information.
- Per-request token generation remains causal.
- No artificial cross-DP barrier is introduced solely into a baseline.
- The candidate is executable for a small fixed graph and, where relevant, several different online exit sequences.

A simulator may intentionally include a clairvoyant oracle, but it must be a separately named non-deployable arm. A feasible schedule with guessed inputs is approximate/speculative, not exact merely because the dependencies were erased from the graph.

---

## 16. Cost accounting and headroom, without false oracles

### 16.1 Evidence labels

Every result must be assigned one of:

- `SOURCE_REPORTED`: a primary source's result, not our measurement.
- `USER_REPORTED_PRIOR`: historical project summary not revalidated here.
- `ANALYTIC_BOUND`: a derived bound with explicit assumptions.
- `MODEL_ESTIMATE`: calibrated but not guaranteed prediction.
- `SCHEDULE_HEURISTIC`: a feasible or clairvoyant heuristic, not an optimum.
- `GPU_MICROBENCH`: a measured component/operator result.
- `LIVE_E2E`: real end-to-end inference with the stated workload.
- `QUALITY_ONLY`: numerical/quality emulation without measured multi-GPU speed.
- `NOT_RUN` or a precise blocked status.

Do not call a greedy hindsight schedule a mathematical upper bound. For a fixed workload, a genuine lower bound on feasible latency can imply an upper bound on speedup; a fitted cost model generally cannot.

### 16.2 Critical-path rather than work-only accounting

For a candidate DAG, estimate latency from the longest feasible resource-constrained path, including:

- All mandatory attention/projection/MLP/norm/head work.
- Compute contention and copy/communication contention.
- Hidden-state and KV transfer, synchronization, queueing, and dispatch.
- Branch or solver setup, verification, correction, rollback, and commit.
- Startup/drain and per-token warm-up.
- Extra memory that changes batch capacity or causes paging.

Do not simply replace `sum(stage_time)` by `max(stage_time)` unless stage inputs are legally available and resource overlap is demonstrated.

For a component occupying fraction f and accelerated by k, a first-order Amdahl screen is:

`S_ideal = 1 / ((1 - f) + f/k)`.

This assumes unchanged other costs. Use it as a screen, not a measured result. Communication and implementation overhead normally reduce the benefit; changed batching or extra work can invalidate the simple decomposition.

### 16.3 Memory and traffic

Track separately:

- Unique model bytes.
- Physically replicated/sharded weight bytes.
- Logical KV values and physically allocated/aliased KV slots.
- Activation/scratch/graph/communication-buffer memory.
- Bytes actually read/written from HBM, if measured.
- Estimated traffic when hardware counters are unavailable.

For standard full depth-indexed KV, a useful capacity check is:

`KV_bytes = 2 * L * R * N * H_kv * d_head * bytes_per_element`.

Use actual config dimensions. Adaptive `last_exited` may still retain all depth slots; fewer executed loops do not automatically mean less allocated memory.

Never equate total KV footprint with attention's latency share or HBM bandwidth use. The previous project already showed why that inference is unsafe.

### 16.4 Topology-aware measurements

Inspect the actual H100 variant, memory size, GPU UUIDs, pairwise topology, P2P support, and link paths. Do not substitute advertised aggregate NVLink bandwidth for achieved unidirectional application throughput.

Measure relevant small and large messages with the actual communication primitive. Account for startup latency, all-to-all/reduction shape, overlap interference, synchronization, and warm-cache versus cold-cache conditions.

For P=8, provide optimistic, central, and conservative projections from actual P=2/4 behavior and a stated topology. Do not imply NVSwitch or free all-pairs access without confirming the future hardware.

### 16.5 Prediction-to-live gap

The previous hindsight study had a material model-to-live gap. Therefore:

- Validate cost estimates on held-out baseline shapes/workloads.
- Report relative prediction error and its scale compared with predicted gain.
- A 1.05× modeled gain with comparable cost-model error is unresolved.
- Re-measure uninstrumented E2E; fine-grained profiling can perturb the workload.
- Do not sum overlapping CUDA-event intervals as wall-clock latency.

---

## 17. Model and runtime matrix

### 17.1 Initial fixed checkpoints

- `ByteDance/Ouro-1.4B`
- `ByteDance/Ouro-2.6B`

Pin model/tokenizer/code revisions and inspect the loaded values, not only filenames. Base and Thinking rows must remain separate.

The inspected Base config family has hidden size 2048, attention/KV heads 16/16, and R4. Re-read actual configs before allocating state or partitioning heads. The 1.4B/2.6B distinction is not just scaling one matrix: the physical layer counts differ. [S01–S03]

### 17.2 Fixed-depth reference

Use exact R4 and its full depth-indexed KV semantics, with a validated native single-GPU reference. Include the best available correct TP2/TP4 implementations. Resolve or explicitly retain earlier numerical caveats rather than deleting the TP baseline because it is inconvenient.

### 17.3 Adaptive reference

Start by inspecting the already used CDB/Ouro implementation and selection manifest. The historical operating point is Base Ouro-1.4B, q=.5, minimum R2, maximum R4, `last_exited`, full-depth prefill plus adaptive decode.

Do not silently assume the identical settings are validated for 2.6B. Audit its original gate and runtime support. If adaptive 2.6B is unavailable or has unresolved implementation issues, report `ADAPTIVE_2P6B_UNVALIDATED` and do not claim the full target is met.

Do not use a replay-only exit path as a live online adaptive baseline. CDB's paper explicitly uses replay in its main performance experiments; the local implementation must be inspected independently. [S07]

### 17.4 Training-free CDB usage

Use the existing original gate path. If a CDB asynchronous mode requires newly trained lookahead/preloop weights, do not train them. Report the dependency and use an existing compatible original-gate path for primary comparisons, or mark the mode ineligible.

The absence of a single convenient serving package is an engineering limitation to record, not an excuse to fabricate measured comparisons.

---

## 18. Workloads, quality, and exactness

### 18.1 Quality comparisons are mode-matched

Primary comparisons:

- Proposed fixed R4 vs unchanged fixed R4.
- Proposed adaptive vs the **same** existing adaptive checkpoint/gate/q/KV policy.

Report adaptive-vs-fixed quality as context, but do not repeat the earlier mistake of automatically rejecting every scheduling/architecture study merely because the existing adaptive operating point is less accurate than fixed R4.

A new mechanism that changes hidden states, KV semantics, exit decisions, or sampling **does** create a new quality responsibility.

### 18.2 Classify exactness before evaluating

| Class | Required statement and tests |
|---|---|
| Exact computation graph | Same mathematical function and state transitions; compare numerics, token outputs, and gate decisions |
| Numerically reordered | Same intended function with floating-point order differences; quantify tolerances and free-running divergence |
| Validated speculative | State exact greedy/distribution guarantee and test verification plus KV commit |
| Approximate inference | No exactness claim; measure matched task quality and compute–latency tradeoff |

Bitwise equality is not guaranteed by algebraic equivalence across batch shapes. Conversely, similar accuracy does not prove a supposedly exact implementation correct.

### 18.3 Tiered quality evaluation

Do not spend the whole ten-hour campaign fully benchmarking twelve ideas.

1. **Correctness smoke:** a small fixed collection spanning short/long prompts and several generated steps; inspect logits/KV/depth/first divergence.
2. **Candidate quality pilot:** approximately 16–32 GSM8K and 16–32 MATH-500 items for a small number of candidates. A pilot can reveal collapse, not prove a one-point noninferiority margin.
3. **Confirmation when warranted:** a distinct held-out cohort, aiming for at least 128/domain for a finalist when resources permit.
4. **Publication-grade follow-up:** full GSM8K and MATH-500 or another explicitly justified sufficient protocol, outside the current time budget if needed.

Use free-running greedy generation for primary quality. Keep raw outputs, correct/N, prompt template, shots, tokenizer, EOS, cap, parser/judge revision, truncation, and failure cases. Do not treat teacher-forced logit similarity as task accuracy.

Ground-truth answers belong only in evaluation. Candidate initialization/branch selection cannot consult them. Held-out quality items may not be used for coefficient selection or prompt tuning.

For an approximate finalist, predeclare an acceptance margin against the mode-matched baseline before the final evaluation. Suggested discovery starting points are 1 pp GSM8K and 2 pp MATH-500, but these are research criteria, not published guarantees. If the sample is too small to resolve the chosen margin, label `QUALITY_UNRESOLVED`; do not call that a proof of failure or equivalence.

### 18.4 Benchmark workloads

Use a compact pre-registered matrix rather than a huge Cartesian product:

- Decode-first: prompt/context lengths around 1K, 4K, 8K; 16K only where supported and memory-safe.
- Short and long generated sequences: controlled-work timing around 128/512 generated steps, plus natural completion quality runs with a common adequate cap.
- Prefill: selected 1K/4K/8K prompts; chunk sizes chosen only if a candidate uses chunking.
- C is **system-wide concurrency**, not per-GPU batch size. Prioritize C=1/2/4/8; add C16/32 where needed.
- P=1 native, P=2 and P=4 true multi-GPU; P=8 projected unless separately authorized.

Include more than one task/domain to avoid a mechanism that works only on padding or one artificial input. Existing ShareGPT/Alpaca traces can support serving tests; record their origin and any replay semantics.

### 18.5 Timing must not reward changed output work invisibly

Maintain two complementary views:

- **Controlled-work execution:** matched shapes, token counts, and declared exit traces where necessary to isolate systems cost. Label replay/teacher forcing clearly.
- **Natural free-running inference:** actual online outputs, quality, total tokens, depth work, latency, and truncation.

A candidate producing fewer tokens, earlier EOS, different depths, or lower quality must not claim the entire elapsed-time difference as a scheduling speedup. Do not force continuation beyond EOS for quality runs.

---

## 19. Baselines, metrics, and speedup notation

### 19.1 Strong comparisons

For each finalist, select applicable validated baselines:

- Native 1GPU fixed or adaptive Ouro with competent kernels.
- TP2 / TP4 for single-request or low-C latency.
- DP2 / DP4 for multi-request throughput, using the existing local scheduler.
- CDB DP for dynamic-depth multi-request workloads.
- PP/chunked PP and CP only where relevant to the mechanism and workload.
- The nearest specialized existing method, such as a looped speculative or reuse method, if it implements the same claimed opportunity and is within the comparison scope.

Use matched precision, checkpoint, q/KV, output policy, and memory/admission budgets. The candidate cannot win by intentionally disabling graph capture, efficient kernels, or competent scheduling only in the baseline.

A closest prior that cannot run must be marked unavailable with reason; do not treat it as a defeated baseline.

### 19.2 Report all acceleration as ×

For the same metric and workload:

`S_1GPU = T_native_1GPU / T_candidate`

`S_incremental = T_best_same_P_baseline / T_candidate`

Report raw latency beside speedup. Values below 1.000× are slowdowns. For throughput, use `Q_candidate/Q_baseline`, separately labeled. Do not convert throughput increases using a latency formula.

Do not confuse:

- Single-request decode latency / ITL / TPOT.
- TTFT and prefill latency.
- Request E2E latency distribution.
- Finite-cohort makespan.
- Throughput and SLO-constrained goodput.

A four-GPU cohort gain near 4× can come from four replicas alone. It is not evidence of a new parallelism axis.

### 19.3 Measurement uncertainty

Use repeated paired trials with the same prompts/arrival traces and alternate or randomize method order. Record thermal/clocks/other-process activity rather than assuming a quiet server.

Bootstrap at the independent request/cohort/trial level appropriate to the metric. Do not treat thousands of generated tokens in one request as thousands of independent performance trials.

Compute intervals from paired speedup estimators and show their definition. A broad interval including 1× is inconclusive, not proof of no effect. Search many candidates on calibration data, then validate selected candidates independently to limit selection bias.

Report median/P95 where meaningful, sample count, uncertainty, and possible quality confounds. Do not silently pick the fastest run.

---

## 20. Required result tables

Use these as report templates, with real numbers only where measured.

### Table A — Candidate search funnel

| ID | Mechanism | New axis / hybrid / generic | Exactness | Prior-art overlap | Fixed headroom | Adaptive headroom | Evidence | Verdict |
|---|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Table B — Both-model, both-mode scope matrix

| Candidate | 1.4B fixed | 1.4B adaptive | 2.6B fixed | 2.6B adaptive | P2 | P4 | P8 | Low-C / prefill / throughput scope |
|---|---|---|---|---|---|---|---|---|
| ... | measured / modeled / unknown | ... | ... | ... | ... | ... | projected | ... |

### Table C — Paper-style performance and quality

Create separate panels by model, depth mode, and operating point. Do not mix an adaptive candidate with a fixed baseline in one normalized speedup column.

| Method | GPUs | GSM8K correct/N; Acc. | MATH correct/N; Acc. | Raw latency + exact metric | Speedup vs native 1GPU | Speedup vs strongest same-P | Evidence |
|---|---:|---|---|---|---:|---:|---|
| Native mode-matched Ouro | 1 | measured | measured | measured | 1.000× | — | LIVE_E2E |
| Strong conventional baseline | 2 or 4 | measured | measured | measured | measured | 1.000× | LIVE_E2E |
| Candidate | 2 or 4 | measured/NOT_RUN | measured/NOT_RUN | measured/modelled, separated | value or NOT_RUN | value or NOT_RUN | explicit |

A component microbenchmark belongs in a separate table, not in the LIVE_E2E latency column.

### Table D — Headroom decomposition

| Candidate | Baseline exposed time | Removed/overlapped cost | Added compute | Added communication | Added synchronization/state | Critical-path estimate | Live validation/error |
|---|---:|---:|---:|---:|---:|---:|---|
| ... | ... | ... | ... | ... | ... | ... | ... |

### Table E — Novelty and nearest work

| Candidate | Nearest looped prior | Nearest generic prior | Shared mechanism | Actual new element | Evidence that element matters | Remaining uncertainty |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

### Table F — Negative results and reopening rules

| Rejected candidate | Decisive evidence | Why rejected | Scope of conclusion | Specific new fact needed to reopen |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

Fill every candidate's status even if the result is negative. Filling a table does not mean inventing a speedup when no live run occurred.

---

## 21. Final ranking and decision rules

### 21.1 Separate novelty from evidence strength

Score or discuss separately:

- Recurrence-specific mechanism.
- Distinction from prior work.
- Dependency/ownership feasibility.
- Quality and numerical integrity.
- Incremental same-P benefit.
- Both-size/both-mode robustness.
- Low-concurrency relevance.
- Engineering burden and follow-up risk.

A high novelty score cannot compensate for an impossible DAG. A large modeled speedup cannot compensate for unvalidated output semantics. A small but reproducible gain can justify a focused follow-up when the architectural mechanism is genuinely interesting.

### 21.2 Candidate verdicts

- `PROMISING_NEW_AXIS`: distinct mechanism, credible legal/quality-controlled execution, and supporting headroom; not yet a publication claim.
- `PROMISING_RECURRENCE_HYBRID`: strong second-priority candidate with a demonstrated recurrence-specific advantage.
- `CONDITIONAL_MECHANISM`: plausible idea but missing a decisive measurement, quality result, or baseline.
- `MODE_SPECIFIC` / `SIZE_SPECIFIC`: useful partial scope, not the full user target.
- `GENERIC_ONLY`: benefit explained by conventional execution/optimization.
- `DEPENDENCY_INVALID`: requires unavailable data or illegal state use.
- `QUALITY_FAIL`: new mechanism damages matched-baseline quality beyond the declared tolerance.
- `NO_NET_HEADROOM`: added cost removes the expected benefit in tested conditions.
- `INFRASTRUCTURE_BLOCKED`: cannot establish empirical feasibility under current resources.

### 21.3 Campaign verdicts

- `SHORTLIST_READY_FOR_TARGETED_POC`
- `MECHANISMS_FOUND_BUT_EVIDENCE_INCOMPLETE`
- `NO_QUALIFYING_CANDIDATE_IN_SEARCHED_SPACE`
- `PAUSED_RUNTIME_LIMIT`
- `PAUSED_RESOURCE_OR_ACCESS_LIMIT`

Do not claim “no new parallelism can exist for Ouro.” A negative dossier is scoped to explored transformations and evidence. Do not promise a qualifying candidate simply because the campaign lasted ten hours.

### 21.4 Finalist proposal requirements

For up to three finalists provide:

- A compact proposed architecture and one editable execution figure.
- One paragraph of motivation based on actual evidence.
- An exact novelty statement and its nearest comparisons.
- A small implementable next PoC, with files/functions to modify when identified.
- Expected gain expressed as a measured result, calibrated range, or explicitly speculative hypothesis.
- The single most likely failure mode and a cheap test to discover it.
- What must still be verified on 2.6B, adaptive mode, P4, or future P8.

Do not write a confident paper abstract claiming achieved speedups from an untested idea.

---

## 22. Resource isolation and concurrent-experiment safety

The CDB factorial experiment is running. This campaign must be non-interfering.

- Inspect active processes, GPU UUIDs/memory, and the existing experiment's ownership/lock conventions before a benchmark.
- Authorized pool is physical 4/5/6/7, not a claim that all four are currently idle.
- Use two explicitly available devices for P2 and four only when all four are released for this campaign. Record `CUDA_VISIBLE_DEVICES` and visible-rank mapping.
- Do not touch physical 0/1/2/3, reset GPUs, kill burners, terminate users' jobs, or change power/clock settings to make room.
- If no GPUs are free, continue the CPU/web/theory portions; leave measurements blocked rather than biasing the ongoing study.
- Do not run heavy shared-disk scans, multiple simultaneous checkpoint downloads, or unbounded activation dumps.
- Use existing checkpoint caches read-only; avoid duplicate large downloads.
- Set documented memory/token limits and stop individual runaway/OOM probes without altering the overall research goal.
- Profile GPU counters only with available permissions. Record `COUNTERS_UNAVAILABLE` rather than escalating privileges or inferring counters from utilization.
- Use a separate environment/worktree/workspace and uniquely named logs. Never reset, clean, checkout over, or commit unrelated dirty/staged files.
- No global dependency upgrades. Inspect code before executing downloaded scripts or `trust_remote_code` implementations.
- Do not expose private reports, model outputs, access tokens, or internal paths in public searches.
- No paid API evaluation, paid cluster provisioning, external posting, or new service account actions without explicit authorization.
- The requested output is local research artifacts. Git commits, pushes, or PRs require existing explicit authorization and must contain only this campaign's work.

This specification does not grant eight-GPU access or permission to preempt the experiment already running.

---

## 23. Files and reproducibility contract

Suggested layout inside the isolated workspace:

```text
poc_ouro_parallelism_deep_discovery/
  reports/
    spec_snapshot.md
    scope_and_prior_evidence.md
    source_verification.md
    ouro_semantics.md
    search_map.md
    candidate_register.md
    dependency_red_team.md
    cost_and_headroom.md
    quality_and_exactness.md
    model_mode_scaling_matrix.md
    novelty_matrix.md
    negative_results.md
    progress_log.md
    next_actions.md
    final_report.md
    final_summary_ko.md
  candidates/
    C01_<slug>.md
    C02_<slug>.md
    ...
  artifacts/
    source_registry.jsonl
    claim_ledger.jsonl
    search_log.jsonl
    time_ledger.jsonl
    candidate_status.jsonl
    environment_manifest.json
    model_runtime_manifest.json
    experiment_manifest.jsonl
    raw_metrics.csv
    quality_outputs.jsonl
    dag_checks/
    traces/
  scripts/
    verify_dag.py
    benchmark_commands.sh
    analysis_and_plots.py
  figures/
  research_state.json
```

Do not create empty code files and imply their tests ran. Where no implementation is warranted, state that explicitly.

Each measurement must link to a command, timestamp, model/runtime revision, GPU set, input IDs, metric definition, and raw output. Each key external claim must link to the primary source and a locator. Each negative verdict must link to evidence.

The final Korean summary should explain the best ideas simply, distinguish measured findings from conjecture, and state what was not verified. The detailed dossier and candidate cards remain in English.

---

## 24. Seed source registry — independently recheck during execution

The following primary sources were accessed while preparing this specification. They are a seed map, not a complete novelty review. “Metadata checked” does not mean every theorem, experiment, or code path was audited.

| ID | Source | Verified/useful role | Follow-up requirement |
|---|---|---|---|
| S01 | *Scaling Latent Reasoning via Looped Language Models*, arXiv:2510.25741 | Ouro model family and repeated shared core | Read architecture, inference, and limitations; pin version |
| S02 | ByteDance Base Ouro checkpoint configs | 1.4B vs 2.6B dimensions, R4, dtype, gate defaults | Pin loaded revisions; inspect Thinking separately |
| S03 | ByteDance `modeling_ouro.py` | Actual shared-stack loop, normalization, gate, cache implementation | Audit full path and actual local runtime |
| S04 | *LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding*, arXiv:2512.16229 | Token-filling-order exploration plus branch-parallel runtime; Figures 2/3 inspected | Study algorithm and cache commit; no assumption of direct AR transfer |
| S05 | *DistriFusion*, arXiv:2402.19481 | Displaced patch parallelism; model-specific reuse/communication tradeoff | Extract transferable requirements, not image-model guarantees |
| S06 | *AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising*, arXiv:2406.06911 | Asynchronous model partitioning using neighboring-step information | Distinguish from unrelated same-name papers; revisit old project failure |
| S07 | *Depth-adaptive Inference ... Continuous Depth Batching*, arXiv:2608.09444, and author repo | Local loop-step batching; paper explicitly excludes production multi-GPU features and uses exit replay in main performance tests | Inspect used code, no-refill/refill, online vs replay, original vs trained extra gates |
| S08 | *LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers*, arXiv:2609.17184 | Direct nearby recurrent draft/verification prior; primary abstract checked | Full algorithm/device/quality/cost audit before proposing similar work |
| S09 | *WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models*, arXiv:2609.23033 | Direct neighboring prior; identity checked | Full method/runtime audit |
| S10 | *FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates*, arXiv:2609.29812 | Reuse/update neighboring prior; identity checked | Determine exact training/approximation and device scope |
| S11 | *Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers*, arXiv:2607.15456 | Cross-loop state-compression neighboring prior; identity checked | Determine whether each method/checkpoint requires training |
| S12 | *Parallel Loop Transformer for Efficient Test-Time Computation Scaling*, arXiv:2510.24824 | Architecture that changes cross-loop/token execution | Determine training changes; not automatically drop-in Ouro inference |
| S13 | *Parallelizing non-linear sequential models over the sequence length*, arXiv:2309.12252 | DEER numerical parallel evaluation | Assess applicability to finite R4 and mutable KV |
| S14 | *Towards Scalable and Stable Parallelization of Nonlinear RNNs*, arXiv:2407.19115 | Numerical method scaling/stability | Read costs and assumptions, not asymptotic marketing |
| S15 | *Parallel Sampling of Diffusion Models*, arXiv:2305.16317 | ParaDiGMS/Picard parallel-time analogy | Many-step diffusion gains are not Ouro R4 evidence |
| S16 | *TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models*, arXiv:2102.07988 | Generic causal token/chunk PP nearby method | Training paper; identify inference-transferable mechanism precisely |
| S17 | *Enabling Dynamic Computation in Looped LMs*, arXiv:2610.09013 | Dynamic KV/runtime semantics neighboring work; identity checked | Reverify code, protocol, and comparability; do not inherit old score claims |
| S18 | NVIDIA CUDA Programming Guide, Nsight Compute Profiling Guide, NVSHMEM documentation | Real hardware/runtime/counter semantics | Read docs matching installed versions and actual hardware |
| S19 | *Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference*, arXiv:2604.26294 | Generic layout/parallelism composition lead | Full paper read required before using specific claims |

### Reproducible primary URLs

```text
S01 https://arxiv.org/abs/2510.25741
S02 https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/config.json
S02 https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/config.json
S03 https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/modeling_ouro.py
S03 https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/modeling_ouro.py
S04 https://arxiv.org/pdf/2512.16229
S05 https://arxiv.org/abs/2402.19481
S05 https://github.com/mit-han-lab/distrifuser
S06 https://arxiv.org/abs/2406.06911
S07 https://arxiv.org/html/2608.09444v2
S07 https://github.com/kschwethelm/continuous-depth-batching
S08 https://arxiv.org/abs/2609.17184
S09 https://arxiv.org/abs/2609.23033
S10 https://arxiv.org/abs/2609.29812
S11 https://arxiv.org/abs/2607.15456
S12 https://arxiv.org/abs/2510.24824
S13 https://arxiv.org/abs/2309.12252
S14 https://arxiv.org/abs/2407.19115
S15 https://arxiv.org/abs/2305.16317
S15 https://github.com/AndyShih12/paradigms
S16 https://arxiv.org/abs/2102.07988
S17 https://arxiv.org/abs/2610.09013
S18 https://docs.nvidia.com/cuda/cuda-c-programming-guide/
S18 https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
S18 https://docs.nvidia.com/nvshmem/api/index.html
S19 https://arxiv.org/abs/2604.26294
```

Do not reproduce large sections of source papers in the dossier. Paraphrase relevant mechanisms with attribution and preserve enough locators for verification.

---

## 25. Final completion checklist

- [ ] Substantive duration logged honestly; at least ten active campaign hours, or an explicit resumable pause.
- [ ] Ongoing CDB factorial study preserved and not contended with.
- [ ] Both Base Ouro sizes audited; fixed and adaptive modes distinguished.
- [ ] Training-free and BF16 constraints honored.
- [ ] Primary-source novelty review beyond only searching the model name.
- [ ] At least three agent-originated candidates outside the seed families, or an explained shortfall.
- [ ] Every finalist has a legal DAG, ownership ledger, and complete added-cost accounting.
- [ ] Past NO-GO findings addressed, not erased or needlessly rerun.
- [ ] No fabricated free KV migration, future activation, branch independence, or global DP barrier.
- [ ] No conflation of HBM footprint, HBM traffic, FLOPs, and latency.
- [ ] No conflation of predicted speedup, component timing, replay, and live E2E.
- [ ] Quality assessed against the correct mode-matched baseline.
- [ ] All speedups expressed in × with raw metric and baseline definition.
- [ ] P8 clearly projected/not run unless genuinely authorized and measured.
- [ ] Novelty and systems evidence assessed separately.
- [ ] Ranked shortlist or scoped negative result, with concrete next-PoC instructions.

**End of specification.** The task is to discover a defensible architecture opportunity in Ouro, not to prove that a favorite analogy must work, and not to abandon the campaign after the first failed hypothesis.
