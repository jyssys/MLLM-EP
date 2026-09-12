# Deep PoC Specification
# Independent Training-Free Latency Reduction for Multi-GPU EP MoE Diffusion LLMs

## 0. Research Objective

The primary goal is to discover an independent, training-free inference method for vanilla MoE diffusion LLMs under multi-GPU Expert Parallelism (EP).

Target paper framing:

> Training-Free Low-Latency Inference for Expert-Parallel MoE Diffusion Language Models

TEAM is secondary: motivation + adaptation after the vanilla method works.
Epoch is the principal distributed related-work baseline conceptually.

The new method should ideally reduce vanilla dLLM-MoE EP latency independently of TEAM, REFLEX, or DES.

---

## 1. Hardware

Use only physical GPUs 0,1,2,3.

- Single GPU: GPU 0
- EP2: GPU 0,1
- EP4: GPU 0,1,2,3

Hard-fail if another GPU is selected.

Record `nvidia-smi -L` and `nvidia-smi topo -m`.

---

## 2. Primary Experimental Substrate

### Immediate primary model
Use the faithful `SDAR-30B-A3B-Chat-b32` substrate already validated through the TEAM study.

Reasons:
- official execution exists;
- single-GPU behavior is trustworthy;
- true EP2/EP4 paths exist;
- TEAM adaptation can later be tested on the same model.

SDAR fits on one H100, so EP is not required for capacity here. Treat it as a controlled scaling substrate.

### Secondary model
If a correct execution contract is available without major engineering:
- `LLaDA-MoE-7B-A1B-Instruct`, or
- a larger LLaDA2 MoE model.

Do not reuse prior quality-invalid REFLEX/DES ports as quality evidence.

---

## 3. Stage 0 — Fair Vanilla Single vs EP2 vs EP4

This is mandatory before method development.

Historical numbers are not an apples-to-apples raw scaling comparison because single-GPU and bounded EP runs used different generation budgets/harnesses.

Rerun vanilla SDAR under exactly matched conditions.

### Matched variables
Keep identical:
- checkpoint
- prompt/sample
- seed
- decoding algorithm
- block size
- threshold
- generation length
- output budget
- dtype
- fused expert kernel
- warmup
- timing boundary
- harness
- CUDA synchronization

Only topology changes:
- Single
- EP2
- EP4

### Primary datasets
Use bounded identical subsets from:
- GSM8K
- HumanEval

Prefer 8–16 fixed prompts and multiple restarts if practical.

### Optional workload sweep
After batch/request=1 is stable, optionally test:
- 1
- 4
- 8

Do not mix throughput and single-request latency claims.

### Required metrics
- request latency
- NFE
- whole MoE
- attention
- router
- route preparation
- dispatch
- expert
- combine
- other decoder
- token-expert assignments
- remote assignments
- remote bytes
- destination-rank fanout
- rank-wise load
- collective count
- message shapes
- expert kernel shapes
- GPU utilization / idle gaps

### Required table

| Topology | Accuracy | Latency | vs Single | MoE share | Dispatch | Expert | Combine | Remote bytes | Fanout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | | | 1.00x | | N/A | | N/A | N/A | N/A |
| EP2 | | | | | | | | | |
| EP4 | | | | | | | | | |

Do not assume EP4 is slower. Measure it.

---

## 4. Stage 1 — Vanilla EP4 Temporal / Refinement Structure

Profile vanilla EP4 across denoising iterations.

Collect:
- active masked tokens
- accepted tokens
- expert IDs
- router scores
- token→expert assignments
- destination ranks
- fanout
- remote bytes
- hidden similarity
- expert-input similarity
- expert-output similarity
- route-set similarity
- critical rank
- stage latency

For lag Δ = 1,2,4,8 compute:
- route Jaccard
- expert-load cosine
- rank-load cosine
- same critical rank probability
- same top-k expert probability
- hidden cosine / relative-L2
- expert-output cosine / relative-L2

Condition on:
- early/middle/late denoising
- high/low confidence
- accepted/still-masked positions
- layer depth

The purpose is to find dLLM-specific structure that can remove or localize global EP work.

---

## 5. Core Research Targets

Prefer methods that reduce:
1. frequency of global EP execution
2. number of positions needing global EP
3. number of ranks needed per position
4. remote expert work
5. communication on the critical path
6. repeated work across denoising iterations

Prefer changing/removing work over rearranging identical work.

---

## 6. Candidate A — Rank-Local Refinement with Global Verification (PRIMARY)

Working name: `RLR-GV`.

### Idea
Use each EP rank's existing expert shard as a cheap local diffusion view/drafter.

Example EP4 with 128 experts:
- GPU0: E0–31
- GPU1: E32–63
- GPU2: E64–95
- GPU3: E96–127

For rank-local draft:
- run dense/attention locally
- restrict eligible MoE experts to local experts
- local top-k + renormalization
- produce token proposals/confidence
- no remote expert dispatch during draft

Four ranks produce four independent local proposals in parallel.

### Consensus signals
- top-1 agreement
- top-k overlap
- confidence agreement
- entropy
- logit margin
- vote count
- proposal union

High agreement → draft proposal candidate.
Low agreement → exact global EP immediately.

### Global verification
Do not blindly commit local output.

Use exact global EP as verifier:
1. local ranks propose token identities/positions;
2. one exact global forward computes target logits;
3. accept only proposals satisfying the baseline target rule;
4. reject/fallback otherwise.

Investigate whether one global verifier forward can validate multiple local draft steps or multiple positions.

Goal:
`many cheap local proposal steps + fewer exact global EP steps`.

### Speculative nature
Yes, this is speculative-like:
`cheap draft → exact target verification`.

But novelty target is:
- no separately trained drafter;
- use physical EP shards as draft views;
- parallel rank-local diffusion proposals;
- reduce global EP refinement frequency.

### Offline oracle
Using saved vanilla EP4 states, measure:
- per-rank top1 match to global target
- union-of-4 hit rate
- 4/4 agreement precision
- 3/4 agreement precision
- 2/4 agreement precision
- accepted-position overlap
- confidence correlation

Measure especially:
- P(global match | 4/4 agreement)
- P(global match | 3/4 agreement)

### Gate
Promising if high-consensus cases provide useful coverage with near-exact precision and projected E2E >=10%.
Strong if projected removal of global EP forwards yields >=15% E2E.

Kill if projected E2E <5% or local proposals are too inaccurate/rare.

---

## 7. Candidate B — Verification-Coalesced Multi-Step Local Drafting

Extension of Candidate A.

Instead of:
`local draft → global verify → local draft → global verify`

attempt:
`local step 1 → local step 2 → ... → local step K → one global verification`

Sweep:
- K=1
- K=2
- K=4
- K=8

Measure:
- proposal survival
- verifier acceptance
- global NFE reduction
- draft overhead
- final quality

This candidate can be much larger than communication-only optimizations if one verifier can validate multiple proposed commitments.

---

## 8. Candidate C — Block-Hot Expert Replica Cache (EXACT)

Working name: `BHERC`.

### Idea
Exploit temporal expert persistence across a diffusion block.

Instead of repeatedly sending tokens to the same remote experts:
1. observe/predict hot `(source rank, remote expert)` pairs;
2. replicate a small hot set under an HBM budget;
3. copy once per block or infrequently;
4. execute future requests on identical local replicas.

Exact outputs because weights are identical.

### Oracle
Measure:
- remote assignment coverage
- bytes coverage
- dispatch/combine time removable
- expert weight size
- copy time
- reuse count
- HBM overhead

Compute actual copy-amortized request gain.

### Prior-art risk
High. Audit generic expert replication, MoonEP-like replication, TIDE, Epoch, placement systems.

Kill quickly if oracle/novelty is weak.

---

## 9. Candidate D — Elastic EP Degree Across Denoising

Working name: `E2P`.

### Hypothesis
Large early workload may favor EP4; small later workload may favor EP2 or EP1 due to communication overhead.

### Mechanism
If replicas/memory allow, enable:
- EP4
- EP2
- EP1

Select degree using current workload:
- active positions
- expected assignments
- fanout
- predicted latency

### First oracle
Benchmark identical observed MoE shapes under EP1/EP2/EP4.
Build a perfect per-iteration topology oracle.

Require >=10% request-level headroom before implementation.

---

## 10. Candidate E — Confidence-Gated Global Expert Scope

Working name: `CGES`.

Use:
- local rank
- 2-rank subgroup
- full EP4

depending on confidence/uncertainty.

Signals:
- token confidence
- confidence margin
- routing entropy
- router mass by rank
- temporal route stability
- rank-local consensus

Prefer verification/conservative gating.

Prior-art audit:
- TEAM
- REFLEX
- DICE token conditional communication
- topology-aware routing

Only keep if novelty is conditional dLLM EP scope, not generic pruning.

---

## 11. Candidate F — Temporal Delta Dispatch

Working name: `TDD`.

For same logical `(position, expert)` across adjacent iterations, test sending:
`Δ = h_t - h_(t-1)`
instead of full `h_t`.

Explore:
- FP8 / lower precision delta
- sparse delta
- error-bounded quantization
- block compression

Measure:
- relative-L2 delta
- sparsity
- compression ratio
- compress/decompress overhead
- actual A2A latency change

Kill if bytes do not translate into real latency.

Audit DICE and communication-compression prior art.

---

## 12. Candidate G — Layer-Gated Global Synchronization

Working name: `LGS`.

Some MoE layers may need exact global experts more than others.

Test:
- exact global EP on sensitive layers
- local/stale/cached approximation on insensitive layers
- periodic exact refresh

For each layer estimate:
`quality sensitivity / EP latency cost`.

Prior-art risk is very high due DICE/Epoch.
Treat as exploratory.

---

## 13. Candidate H — Cross-Iteration / Draft-Verify Overlap

Capture Nsight Systems.

Look for truly independent work:
- local draft attention while global verifier A2A runs
- one draft branch compute while another communication runs
- cross-request independent work if serving

Build perfect overlap oracle first.

Kill if <5% E2E.
Serious only if >=10%.

Audit DICE/generic overlap.

---

## 14. Candidate I — Rank-Coherent Near-Tie Routing

When router scores are nearly tied, prefer expert sets touching fewer ranks.

Sweep allowable router-score sacrifice:
- 0%
- 0.1%
- 0.5%
- 1%

Measure:
- accuracy
- fanout
- remote bytes
- latency

Prior-art risk is very high. Only keep if dLLM temporal structure creates a distinct opportunity.

---

## 15. Candidate Tournament

Before deep implementation, compute a perfect or structural oracle.

| Candidate | Exact? | Training-free? | Perfect E2E oracle | Quality risk | Prior-art risk | Complexity | Decision |
|---|---|---|---:|---|---|---|---|
| A RLR-GV | verifier-exact possible | yes | | | | | |
| B Multi-step drafting | verifier-exact possible | yes | | | | | |
| C BHERC | exact | yes | | low | high | | |
| D E2P | exact with replicas | yes | | low | med/high | | |
| E CGES | approx/verified | yes | | medium | high | | |
| F TDD | approximate | yes | | medium | med/high | | |
| G LGS | approximate | yes | | high | high | | |
| H overlap | exact if dependency-safe | yes | | low | high | | |
| I rank-coherent | approximate | yes | | medium | high | | |

### Gates
- <5%: KILL
- 5–8%: WEAK
- 8–12%: PROMISING
- 12–20%: STRONG
- >20%: VERY STRONG

Prefer >=12% oracle and >=8–10% causal live prototype for a paper direction.

---

## 16. TEAM Adaptation — Secondary Only

Only after a vanilla method works.

Evaluate:
- Vanilla
- TEAM
- Ours
- TEAM + Ours

Ideal:
- Ours improves vanilla independently
- TEAM + Ours recovers/improves TEAM under EP4

This proves it is not a TEAM-specific patch.

---

## 17. Epoch Boundary

Conceptual separation:

**Ours**
- reduce when/where exact global EP refinement is needed, or remove other dLLM-specific EP work.

**Epoch**
- execute remaining required fresh/live global EP work efficiently using block compilation / sparse fresh worklists.

Run sensitivity for strongest candidate with communication/runtime cost multipliers:
- 1.0
- 0.75
- 0.5
- 0.25

If benefit disappears with a modestly faster backend, mark durability risk.

---

## 18. Speculative / Self-Speculative Prior Art

For Candidate A/B audit:
- classic speculative decoding
- self-speculative MoE
- DraftExpert
- diffusion speculative decoding
- TEAM
- PoE-Bridge / DLM drafting
- rank-local/shard-local draft systems

Novelty cannot be merely:
`use fewer experts as draft`.

It should be:
`use existing physical EP rank partitions as parallel dLLM refinement views and condition exact global EP verification on their proposals/consensus`.

Kill if exact prior art exists.

---

## 19. DICE Boundary

DICE targets image diffusion MoE and communication staleness/overlap/conditional synchronization.

Therefore simple:
- stale overlap
- selective synchronization
- token conditional communication

is risky.

Language-dLLM novelty should exploit:
- masked-token acceptance
- refinement confidence
- block decoding
- rank-local proposal verification

---

## 20. Correctness / Quality

For exact candidates:
- numerical equivalence
- generation equality/benchmark quality

For speculative verified candidates:
- precise verifier acceptance rule
- bounded/no quality loss

For approximate candidates:
- full quality/latency Pareto

Benchmarks:
- GSM8K
- HumanEval
- MBPP if feasible

Use identical samples across baselines.

---

## 21. Production Controls

Do not use Python expert loops for headline latency.

Use fused expert kernels.

Use the most production-like correct EP A2A backend available.
If reference NCCL is the only correct option, label it and run faster-backend sensitivity.

Clean timing must be separate from instrumentation.

---

## 22. Required Reports

Create:
- `reports/00_environment.md`
- `reports/01_vanilla_single_ep2_ep4.md`
- `reports/02_vanilla_ep4_temporal_structure.md`
- `reports/03_rank_local_draft_oracle.md`
- `reports/04_multistep_draft_oracle.md`
- `reports/05_block_hot_replica_oracle.md`
- `reports/06_elastic_ep_oracle.md`
- `reports/07_conditional_scope_oracle.md`
- `reports/08_temporal_delta_oracle.md`
- `reports/09_layer_sync_overlap_oracles.md`
- `reports/10_candidate_tournament.md`
- `reports/11_prior_art_audit.md`
- `reports/12_best_candidate_live_poc.md`
- `reports/13_team_adaptation.md`
- `reports/final_decision.md`

---

## 23. Required Plots

1. Vanilla latency vs EP degree
2. Vanilla stage breakdown single/EP2/EP4
3. Remote traffic vs EP degree
4. Fanout vs denoising iteration
5. Temporal route similarity vs lag
6. Hidden/expert-input similarity vs lag
7. Rank-local proposal hit rate
8. Consensus precision vs coverage
9. Global-EP invocation reduction oracle
10. Hot-replica coverage vs HBM budget
11. Best EP degree vs workload shape
12. Candidate E2E oracle comparison
13. Quality-latency Pareto
14. Faster-runtime sensitivity
15. Vanilla / TEAM / Ours / TEAM+Ours

---

## 24. Detailed Rank-Local Draft Tests

Perform before full implementation.

### A0 Local expert restriction
For each rank/layer:
- restrict to local experts
- local top-k
- renormalize

### A1 Local forward quality
Measure:
- hidden cosine
- logits cosine
- top1 agreement
- top5 overlap
- confidence correlation

### A2 Four-view ensemble
Measure:
- union contains global target
- majority vote match
- weighted vote match
- agreement predicts target match

### A3 Token acceptance
For each baseline-accepted token:
- same token locally proposed?
- rank vote count?
- can it be verified/committed?

### A4 Multi-step oracle
Simulate 1/2/4 local draft steps and estimate how many exact global iterations can be coalesced.

### A5 E2E projection
Include:
- local draft compute
- proposal exchange
- verifier cost
- fallback
- reduced global EP count

Do not count communication savings alone.

---

## 25. Exact Replica Economics

For each diffusion block:
- remote-demand histogram
- hot source/expert pairs
- persistence
- saved bytes/time
- expert weight size
- copy time
- reuse count

Net gain:
`saved remote critical-path cost - copy overhead - memory penalty`

Require request-level gain after amortization.

---

## 26. Elastic Topology Crossover

For real observed shapes, benchmark:
- EP1
- EP2
- EP4

Find crossover as a function of:
- rows
- assignments
- fanout

Replay vanilla trace with perfect per-iteration topology selection.

---

## 27. Ablation

For strongest candidate separate:
- algorithmic effect
- communication effect
- kernel utilization effect
- synchronization effect

Do not attribute everything to communication automatically.

---

## 28. Final Labels

- `NO-GO`: no independent vanilla candidate >=8% credible E2E oracle
- `CHARACTERIZATION-ONLY`: strong pathology but no method
- `HOLD-CANDIDATE`: >=8% credible oracle + plausible novelty
- `STRONG-CANDIDATE`: >=12% credible oracle + live >=8% E2E gain
- `VERY-STRONG-CANDIDATE`: >=15% live E2E by removing whole global refinements, bounded quality, plus TEAM composition or second-model validation

---

## 29. Final Questions

1. What is fair vanilla single→EP2→EP4 scaling?
2. Where does vanilla EP4 spend time?
3. Which temporal/refinement properties exist in vanilla dLLM-MoE?
4. Can rank-local shards predict global refinement decisions?
5. Can global EP refinement frequency be reduced?
6. Can block-hot exact replicas remove enough remote work?
7. Is adaptive EP degree useful?
8. Is conditional EP scope useful beyond Epoch/DICE/TEAM?
9. Which candidate has strongest E2E oracle?
10. Does it survive a faster Epoch/DeepEP-like runtime?
11. Does it improve vanilla independently of TEAM?
12. Does TEAM+Ours recover TEAM's EP4 reversal?
13. Is second-model validation needed?

---

## 30. Final Instruction

Do not force a communication-aware controller.

Search for a genuinely large independent systems opportunity.

Workflow:

`fair vanilla scaling → vanilla EP4 profiling → dLLM-specific structure → multiple independent candidate oracles → prior-art attack → implement strongest only → TEAM adaptation second`

The strongest desired direction changes:

`every refinement → full global EP`

toward:

`cheap/local/partial refinement → exact global EP only when necessary`

while remaining training-free and quality-safe.
