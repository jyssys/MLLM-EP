# Rank-Fanout-Aware EP Communication PoC

## Decision

**NO-GO. Do not implement dynamic per-layer/per-step AGRS↔DeepEP-HT
switching for this Qwen3-VL top-k=8 / EP4 regime.**

The mechanistic premise is real: expert sparsity does not imply rank-level
communication sparsity, and controlled fanout alone can reverse the AGRS vs
DeepEP-HT winner. The serving premise fails. Natural Qwen3-VL routing is
already concentrated near full EP-rank fanout, stabilized real-route replay
usually has one dominant backend, and the optimistic zero-cost per-layer
oracle projects only **0.197% median and 3.068% maximum TTFT improvement**.
This is far below the required 12–15% gate.

No switching mechanism was implemented. Reported dynamic results are
perfect oracles, not measured method speedups.

## Evidence boundary

| Label | Meaning in this report |
|---|---|
| Measured | Fresh execution on physical H100 GPUs 4–7 |
| Exact replay | Production communication primitives and Qwen expert weights on a bit-identical captured route/input |
| Oracle | Offline zero-cost choice between measured backend costs |
| Projected TTFT | Measured removable MoE milliseconds divided by clean measured TTFT; not an integrated method result |

The original working tree was dirty, so all changes were made in the isolated
worktree `/home/esjung/MLLM-EP-rankfanout`. The starting
`jyssys/MLLM-EP` HEAD was
`e228b44cec1f5ffa32953e52540086423f84f33f` (`flashvep: add offline
wavefront quick PoC`). No file under `poc_flashvep/` was changed.

## Runtime and path validity

- Model: Qwen3-VL-30B-A3B-Instruct, BF16, 128 experts, top-k=8.
- Runtime: vLLM `0.20.0+cu129`, torch `2.11.0+cu129`, DeepEP distribution
  `1.2.1+73b6ea4`, NCCL package `2.28.9`.
- Topology: TP2 / DP2 / effective EP4, linear placement, 32 experts/rank,
  TritonExperts, eager, DBO off, EPLB off.
- Hardware: four H100 80GB GPUs, physical indices 4–7, full NV18 links.

G0 is **PASS**. Verification used source hashes plus live execution hooks:

| Backend | Source contract | Calls observed in full-model execution |
|---|---|---|
| AGRS | dispatch `all_gatherv`; combine `reduce_scatterv` | `AgRsAll2AllManager.dispatch/all_gatherv`, `AgRsAll2AllManager.combine/reduce_scatterv`, `agrs.prepare` |
| DeepEP HT | layout + `Buffer.dispatch`; `Buffer.combine` | `deep_ep.Buffer.get_dispatch_layout`, `deep_ep.Buffer.dispatch`, `deep_ep.Buffer.combine`, `deepep_ht.prepare_async` |

All four placement dumps agreed on effective EP4 ownership. Full-model clean
outputs agreed exactly for all 150 aligned output pairs. Exact replay had
minimum cosine `0.99999565` and maximum relative L2 `0.2951%`, passing the
specified `0.9999` / `0.5%` gates.

DeepEP LL was optional and was skipped because this exact installed stack had
already failed two bounded initializations, including a reduced token budget,
at its `nvshmem_qp_depth` assertion. The required HT comparison is complete.

## Controlled synthetic causal result

The route generator fixes, for every M and F condition:

- M and top-k=8;
- exactly `8M` assignments;
- exactly `2M` assignments per rank;
- exactly `M/16` assignments per expert;
- all 128 experts active;
- identical expert and rank-load histograms;
- exact unique destination fanout F1/F2/F3/F4.

The benchmark used actual Qwen layer-24 expert weights, vLLM Triton fused
experts, and the production AGRS/DeepEP communication primitives. Each cell
has 90 measurements from 30 repetitions in each of three independent process
restarts.

Selected pooled medians (positive gap means the listed winner is faster):

| M/source | Fanout | Winner | Winner gap |
|---:|---:|---|---:|
| 1,024 | 1 | DeepEP HT | 31.97% |
| 1,024 | 4 | DeepEP HT | 7.98% |
| 2,048 | 1 | DeepEP HT | 16.89% |
| 2,048 | 4 | AGRS | 15.70% |
| 4,096 | 1 | DeepEP HT | 27.81% |
| 4,096 | 4 | AGRS | 23.37% |
| 8,192 | 1 | DeepEP HT | 42.87% |
| 8,192 | 2 | DeepEP HT | 14.69% |
| 8,192 | 3 | AGRS | 6.78% |
| 8,192 | 4 | AGRS | 25.36% |

At M=4,096, F1 was DeepEP-favored and F4 AGRS-favored in all three restarts.
At M=8,192, F1/F2 were unanimously DeepEP-favored and F3/F4 unanimously
AGRS-favored. Smaller M cells had more runtime noise and are not used for the
causal headline. G1 is **PASS**: destination-rank fanout causally changes the
relative communication regime when all specified work/load controls are
fixed.

## Natural Qwen3-VL fanout

Routes were captured directly from `BaseRouter.select_experts`, rather than
from the stock return buffer (which returned zeros on this path). The data
cover three restarts, five workloads, all 48 MoE layers, four source EP ranks,
and 8,640 source-rank/layer observations.

| Statistic | Natural result |
|---|---:|
| invocation mean | 3.6276 / 4 |
| invocation p10 / p50 / p90 | 3.4907 / 3.6367 / 3.7542 |
| invocation min / max | 3.2305 / 3.8867 |
| observations with mean F≥3.5 | 88.33% |
| token-weighted F1 | 0.0095% |
| token-weighted F2 | 1.6482% |
| token-weighted F3 | 33.8737% |
| token-weighted F4 | 64.4686% |

Workload-weighted mean fanout was 3.616–3.632 for image-448, image-1024,
text-512, text-2K, and text-8K. The workload type did not open a low-fanout
regime. H1 is supported descriptively, but G2 is **FAIL** for dynamic use:
almost every token is F3/F4 and invocation means occupy a narrow high-density
band.

## Clean static serving

Five independent engine-restart pairs used the same five requests/workloads,
one common warmup, and three clean measured iterations. DeepEP HT won the sum
of workload TTFT medians in four restarts; AGRS won one. A descriptive
per-workload choice improved over the best static backend by a median 0% and
maximum 3.62%.

These full-model TTFT comparisons carry a **CAUTION** label: AGRS exhibited
large restart-state drift (sum of workload medians 599.8–1,071.5 ms) while
DeepEP was much tighter (630.6–676.1 ms). They establish that both static
engines execute correctly, but are not used as the causal layer oracle.

Moreover, although final generated tokens agreed exactly, 470/480 profiled
layer keys had different later-layer route hashes across separately executed
backends. Tiny numerical ordering differences compound through the model.
Directly taking a minimum across those non-identical routes would be invalid.
This motivated the strict replay below.

## Bit-identical real-route replay and dynamic oracle

The replay uses one real Qwen route snapshot for each of five workloads and
all 48 layers, identical seeded BF16 hidden inputs and bit-identical captured
routes across backends, Qwen layer-24 expert weights, production communication
primitives, ten measurements per shape, and five process restarts. It contains
24,000 measured rows and 1,200 strictly aligned backend/layer pairs. The route
geometry is real; the replay hidden tensor is a controlled input, not a claim
that full hidden activations were captured.

Restart 1 exposed a cold AGRS dispatch confound: aggregate dispatch was
152.379 ms versus 32.465–35.456 ms in stabilized restarts 3/5. It remains in
raw data but is excluded from the stabilized headline. Restart 4, where AGRS
expert/combine time also rose, is retained as an adverse state and supplies
the most optimistic stabilized oracle.

Spec-defined global oracles:

| Restart | Best static | O1 per-layer MoE | O2 per-step MoE | O1 projected TTFT | O2 projected TTFT |
|---:|---|---:|---:|---:|---:|
| 2 | AGRS | 0.564% | 0.000% | 0.157% | 0.000% |
| 3 | AGRS | 0.330% | 0.000% | 0.118% | 0.000% |
| 4 | DeepEP HT | 7.402% | 5.642% | 3.068% | 2.338% |
| 5 | AGRS | 1.029% | 0.000% | 0.236% | 0.000% |
| stabilized median | — | **0.797%** | **0.000%** | **0.197%** | **0.000%** |

O3 equals O2 in this matrix because each workload is one synchronized
prefill step and the two DP requests must share one collective/backend.

On the held-out selector test, restart 2 calibrated threshold
`mean_fanout >= 3.335938 → AGRS`; restarts 3–5 were held out. The zero-cost
perfect layer oracle was 3.475% of replay MoE time, but the threshold changed
best-static cost by **−0.040%** and recovered **−1.14%** of the oracle. G3 is
**FAIL**: winner diversity is dominated by restart state rather than stable
natural fanout. G4 is **FAIL**: even the most favorable retained O1 TTFT
projection is only 3.068%.

## Causal interpretation

The combined evidence resolves the apparent contradiction:

1. With M, top-k, expert histogram, active experts, and per-rank load fixed,
   low fanout reduces sparse A2A traffic enough for DeepEP HT to win at large
   M, while F3/F4 lets AGRS win. This is a real operator regime transition.
2. Real top-k=8/EP4 Qwen3-VL routes almost never visit the controlled F1/F2
   regimes. Expert sparsity therefore coexists with effectively dense
   rank-level communication.
3. Within the narrow natural F3/F4 band, backend winner changes are neither
   robust across restarts nor predictable from fanout.
4. Consequently, a zero-cost selector has at most 3.068% projected TTFT
   headroom in the retained worst state, before paying dual-workspace,
   selection, layout-conversion, graph, or switching overhead.

The operator result may still be useful as a static backend-selection warning
for a different model/topology whose natural routes span F1–F4. It is not a
method opportunity for the tested Qwen3-VL EP4 regime.

## Decision gates

| Gate | Result | Reason |
|---|---|---|
| G0 runtime validity | PASS | source + live calls + placement + correctness |
| G1 synthetic causality | PASS | unanimous crossover at M=4K/8K |
| G2 natural fanout diversity | FAIL | 98.34% token mass at F3/F4; narrow workload means |
| G3 real winner diversity | FAIL | stabilized static dominance; threshold recovers none |
| G4 TTFT oracle | FAIL | O1 median 0.197%, max 3.068% |

## Reproduction and artifacts

- Working contract: `poc_rankfanout/SPEC.md`
- Environment/source path: `poc_rankfanout/reports/evidence/`
- Code and tests: `poc_rankfanout/rankfanout/`, `scripts/`, `tests/`
- Raw result root:
  `poc_rankfanout/results/rank_fanout_ep_communication_poc_20260910_192050/`
- Key derived tables: `synthetic/analysis/`, `real/fanout_analysis/`,
  `real/clean_analysis/`, and `analysis/` under that result root.
- Figures A1–A3 and B1–B8 are stored beside their derived tables.

The result root is intentionally ignored by default because it is 1.2 GiB;
the compact derived evidence and figures used by this report are force-added
to the research commit. Tests cover fanout metrics, exact route controls,
trace schemas, and oracle arithmetic.

> The rank-fanout-aware dynamic communication direction does not provide sufficient E2E headroom or backend regime diversity. Do not implement per-layer/per-step AGRS↔A2A switching.
