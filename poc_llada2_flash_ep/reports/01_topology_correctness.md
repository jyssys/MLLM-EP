# Topology correctness and quality gate

## Exact execution contracts

| Path | Dense layers | Routed experts | Shared expert | Communication |
|---|---|---|---|---|
| Official TP4 | TP4 | all 256 experts, tensor-sharded across TP4 | TP4 | tensor-parallel reductions |
| Bounded sparse EP4 | TP4 | 64 complete experts/rank | replicated, applied to disjoint source rows | DeepEP normal dispatch/combine plus exact source-row gather |

Both paths run BF16 expert kernels. DP=1 and sequence parallelism is disabled.
The EP path is not inferred from a flag: ownership, remote dispatch, local IDs,
local execution and reverse combine were each checked at runtime.

## Substrate defects found and repaired

Two official-loader defects had to be isolated before performance work:

1. a filtered rank-local state dictionary was computed but the unfiltered
   shard was retained, causing full-checkpoint host staging per rank;
2. unquantized routed weights were fused only when global expert 0 appeared in
   the rank-local dictionary, leaving ranks 1--3 with zero routed weights.

The second defect was caught by an independent layer-1 checkpoint replay: the
first EP implementation matched only rank 0's contribution and had 15.39%
routed-output relative L2. After changing the gate to “rank-local dictionary is
nonempty,” the relative L2 against the independent reference became 0.373%.

## Numerical checks

| Check | Result | Interpretation |
|---|---:|---|
| DeepEP identity dispatch/combine | median rel-L2 0.113% | expected BF16 transport/reduction order |
| Layer 0 TP4 vs EP4 | bitwise equal | dense path unchanged |
| Layer 1 hidden | rel-L2 0.5319%, cosine ~1.0000 | corrected routed path |
| Layer 31 hidden | rel-L2 4.1559%, cosine 0.999146 | accumulated BF16 order difference |
| Initial 8-answer topology pair | 8/8 prediction strings identical | direct bounded trajectory check |
| GSM8K 8-sample, generation 128 | TP4 4/8; EP4 4/8 | benchmark score preserved |

For the 128-token pair, only 2/8 surface strings were byte-identical and NFE
was 185 versus 183. The extracted numeric answers and aggregate GSM8K score
were identical. This confirms that topology-dependent BF16 ordering can alter
the threshold-decoder trajectory; comparisons therefore report both clean
request wall and forward-normalized wall. It also motivates the larger bounded
GSM8K/HumanEval quality pair that is run separately from profiling.

## Gate status

**PASS for bounded performance characterization.** The sparse path performs
real owner-rank expert work and preserves the bounded task metric. It is not a
bitwise-exact trajectory, so no topology result is presented as exact token
agreement. Any method that changes semantic work remains subject to a separate
quality gate.
