# Plan-path reproduction, not a successor

Official H100 Qwen1.5 MoE EP4 forward passed exact FP16 HF comparison (82 tokens).
Its default factory provides no search result, so this does not demonstrate the
paper's nano-batching or compute/communication overlap.

Source constraints at pinned `915790ea862d1ddd52a8871282c1eb5be88f1391`:

- Optional splitter reads the former `is_auto_search_enabled` spelling.
- Qwen graph leaves operation categories unset.
- `FusedMoE` has no `copy_nano`; inherited method raises NotImplementedError.
- Published dense plan JSON nests `operations`; runtime expects a flat mapping.
- Original copy methods for all-reduce allocate separate official NCCL IDs for
  nano partitions. Routing and per-rank expert placement remain unchanged.

The opt-in `nano_plan_adapter.py` uses a minimal fixed-plan compatibility port:
same operation graph, native splitter, buffer planner and executor; same weights,
routing and numerical kernels. Its FusedMoE clone follows the existing official
LayerNorm/GEMM weight-sharing contract. A fixed decode cohort activates the plan
after ordinary prefill; any subsequent shape change is rejected. FFN nano
partitions use existing full-SM COMP and MEM streams; no new kernel or policy.

This is an activation/correctness diagnostic, NOT faithful reproduction of the
paper's optimal auto-searched plan, NOT a dynamic serving port, and NOT performance
evidence until live outputs and actual stream overlap are checked. Parts=1 is the
same-operation multi-stream control; parts=2/4 expose the native nano path.

Further plan search requires measured per-operation costs and a compatible
search input. Default-versus-this-plan timing cannot establish a failure of the
published optimum. A future plan-regret calculation must retain that distinction.

An independent `--cuda-graph` control captures only after the official smoke
reaches its stable four-request decode cohort, then replays the same native
executor graph. Both graph and nano variants must pass HF token comparison;
the first capture step is excluded from steady-state timing. Double-buffer
lookahead is not enabled or claimed by this control.

## Live activation result, 18:53–18:59 KST

Eager, CUDA graph, one-part separate-stream, two-part eager and two-part graph
all match the same 82 HF reference tokens. Two-part activation required preserving
the original sigmoid in the official `Activation.copy_nano` factory; see the
RED/GREEN regression and compatibility log. This repair is not research novelty.

Descriptive fixed-four-request median host step intervals (one restart each,
initial shape/capture steps and final Terminate barrier excluded) are about
33.24 ms eager, 5.65 ms graph, 32.84 ms one-part eager, 50.45 ms two-part eager,
and 7.93 ms two-part graph. These are mechanism-smoke diagnostics, not request
regret, not independent repetitions, and not evidence that the paper's optimal
plan fails. CUDA graph is therefore the necessary baseline for the next screen.

The next fixed-cohort probe uses real FineWeb content, two exact-shape prefill
warmups and actual per-request first/completion timestamps. Numerical workers
remain official. Plan setup and capture on the target path are charged, not
silently subtracted. This is still a simultaneous-cohort experiment, not a claim
of a complete continuous-batching NanoFlow port. All cross-plan output IDs must
be checked before accepting the resulting performance rows.
