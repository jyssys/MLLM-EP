# Source discovery pass 1 — instrumentation and EP execution

Date: 2026-09-06/07. Local runtime: vLLM 0.20.0 from
`/home/esjung/anaconda3/envs/flashvep-poc`, DeepEP HT, NCCL 2.28.9.

This pass deliberately read execution and prior instrumentation paths before
choosing a new systems idea.

## Unexpected implementation facts

1. The inherited online route observer synchronizes every timed layer CUDA
   event and copies top-k IDs to CPU inside the forward path. This can
   serialize the otherwise asynchronous DeepEP path and perturb scheduler
   launch cadence.
2. `DeepEPHTPrepareAndFinalize` uses async dispatch/finalize and explicit
   event dependencies; observation must therefore remain nonblocking until
   an event is already complete.
3. OpenAI serving accepts `X-data-parallel-rank`, providing a supported way to
   perturb the DP placement of an identical request multiset without changing
   scheduler or model code.
4. Existing route and resource-atlas traces are valuable for hypothesis
   generation, but absolute wall times from their synchronous hooks cannot be
   treated as stock-runtime performance.

## Generated hypotheses

H01, H02, H03, H04, H05, H07, H20, and H30.

## Live falsification status

The new deferred-event observer costs +3.27% median E2E versus no hook and
+1.62% TTFT in the H01 smoke workload. This is small enough for randomized
within-server effects, but absolute cross-server effects below 5% require a
no-hook replication.
