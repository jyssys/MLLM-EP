# Action semantics and evidence boundary

## KEEP

`KEEP` is the validated stock Qwen3-30B-A3B EP8 route. The existing
read-only hook records eight local expert histograms and CUDA-event spans for
DeepEP dispatch, TritonExperts, and DeepEP combine. No route, placement, or
model output is changed.

## TEMP_BALANCE

The experiment wrapper follows the public Capacity-Aware-MoE score-ranked
capacity-selection idea: it over-selects candidates from the router logits,
clips per-expert candidates, then selects top-k among survivors. This is a
bounded reference-aligned wrapper, not a claim that the public package is
integrated into vLLM. The real Qwen EP8 run completed on all eight GPUs.

The measured run is not a valid quality-preserving production action:
capacity clipping produced invalid route slots (median 12.16% of original
slots), and 15/24 measured driver outputs changed versus KEEP. Therefore the
lower routed-MoE stage sum is reported as a route/quality-cost diagnostic, not
as a usable latency gain.

## PERSIST_BALANCE

The public EPLB `rebalance_experts` implementation was run on the exact
48-layer × 128-expert histogram reconstructed from the KEEP trace. Diagnostic
plans used 136 and 144 physical experts (17/18 per GPU) and preserved the
16-expert-per-GPU baseline only as the comparison. No new placement was
installed into vLLM, so there is no end-to-end PERSIST latency claim.

A one-shot actual Qwen expert-weight broadcast was measured on the EP group.
The two tensors have shape `[1536,2048]` and `[2048,768]`, BF16, 9,437,184
bytes total. This is a lower-level migration primitive measurement, not a
full EPLB migration schedule.

## Oracle rule

The raw A0/A1 timing winner is retained for diagnosis. The safe action oracle
rejects TEMP when it drops any route slots and charges PERSIST with the
measured migration cost. This prevents a route-dropping implementation from
being selected as an apparent optimization. Under that preregistered rule,
KEEP wins every captured invocation/layer row; an RL policy is therefore not
trained.
