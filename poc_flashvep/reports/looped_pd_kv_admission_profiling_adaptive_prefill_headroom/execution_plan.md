# Execution plan and frozen gates

Status: PREPARED. The prior campaign is closed.
This namespace has a separate controller and preservation contract.
No previous deadline or terminal state applies to this campaign.

## Order

1. Replay the original Ouro 8192/64 configuration.
2. Test D token budget 128 at unchanged KV pool and arrivals.
3. Test the previously qualified larger D pool at that budget.
4. Bound pending reservations through supported source draining limits.
5. Qualify Huginn 3500/32 without profiling.
6. Test a P-only short capture with one, four, eight, and 24 PD requests.
7. Use the requested lighter primary lengths only after these original-length controls.
8. Measure matched independent capacity and freeze calibration SLOs for new lengths.
9. Search a bounded QPS grid. Confirm eligible candidates with 128 requests and three independent starts.
10. Test small real D-side Prefill work only after the relevant prerequisite gates pass.

Every changed runtime geometry requires exact token, valid KV, hidden-state, and full-depth checks.
Failed runs retain their namespace and log.
No failed namespace is overwritten on resume.

## Hypotheses

- Oversized D workspaces reduce allocatable HBM. A smaller D budget should reduce allocation without changing outputs.
- Pending D reservations consume capacity before active Decode. Bounded staging should reduce that occupancy if it is causal.
- Broad profiling causes a progress failure. A P-only bounded capture should retain forward progress if scope is causal.
- If the short capture also stalls, worker stacks and progress records must identify the blocked boundary before further scaling.

## Prospective acceptance rules

P kernel busy must reach 80% in an attributed device window.
P service must reach 85% of the strongest matched finite P reference.
Actual compute-ready backlog must grow without a dominant D-credit, transfer, or host stall.
D request TPOT p95 must remain below 90% of its frozen Decode limit.
The separate frozen token-ITL limit also applies.
Actual proposed work must fit D KV and workspace capacity.
All conditions must overlap in the same start and time window.
Three independent positive starts are required for a strict positive regime.

A profile can represent a matched clean run only within the inherited 5% wall-difference threshold.
Kernel unions, graph-active spans, CUDA-event spans, and host intervals remain separate metrics.
Missing profiling is `NOT_MEASURED`, not zero utilization.
Internal parity is not original-HF fidelity or downstream accuracy.

## Resource and publication rules

Clean timing uses one all-GPU quiet lane.
Only verified owned burner sessions can stop.
The prior data and unrelated Git index remain unchanged.
Publication uses a temporary index and never force-pushes.
Approved burners resume after the final measurement and resource audit.
