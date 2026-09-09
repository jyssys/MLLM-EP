# FastPP: the causal intervention that rejects the proxy story

Native mechanism measurements join 3,950 invocations across four PP ranks;
rank0 is slowest in 3,885. A contiguous measured-cost minimax partition suggests
~23–26% stage-max reduction. That proxy assumes cost moves with a layer and
balanced local spans shorten actual requests.

Intervention: same Qwen3 model, greedy scheduler, arrival trace, full-trace
warmup and KV capacity; equal 12/12/12/12 versus existing 8/12/14/14 partition.
Three randomized independent restart pairs, no performance timing hooks.
The opt-in KV owned-layer-count compatibility fix was regression-tested;
failed startups are excluded.

Actual median paired request E2E reductions: **-14.91% steady, -11.98% bursty**.
All three pairs regress. Restart-median bootstrap intervals respectively
[-17.54,-6.93]% and [-16.42,-11.74]%; small-n descriptive intervals.

This is causal evidence **against this cost-proxy partition**. It is not a
unique CPU/attention/communication attribution, proof against every partition,
or an EP/modality interaction. An incorrect proxy is not a positive paper.
