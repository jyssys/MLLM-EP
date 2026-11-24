# MLLM EPD checkpoint v11

Status: ACTIVE and INCOMPLETE. This checkpoint publishes completed N3 interference evidence.
All three prospective probe blocks completed on the same E/P/D GPU group.
Every E128/E512/D128/D512 Natural and Replay condition failed the complete frozen protection gate.
Some metrics improved. A point improvement does not erase other failed gates or wide confidence intervals.
All conditions executed additional real prefix work. Their discarded samples do not count toward client task throughput.
Four of24 probe cohorts had client ITL p99 below34ms but failed the request-level95% decode gate.
That gate requires every ITL and request TPOT to pass. A pooled p99 does not replace it.
All request/token failure combinations remain in the CSV records. No failed start was removed.
Worker-level GPU competition versus host or client delivery was not measured for these clean cohorts.
Fixed period order and numerical/batch variation limit a unique runtime-cause attribution.
No encode-safe or decode-safe additional-prefill success is claimed.
E/helper allocator and shared device memory snapshots are published separately from logical weight and KV bytes.
The helper used8GiB KV. Native D used42GiB. This is not equal KV capacity or maximum encoder-batch validation.
Imported-KV suffix correctness and cost belong to a separate study, not these no-import prefix probes.
At this checkpoint the suffix, Native transport policy and actual batch studies remain live or queued.
CPU regression:223PASS. Full task quality and high-load prefill-heavy feasibility remain unresolved.
Final reports, final resource closeout and final campaign publication remain pending.
