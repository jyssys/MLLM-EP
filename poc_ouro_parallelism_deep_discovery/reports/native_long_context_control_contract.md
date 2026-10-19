# Native C1 long-context control

This control tests the remaining capacity assumption behind C19.
It changes the operating regime, not the lossless-KV method.
The [source audit](long_context_capacity_source_audit.md) supports the configured positions and exact KV minimum.
It does not establish long-context execution or task quality.

The protocol freezes N=1024/8192/16384/32768 before any result.
Both Base sizes use fixed R4, BF16, native asynchronous scheduling, and TP1/2/4 where safe.
The workload uses two frozen patterns with truncation or repetition.
It is a controlled hardware workload, not natural long-context reasoning.
The native singleton allowed-token sampler supplies 128 matched output tokens.

Every cell uses the same 32896-position model and scheduler cap.
No RoPE scaling or unsupported-length override enters the configuration.
The engine uses C1 and graph capture size 1.
Its aggregate native KV pool is 52 GiB.
TP1/TP2/TP4 divide that pool into 52/26/13 GiB per rank.
The pool has enough structural capacity for the audited largest request.
It does not guarantee activation or workspace safety.
An OOM or configuration failure remains evidence, not a reason to force execution.

Each pattern and context receives three short warm-ups and three clean repetitions.
The clean interval encloses the native `generate` API call.
Native request metrics remain separate from that API wall.
Memory inspection and numerical capture occur after clean timing.
The capture retains logits, hidden states, and sampled actual KV at N32768.
Forced output identity alone cannot establish finite model values.
Independent analysis must check those captured values before using the result.

The runner preserves physical UUID and 4L cache-slot assertions.
The existing guard rejects foreign contexts and restores only scoped owned burners.
The control creates no serving scheduler, CP adapter, replica policy, or candidate quality result.
