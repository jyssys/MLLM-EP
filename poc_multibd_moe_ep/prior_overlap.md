# Prior boundary

This experiment changes only the official Block Buffer capacity (`buffer_size`,
R=1/2/3/4), with one pinned non-DMax MultiTF checkpoint. It introduces no
decoding method, expert approximation, or admission controller.

## MBD-LMs

[MBD-LMs](https://arxiv.org/html/2606.29215v1) already introduces MultiTF,
concurrent block refinement, fixed-shape Block Buffers, and decode/store
overlap. Its Section 4.5 explicitly distinguishes processed buffer rows from
useful committed tokens and analyzes why larger TPF need not translate
proportionally into TPS. Therefore neither MultiBD nor conversion loss alone
is a new finding. The additional question here requires measured physical
EP4 transport/owner execution, working-set/rank pressure, and a same-weights
R sweep. TP-only comparison is desirable to separate generic additional
forward work from EP-specific cost.

The [official reproduction branch](https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms)
is pinned to `9d1a04e794351f18039b3234a10e8b07e15bcbbf`.
The [non-DMax coding checkpoint](https://huggingface.co/SJTU-DENG-Lab/MBD-Code-LLaDA2-mini-16B)
is pinned to `83434493ea9f026cbc139d681d7ae04ffefe052f`.
Upstream does not expose a supported EP configuration: its Config rejects
A2A backends. Included experimental EP modules are not evidence of validated
end-to-end EP support.

## Epoch

[Epoch](https://arxiv.org/html/2609.09748v1) combines a diffusion-block expert
plan, stable-decoded reuse, liveness-sharded work, and compact fresh-row EP
dispatch/combine with full-shard restoration. This PoC performs none of those
changes and does not credit dead-token removal as its contribution. The
earlier 5.77x live-only accounting is not measured Epoch performance.

## TEAM

[TEAM](https://arxiv.org/html/2602.08404) uses delayed decoded-token caching,
hot-token speculative exploration, and restricted cold-token expert activation.
We preserve fresh official routing, top-k and weights for all processed rows;
we neither reproduce TEAM nor claim token/expert reduction.

## BlockServe

[BlockServe](https://arxiv.org/html/2607.08930) addresses inter-request
convergence heterogeneity through block-grained eviction/refill, mixed-state
execution, and token-budget admission. The primary experiment here is B1
intra-request block concurrency and physical EP cost. No continuous-batching
or admission method is implemented. A positive result would motivate only a
subsequent small EP-aware running-set control experiment.
