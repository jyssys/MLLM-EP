# Native attention contract

## Model and physical layout

The checkpoint is `inclusionAI/LLaDA2.0-mini`, revision
`dad945cac317da394b390f82c7b40691d8a881ed`. It is not mini-preview.
The loaded configuration permits 32,768 positions. RoPE uses theta600000,
head dimension128, and rotary dimension64. No extrapolation is enabled.

The native no-A2A baseline uses TP4, EP1, DP1, PP1, and CP1.
Physical GPUs4/5/6/7 map to logical ranks0/1/2/3. Each rank owns four query
heads and one KV head. Each head owner stores the complete sequence.
This is head parallelism, not context parallelism. TP4 and independent CP4
would require additional ranks unless an explicitly supported hybrid exists.

All20 attention layers are full-attention layers. The configuration contains
`sliding_window=4096`, but `use_sliding_window=false`. The actual attention
implementation does not apply a sliding mask. Nominal fields alone cannot
establish an effective window.

## Native cache lifetime

`BlockRunner.prefill` computes the prompt cache. Completed blocks update it
through native cross-block refresh. Within a block, regular refinement reads
the existing cache without publishing new prefix values.

The SGLang cache shape is `[layer,2,batch,KV-head,key-position,head-dimension]`.
Every layer receives only its local KV-head shard. Current block Q/K/V are
computed from fresh hidden states on every forward. Q/K normalization and
absolute-position RoPE precede attention.

`LLaDA2Attention.forward` reads cached K/V and uses out-of-place
`slice_scatter` to replace the final query-length positions. It does not
recompute cached prefix projections. It then expands the local KV head for
four query heads. These copies and query-dependent attention remain recurring
costs. Their logical bytes are not measured HBM traffic.

## Padding and masks: important native limitation

`KVCache` adds64 positions, aligns to128, then rounds capacity to a power
of two. For example, a512-token prefix allocates1024 cache positions.
Regular refinement passes no attention mask. It therefore also attends to
native zero-padding slots. Current K/V are written into the last32 slots,
not necessarily logical positions512–543.

This behavior is pre-existing native code. Its comment acknowledges possible
accuracy effects. This PoC neither repairs the mask nor treats removal of
padding as a new optimization. Effective key lengths and logical positions
must both appear in traces. Native observer parity does not prove equivalence
to an independently implemented HF attention path.

Initial prefill uses the native block-causal boolean mask. Within each block,
attention is bidirectional. Cross-block refresh uses its native two-block mask.
The runner pads that mask to physical cache capacity. Regular refinement is
unmasked. Accepted positions inside the current block remain fresh canvas
positions, not immutable prefix positions.

## Communication and publication

The QKV projection is column-sharded. The attention output is row-sharded.
Native layer communication restores the dense hidden representation through
TP collectives. The loaded implementation uses a replicated `nn.Linear`
LM head, not a vocabulary-sharded serving head. No prefix sequence
sharding or distributed softmax merge exists in this baseline.

CUDA events measure intervals on one rank's current stream. Cross-rank
absolute timestamps are not aligned. Analysis must select one cumulative
critical rank per request, not sum component maxima from different ranks.

## Context ladder and validation scope

The experiment freezes P512/1024/2048/4096/8192/16384 and output64, block32.
P32768 is excluded because prefix plus output exceeds trained context.
The old4096 runtime limit is not treated as the checkpoint limit.
Physical execution validates regular-query key capacities through32768.
Full-prefill graph capture fails at P16384. Native eager prefill with graph
refinement succeeds there, without changing the mask or attention math.

The full native prefill limit is explicitly configured to32768. This uses
an existing parameter, not a new scheduler. It avoids treating the old
128-token prefill setting as an immutable optimized baseline. All requests
preserve native block masks, sampler threshold0.9, BF16, and temperature0.

Native eager TP4 completes a deterministic three-repeat sanity probe.
Native graph inference preserves that probe's tokens and NFE. Stage A records
48 completed request-length episodes, with exact observer parity across ranks.
The sampled prefix tensors remain equal across three refinements. Canvas K/V
change. Process teardown is a separate limitation, not a hidden timing gain.
