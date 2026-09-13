# Variable-block decoder correctness

Status: static B=8/16/32/64/128 and completed-block mixed schedules passed
execution sanity; bounded quality and fixed-schedule negative controls are
complete.

## Static semantics checked

- completed prefix remains in KV and visible;
- current open block is evaluated bidirectionally;
- future blocks are absent from the current model input/cache;
- position IDs use the absolute `block_start + arange(B)`;
- KV writeback uses the exact completed block width;
- EOS only terminates after a block completes;
- all tested widths are divisible by EP4 source partitioning.

The dInfer tests already exercise non-32 widths in generic block generation, but
the benchmark entry point had hidden B32 overrides.  Those entry-point bugs were
fixed in the isolated worktree.

## Matched-extent correction

The original benchmark floors `prompt_length + gen_len` to the tested B.  Thus
the same nominal `gen_len=128` produced different absolute endpoints.  This made
the initial B128 GSM8K score appear to collapse.  With common absolute endpoints
(GSM8K 256, HumanEval 384), the HumanEval B128 score recovered from 10/32 to
18/32, equal to B32.  Native and matched-extent results are never pooled.

## Mixed schedule implementation

`LLADA_BLOCK_SCHEDULE` accepts positive comma-separated widths.  The next width
is selected only after the current block is fully decoded.  Each sequence keeps
its own schedule index; sequences with different current widths are placed in
separate physical waves.  A terminal block may be shortened only to the exact
remaining extent.  No open block is resized.

The `[32]` and `[64]` negative controls reproduced the corresponding bounded-set
accuracy and NFE.  They also reproduced exact answer strings and generation
lengths for 32/32 samples on both tasks.  An initial `[16,64,32,16]` schedule preserved GSM8K 5/8 but
required 331 physical forwards versus 106 for fixed B32: independent sequences
diverged in block-completion time, split across width-specific waves, and exposed
a large dynamic-schedule fragmentation tax.  This is an observed system cost,
not counted as an oracle benefit.
