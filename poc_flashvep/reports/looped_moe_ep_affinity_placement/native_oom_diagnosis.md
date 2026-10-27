# Native generation OOM diagnosis

## Preserved failure

The first native hardware campaign stopped at IFM Ouro, B16, prefix 2,048.
Its earlier fourteen cells completed clean rollouts and functional observations.
Their original evidence remains under `raw/ifm_ouro_native_full_profile_v1/`.
The failed cell must not receive a zero latency or a success status.

The failure occurred in the official LM-head output conversion to FP32.
That native conversion remains unchanged.
It is not a change to BF16 model weights, attention, or expert computation.

## Actual checkpoint regression

| Attempt | Allocator | Input | Result |
|---|---|---|---|
| `ifm_ouro_native_oom_repro_default_v1` | Default | B16, prefix 2,048, cap 32 | OOM at prefix 2,053 |
| `ifm_ouro_native_oom_expandable_v1` | `expandable_segments:True` | Identical model, GPU, input, and cap | All 32 outputs completed |

Both attempts use the real checkpoint on physical GPU0.
The diagnostic calls the same native generation function as the failed campaign.
It omits the fourteen unrelated preceding cells.
The only experimental setting change is the allocator flag.
The diagnostic identifier and output directory also differ to preserve both attempts.

At the failing allocation, PyTorch reported 32.11 GiB of live allocation.
It also reported 15.75 GiB of reserved but unused memory.
The requested allocation was 30.67 GiB, against 30.66 GiB of immediately free device memory.
The live allocation at each forward entry stayed near 17.7 billion bytes.
It did not grow by one retained vocabulary-sized tensor per output step.

The changed allocator completed the same workload without reducing batch, context, depth, or precision.
These observations identify allocator fragmentation as the cause of this reproduction.
They do not prove that every possible OOM has the same cause.

PyTorch documents expandable segments for changing allocation sizes.
It marks this setting experimental.
See the [PyTorch 2.11 memory documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).

The successful diagnostic records 25,433.480 ms of rollout wall time.
That run contains diagnostic memory logging.
It is not a clean native benchmark repetition.
The resumed benchmark must obtain its own clean measurements.

## Resume and fairness

The fourteen valid Ouro cells contain 551,936 complete router records.
The resume code checks their result structure before copying them into a new composite attempt.
It records original paths and hashes for every inherited file.
It does not alter the failed attempt or pretend to measure those cells again.

Only the missing IFM B16/prefix-2,048 native cells use the recovery allocator.
Smaller native cells and all original EP timing cells retain the default allocator.
The native-versus-EP comparison records this difference for the affected cells.
It must not label those rows allocator-matched.
The generic EP policy comparison keeps its allocator unchanged across policies.
Whole-E2E native attribution separately identifies its allocator and its own clean denominator.

This correction is runtime recovery, not a new loop-aware optimization.
All prior failed logs remain available.
