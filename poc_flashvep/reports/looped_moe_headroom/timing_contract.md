# Timing contract

## Clean measurements

Primary forward timing uses B1 prefixes of 128, 512, and 2,048 tokens.
The hardware passage repeats a frozen language passage to obtain exact lengths.
It is not a downstream accuracy sample.
Each cell has 5 warmups and 20 measured repetitions.
Three independent process starts provide the interval denominator.

The clean wall timer encloses the complete native forward and CUDA synchronization.
It excludes model loading, artifact hashing, downloads, and one-time environment installation.
The clean timing path has no routing observer or expert intervention.

Generation measurements use prefixes of 128 or 512 tokens.
They generate 16 or 64 tokens for fixed hardware shapes.
EOS is disabled only for those measurements.
E2E ends when the last output token's GPU work completes.
TPOT excludes the first token and divides by output length minus one.
Tokenization, network delivery, queueing, and final text decoding are outside this model-only timer.
This is not a production serving benchmark.

Both runtime families recompute the full prefix at every generated token.
The term cached decode is therefore not applicable.

## Observer measurements

CUDA-event spans use separate forwards.
Expert spans contain the three native grouped GEMMs and their activation path.
Their internal CPU launch gaps can also appear between the recorded GPU events.
The report does not describe these spans as kernel-only compute.

The observer-to-clean wall ratio measures instrumentation overhead.
A ratio outside 0.90–1.10 produces a PARTIAL fraction.
The optimistic model still states its clean-denominator assumption.
It does not turn a PARTIAL trace into an exact E2E partition.

The generation observer records each actual full-prefix step separately.
It does not multiply the final step by the number of tokens.
The supplementary observer run also records full-generation wall time and token parity.

## Heavy profiling

Heavy profiling records CPU operations, CUDA kernels, and a common timeline.
Its accumulated service times remain distinct from clean wall time.
Kernel interval unions remove overlapping kernel intervals inside that trace.
The remaining gaps can include host work, copies, and synchronization.
They are not automatically GPU idle or removable critical-path time.

Attention, projection, norm, router, and output event spans are separate evidence.
Unresolved dispatch and combine costs remain unresolved.
No nested span sum becomes an invented 100% E2E breakdown.

## Component microbenchmark

The microbenchmark uses real checkpoint weights and observed expert routing geometry.
It compacts expert inputs before the native GEMMs.
It measures quotas from 0% through 50% row removal.
The v2 sequence crosses quota order between independent process starts.
Foil GPU offsets are prepared before the timer.
IFM retains its official native count handling.

Selector, compaction, routing, cache, and combine costs are excluded from this operator-only measurement.
Any component gain is therefore not an achieved E2E gain.
The q=0 operator output must match the captured native output exactly.

## Amdahl interpretation

The reported model is `S = 1 / (1 - f_expert × q)`.
The numerator remains one complete baseline forward.
The q value uses realized omitted edges, not the requested quota.
The model assumes proportional time removal and zero added overhead.
It is an optimistic conditional estimate, not a measured speedup.
It does not bound every possible future engine or nonuniform pruning algorithm.
