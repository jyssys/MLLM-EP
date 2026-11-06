# D-local prefill and depth-mixed batching PoC

The campaign uses the qualified private RLT runtime without editing it.
The source and evidence namespace are separate from prior experiments.
The real Git index remains unchanged. Publication uses an alternate index.

## CPU checks

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m pytest -q poc_flashvep/d_local_cross_depth_batching/test_contract.py
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.analyze
```

## GPU execution

Create an immutable configuration with `prepare.configuration` and `common.save`.
Then use the guarded dispatcher:

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.run RUN_NAME
```

The dispatcher leases the exclusive all-GPU quiet lane. A serving job still uses
only its declared two-GPU pair. It stops only verified approved burner sessions.
It preserves failed logs and bounds worker cleanup. Do not launch workers directly.

`campaign.py` expands an exact N8 path to N32. It does not expand a failed path.
N32 submits 32 requests together. It does not imply 32 simultaneous GPU residents.
Heavy state hashing is not clean latency evidence.

`micro.py` compares ready-state full-core operations and matched dense work.
It retains the original fixed-row projection and normalization fixes.
The same/cross dense-label control uses one identical graph.
`opportunity_micro.py` changes recurrent compatibility only. It preserves the
same native Prelude/Coda priorities and does not report PD online capacity.
The additional `same_align` control permits one legal prefill step before a
same-depth join. Its cost enters the timer. It is not strict decode priority.

## Evidence boundaries

Generated outputs use fixed caps and ignore EOS. Prefix caching is disabled.
CUDA events enclose device work and gaps. They are not exclusive kernel time.
The activity-only collector supplies kernel timestamps, not SM cycles or bandwidth.
Native PD transfer/ACK events remain separate from local ownership events.
No independent local micro request is counted as migrated serving computation.
Original HF equivalence remains a separate, unmeasured contract.

Do not promote a profiler trace unless matched local-tail interference is within
the frozen 5% allowance. Do not run online cross-depth integration unless the
incremental microbenchmark gate passes. Do not interpret generic/CDB reuse as a
new loop-specific operation.
