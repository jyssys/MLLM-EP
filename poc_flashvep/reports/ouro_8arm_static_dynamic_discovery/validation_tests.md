# Validation checkpoint

The CPU contract suite contains eighteen tests. The latest completed invocation passed all eighteen tests.
The tests cover source syntax, frozen question identity, the 400-question target, and all 96 KV ownership slots.
They compare the gate against the pinned official CDB policy in FP32 and BF16.
They check minimum-depth hazard accumulation, current-row last_exited copying, and unchanged historical prefix rows.
They also check no-refill admission order and speedup/interval formulas.
They check generated-token difference indexing and distinguish native custom reduction kernels from NCCL kernels.
They also check rank-load CV and max/mean formulas, including an empty-work case.
The bounded serving-layout flag has CPU validation. Its command plan uses mocks and does not execute CUDA or Nsight.
The launch regression checks the actual native environment's `ninja` executable through a profiler-wrapped command.
It failed before the PATH repair and passed after that repair.
The profile-resume test checks that successful captures are skipped and failed attempts remain unchanged.
The task-cohort ratio test checks the stratified paired-bootstrap formula on exact reference ratios.

```bash
cd /home/esjung/MLLM-EP-ouro-request-pp
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_flashvep.ouro_8arm.test_contract -v
```

Four actual-GPU Static probes compare Single with Layer PP and Loop PP.
Both candidates match output IDs, depth traces, and all 96 logical KV slots bitwise on these probes.
These probes use four requests. They do not establish full benchmark accuracy.
The final 400-question Static Single, Layer PP, and Loop PP outputs match token-exactly.
The paired CSV records 300/300 GSM8K and 100/100 MATH request matches for each PP candidate.
Dynamic SD, LD, and OD also match all 400 output and full exit-depth sequences.
Those observations do not prove parity for untested inputs or a cross-backend CDB match.

The historical strict TP P512 gate remains FAIL.
New task results use a visible exploratory numerical status.
Dynamic physical forced-exit and live-gate probes also completed.
LD and OD match SD output IDs, exit depths, and all 96 logical KV slots bitwise on four requests each.
TD matches the bounded output and exit traces, but its assembled KV comparison remains numerically unresolved.
The official CDB comparison matches four short synthetic output and exit traces.
It does not establish cross-backend full-KV or long-generation parity.
See `dynamic_reference_validation.md` for the measured scope and limitations.
All 72 planned clean replay comparisons completed with five measured repeats each.
All eighteen planned heavy captures completed. The final observer table separates semantic parity from heavy timing overhead.

The artifact validator checks 122 protected earlier files, the checkpoint hash, and canonical prompt IDs.
It preserves the known older mismatched-input record. It excludes that record from canonical paired comparisons.
The executed-work audit compares stage-body counters with completed token-depth traces.
Those checks do not convert kernel service sums into E2E component shares.
