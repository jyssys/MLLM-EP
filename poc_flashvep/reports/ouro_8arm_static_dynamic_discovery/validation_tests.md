# Validation checkpoint

The CPU contract suite contains ten tests. The latest completed invocation passed all ten tests.
The tests cover source syntax, frozen question identity, the 400-question target, and all 96 KV ownership slots.
They compare the gate against the pinned official CDB policy in FP32 and BF16.
They check minimum-depth hazard accumulation, current-row last_exited copying, and unchanged historical prefix rows.
They also check no-refill admission order and speedup/interval formulas.

```bash
cd /home/esjung/MLLM-EP-ouro-request-pp
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_flashvep.ouro_8arm.test_contract -v
```

Four actual-GPU Static probes compare Single with Layer PP and Loop PP.
Both candidates match output IDs, depth traces, and all 96 logical KV slots bitwise on these probes.
These probes use four requests. They do not establish full benchmark accuracy.
The 114-question Static Single, Layer PP, and Loop PP pilot outputs also match token-exactly.
The paired CSV records 64/64 GSM8K and 50/50 MATH request matches for each PP candidate.

The historical strict TP P512 gate remains FAIL.
New task results use a visible exploratory numerical status.
Dynamic physical probes and timing tests remain pending at this checkpoint.

The artifact validator checks 122 protected earlier files, the checkpoint hash, and canonical prompt IDs.
It preserves the known older mismatched-input record. It excludes that record from canonical paired comparisons.
The executed-work audit compares stage-body counters with completed token-depth traces.
Those checks do not convert kernel service sums into E2E component shares.
