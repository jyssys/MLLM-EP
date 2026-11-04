# Token suffix scheduler failure

## Verdict

Both new Token Split GPU tests failed during warm-up.
No measured serving request completed in either test.
The failure is a scheduler progress error, not a measured output mismatch or a performance result.
The four-way serving comparison remains `BLOCKED_CORRECTNESS`.

| Model | Failed run | Return code | Process wall time | Error |
|---|---|---:|---:|---|
| Huginn R32 | `static_gate_v4_huginn_token_partial8` | 1 | 77.202428 s | `wavefront completed prompt chunks out of order` |
| Ouro R4 | `static_gate_v4_ouro_token_partial8` | 1 | 34.164767 s | `wavefront completed prompt chunks out of order` |

The tests used prefixes 1,984/2,048 and 3,072/4,096 respectively.
Both allowed full-prompt execution budgets. Neither restricted Token Split to C256.

## Ranked hypotheses

1. The imported suffix origin differs from the staged scheduler's chunk origin.
2. Batch completion order advances the same frontier twice.
3. Imported KV validity or state corruption causes an invalid transition.

The CPU replay supports hypothesis 1 as a sufficient cause for the observed exception.
It does not exclude additional GPU or lifecycle defects.

## Source evidence

`adaptive_loop_token_pd_2gpu/connector.py` sets both imported frontiers to the prefix length:

```python
request.prefill_window_origin = request.prefill_reclaimable_tokens = k
```

The native RLT scheduler includes this origin in its chunk calculation.
However, the inherited `adaptive_recurrence_cut/staged.py` adapter replaces the native task builder.
That adapter uses an origin of zero:

```python
chunk_start = start // chunk * chunk
```

For the Huginn suffix, `start=1984` and `chunk=2048` produce `chunk_start=0`.
The imported reclaimable frontier is 1,984.
For the Ouro suffix, `start=3072` and `chunk=4096` also produce `chunk_start=0`.
The imported reclaimable frontier is 3,072.

When the final recurrent depth completes the suffix, native progress checks these unequal values.
It raises the recorded exception before it can retire the completed chunk.
Origin-zero Loop continuation does not trigger this particular conflict.

## CPU reproduction and candidate

The new CPU tests extract the real installed task builder and native progress method with Python AST.
They execute both methods with small host objects. They do not construct a GPU model.
Both failed configurations reproduce the exact exception.

The opt-in candidate in `token_suffix_origin.py` uses:

```python
chunk_start = origin + (start - origin) // chunk * chunk
```

It retains absolute hidden-state row offsets and the recurrent cut barrier.
It leaves origin-zero requests on the existing builder.
CPU checks pass for the two failed configurations and origin-zero controls.

The candidate defaults to disabled.
No configuration from this campaign enables `token_suffix_origin_candidate`.
It is **CPU_TESTED_GPU_UNQUALIFIED**, not a completed correctness fix.
The external runtime and earlier experiment files remain unchanged.

## Required next validation

1. Start a new run namespace with the candidate explicitly enabled.
2. Repeat both failing warm-ups and the eight-request state checks.
3. Compare exact outputs, transferred bytes, hidden state, and valid KV with native execution.
4. Check aligned and unaligned prefix boundaries with multiple execution budgets.
5. Complete independent N32 or larger state qualification before clean serving.
6. Test dynamic rejection and Full-PD fallback separately.

The campaign's new-job cutoff already passed when the agent resumed.
The existing 11-hour total budget was not reset.
No late GPU test bypassed that cutoff.

## Reproduction

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest \
  poc_flashvep.fluidtoken_loop_split.test_token_suffix_origin -v
```

Result: three CPU tests passed in 0.035 seconds on the first recorded invocation.
The final explicit suite passed 38 tests.
These results do not establish GPU correctness, HF fidelity, or Token/Loop performance.
