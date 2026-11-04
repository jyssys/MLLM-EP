# Controller namespace race

## Observed failure

The static controller exited during a shared-cell request at 19:44 KST.
The original primary controller still ran that cell under the exclusive quiet lane.
Its worker continued normally.
The static controller did not overwrite that worker's data or terminate it.

The affected cell was `f_huginn_low_token1984_start1_128`.
The exception was `Unfinished namespace requires explicit evidence review`.
This was a controller coordination failure, not an output or KV mismatch.
The failed controller log remains `logs/static_controller_reloaded_v3.log`.

## Cause and scoped change

The older primary controller did not take the newer per-namespace lock.
It could create a namespace after the static controller checked for that namespace.
The inherited dispatcher then rejected the newly created directory.

The new wrapper recognizes only the two exact inherited namespace-conflict messages.
It waits for the existing quiet lane.
It then reads the finalized dispatcher receipt.
A failed receipt remains a failure.
A missing receipt still requires explicit review.
Unrelated runtime errors remain errors.
The change does not alter scheduling, model execution, or acceptance thresholds.

## Regression check

The new test exercises the check-to-dispatch race at the actual wrapper call site.
Before the change, the successful-receipt and failed-receipt cases raised unexpected exceptions.
After the change, all four race tests passed.
The two existing budget tests also passed.

Command:

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest \
  poc_flashvep.dynamic_loop_split_serving.test_namespace_race \
  poc_flashvep.dynamic_loop_split_serving.test_run_budget
```

This CPU test does not certify model correctness.
The original GPU cell retains its separate output, work, and state gates.
Controller reload metadata remains in `state/controller_reloads.jsonl`.
