# Dynamic N128 startup signature recovery

The N32 branch tests use the ordinary Full-PD startup path.
The N128 configuration builder still enabled short-prompt geometry cycling.
This option differed from the frozen cost-table signature.
The cost-table guard would reject that configuration before online evaluation.

No N128 Dynamic GPU execution used the inconsistent configuration.
The active Huginn N32 GPU worker remains unchanged.
The controller reload waits for that worker to finish.
The replacement controller reuses completed immutable run receipts.
The original campaign deadline remains unchanged.

The regression test calls the actual N128 configuration builder.
It compares every execution-signature field with the N32 qualification configuration.
The test failed for both models and both split axes before the correction.
All four cases pass after the correction.
The correction removes one N128-only startup option.
It does not change model work, online decisions, or frozen SLO thresholds.

Reproduce the CPU test:

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest poc_flashvep.dynamic_loop_split_serving.test_dynamic_serving_startup
```
