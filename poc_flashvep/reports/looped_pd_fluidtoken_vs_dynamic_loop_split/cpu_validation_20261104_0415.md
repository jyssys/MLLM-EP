# CPU validation at 04:15 KST

Evidence class: executed CPU unit tests. These tests do not establish GPU correctness or performance.

The explicit test suite passed 30 tests in 0.070 seconds.
The interpreter was `/home/esjung/.venvs/looped-pd-abc/bin/python`.
The working directory was `/home/esjung/MLLM-EP-github`.

The suite covers collector attributes, decision guards, calibration contracts, evidence routing, host event partitions, probe admission, probe pacing, and split accounting.
The queue-relief tests use synthetic costs. They are not measured online scheduling results.
The pacing tests validate the credit counter. They do not validate staged Huginn execution.

An earlier `unittest discover` command failed because the namespace directory was not importable through that discovery entry point.
The explicit module invocation succeeded. No source package layout was changed to hide the discovery failure.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest \
  poc_flashvep.fluidtoken_loop_split.test_collector_buffers \
  poc_flashvep.fluidtoken_loop_split.test_decision_model \
  poc_flashvep.fluidtoken_loop_split.test_dynamic_calibration \
  poc_flashvep.fluidtoken_loop_split.test_evidence_routing \
  poc_flashvep.fluidtoken_loop_split.test_host_breakdown \
  poc_flashvep.fluidtoken_loop_split.test_probe_admission \
  poc_flashvep.fluidtoken_loop_split.test_probe_quantum \
  poc_flashvep.fluidtoken_loop_split.test_queue_relief \
  poc_flashvep.fluidtoken_loop_split.test_recovery_gate \
  poc_flashvep.fluidtoken_loop_split.test_split_accounting \
  poc_flashvep.fluidtoken_loop_split.test_split_adapter
```

New GPU variants remain behind exact state, output, and full-work gates.
The controller cannot authorize method expansion from these CPU tests alone.
