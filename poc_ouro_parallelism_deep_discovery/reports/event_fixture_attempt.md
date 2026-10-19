# Internal-event fixture attempt

## Outcome

The first primitive fixture fails before it establishes a valid captured-graph timing result.
The installed compiler fallback raises `RuntimeError: expected event to be a torch.Event object`.
The requested objects are `torch.cuda.Event(enable_timing=True, external=True)`.
The failure occurs inside the traced `streams::record_event` path.
No valid internal interval or observer-overhead result follows from this attempt.

The fixture uses an artificial four-MLP chain with actual pretrained weights and one captured MLP input.
It is not the native Ouro recurrent stack.
It has no attention, depth-indexed KV, collectives, or original gate.
No result from that fixture could establish native pass timing or architecture performance.

The guard restored the authorized owned burner after the child failed.
No installed Torch or vLLM package changed.
The next diagnostic inspects actual native graph topology without inserting internal event markers.

## Evidence

- Script: `scripts/event_nodes_fixture.py`.
- Contract and error: `outputs/event_nodes_fixture_v1/fixture_contract.json` and `status.json`.
- Full failure stack: the same directory's `console.log`.
- Source feasibility limits: [recurrent_pass_event_feasibility.md](recurrent_pass_event_feasibility.md).

This is a measurement-path failure, not a model-quality failure or an architecture verdict.
