# Supplemental DAG identity checks

## Reason and status

The [independent audit](contract_completeness_audit_v1.md) found three coverage gaps in the original structural checker.
An unregistered external history, an unavailable future token, and another request's gate all passed that checker.
The original eight tests remain valid within their narrower scope.
Those gaps do not establish that a previously generated graph was invalid.

The original checker and its frozen outputs remain unchanged.
The [new supplement](../dag_identity_checks.py) adds explicit identity witnesses.
It does not implement a scheduler or certify a complete Ouro execution.

## Checked obligations

The supplement calls the original structural checks first.
It then requires these declared properties:

- External history has a registered request, generation, depth, layer, position range, and committed version.
- A current-KV producer belongs to the matching request, generation, depth, and layer.
- A current-KV producer has an explicit dependency path to its consumer.
- A committed consumer cannot read provisional current KV.
- Private provisional reads retain the same branch version.
- Registered external history cannot also have an in-graph writer under that same version key.
- A required gate belongs to the same request, generation, token, and preceding depth.
- An input token is a declared prompt/prior committed sample or follows its matching sample predecessor.

The caller declares prefill or decode explicitly.
Known prompt chunks do not require sampled-token predecessors.
A guessed future token cannot enter through the known-token registry as `PREDICTED`.
Opaque history names do not replace the typed state manifest.

## Tests and remaining limits

The new suite contains seventeen CPU tests.
It includes matching-history, sampled-token, private-branch, and small causal-prefill positive controls.
It includes wrong-request, wrong-generation, wrong-depth, future-position, unavailable-token, and provisional-read negative controls.
The exact source graph, node values, and runtime publication do not execute.
The [retained red record](../artifacts/dag_identity_checks_red_v1.json) precedes the supplement implementation.
An independent review then found a declaration collision and unchecked index types.
The [second red record](../artifacts/dag_identity_review_red_v1.json) retains those failures before correction.
The supplement now rejects shared external/write keys and non-integer identity indices.

This supplement verifies declared witnesses, not their correspondence to GPU values.
It does not prove that every required hidden-state edge exists.
It does not recompute the original gate's hazards or cumulative policy.
It does not prove KV allocator lifetime, eviction, exclusive memory leases, or device synchronization.
It does not calibrate compute, copy, collective, or link contention.
Cross-depth aliases and speculative future tokens need a separate explicit contract.

The caller can still provide a false manifest or omit an operation.
A future candidate therefore needs both a complete model-specific DAG and runtime correctness checks.
A synthetic `PASS` remains neither numerical parity nor physical headroom.
No candidate status or timing claim changes from these CPU checks.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_ouro_parallelism_deep_discovery.tests.test_dag_identity_checks_cpu poc_ouro_parallelism_deep_discovery.tests.test_contract_validator_scope_v1 poc_ouro_parallelism_deep_discovery.scripts.verify_dag
```

The combined command covers seventeen supplemental, five characterization, and eight original tests.
