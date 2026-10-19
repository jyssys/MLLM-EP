# Independent supplemental-checker review v1

## Scope and result

The supplement improves declared identity checks. Its stated exclusions are appropriate. However, two malformed declarations still pass at the inspected source hash. Neither finding establishes that an existing generated graph is invalid.

The inspected [checker](../dag_identity_checks.py) has SHA256 `5d12e6a64addbd227cc3e7bfc8665c302a549dbfbf9716b5dd53f34af57f8136`. The [new suite](../tests/test_dag_identity_checks_cpu.py) contains fifteen tests at this snapshot. All 28 combined tests pass: fifteen supplemental, five retained characterization, and eight original tests. They do not cover the two cases below.

## 1. External history and current writes can name the same key

The producer below writes a provisional version. The consumer has no dependency on that producer. The registry simultaneously calls the same key committed history.

```python
p = Node('p', 'q0', 0, 0, 0, 'norm_qkv', [], None,
         [], ['history:shared'], 0,
         committed=False, scratch_version='branch1')
c = Node('c', 'q0', 1, 0, 0, 'attention', [], None,
         ['history:shared'], [], 0)
validate_identities(
    [p, c],
    external_kv={'history:shared': KVState('q0', 0, 0, 0, 0, 1)},
    known_tokens={('q0', 0, 0): 'PROMPT',
                  ('q0', 0, 1): 'PRIOR_COMMITTED_SAMPLE'})
# Observed: True
```

Lines 64–68 choose the external-history path from the name prefix. That path skips the in-graph producer checks at lines 70–77. The combined declaration therefore bypasses the producer-path and provisional-read restrictions.

This is a static namespace collision. Detecting it needs no runtime values, eviction model, or complete model graph. Under this limited contract, external state keys should be disjoint from in-graph write keys. A mutable-history protocol would need a separate explicit version/publication contract.

## 2. Dataclass annotations do not validate integer ranges

This registered state also passes:

```python
c = Node('c', 'q0', 1, 0, 0, 'attention', [], None,
         ['history:fractional'], [], 0)
validate_identities(
    [c],
    external_kv={'history:fractional':
                 KVState('q0', 0, 0, 0, 0.5, 1.0)},
    known_tokens={('q0', 0, 1): 'PRIOR_COMMITTED_SAMPLE'})
# Observed: True
```

Lines 47–49 check the dataclass and numeric inequalities. They do not enforce integer position fields. Fractional token boundaries do not satisfy the declared token-index schema. Float-valued generation/depth/layer fields also compare equal to integer identities in the separate inspected case.

Check the relevant declared index types explicitly, or document a mandatory schema-validation precondition. This issue concerns malformed inputs. It is not evidence of a wrong model state or measured runtime failure.

## Preserved limits

A matching gate identity does not prove original-gate CDF arithmetic. A sampled-token witness does not prove the sampled value. Registered history does not prove its GPU contents or lifetime. No accepted graph establishes resource feasibility, legal aliasing, numerical parity, or quality.

The [scope report](dag_identity_supplement_v1.md) retains those distinctions. No larger validator, checker edit, earlier-report edit, or GPU execution occurs in this review. The [observation record](../artifacts/dag_identity_supplement_independent_review_v1.json) pins both accepted cases and the actual worker interval.
