
# Q2 VDS versus skip damage

The previously frozen primary functional damage (`next-token JSD + nonnegative target-logprob drop + top1 penalty`) was joined to newly measured VDS at exact request/layer/step.

| Frozen VDS tertile | Held-out mean functional damage |
|---|---:|
| low | 0.0572256 |
| mid | 0.0250392 |
| high | 0.0017475 |

- High−low: -0.0554781
- Request-bootstrap 95% CI: `[-0.17318413139824318, 0.0028926955343391864]`
- Monotonic: `False`
- CI strictly positive: `False`
- Direct causal alignment: `False`

Verdict: **VISUAL_DEPENDENCY_NOT_LINKED_TO_SKIP_DAMAGE**. K8→K6/K4 propagation was stopped after the drop-one gate; it is not silently treated as zero damage.
