# REFLEX: reproduction and EP mapping

## Reproduction status

`PORT/EXECUTION-CONTRACT_FAILURE`, not `METHOD_FAILURE`.

No official REFLEX repository was discoverable in the source audit.  The port
implements the paper's RRB/FGER expert-count rules, preserves router ranking,
uses the specified refinement-state budgets, and renormalizes selected weights.
It was run inside dInfer's available no-cache blockwise execution, not the
paper's Fast-dLLM substrate.

On the bounded GSM8K-4 single-GPU gate, vanilla scored 3/4 while the REFLEX port
scored 0/4 and produced malformed text.  Its apparent 4.36× latency reduction
is therefore invalid and is excluded from performance claims.  EP2/EP4 were
run only to map the same action policy to physical assignment-level cost.

## Structural mapping

| Metric per physical row | EP2 vanilla | EP2 REFLEX | EP4 vanilla | EP4 REFLEX |
|---|---:|---:|---:|---:|
| AvgK | 8.000 | 7.516 | 8.000 | 7.516 |
| Pair reduction | – | 6.05% | – | 6.05% |
| Remote assignment reduction | – | 6.29% | – | 7.07% |
| Mean destination fanout | 2 | 2 | 4 | 4 |

Thus expert-pair reduction can reduce assignment-aware payload slightly more
than proportionally when marginal experts are remote, but it does not reduce
fanout in this regime.  More importantly, the stock dInfer/vLLM naive manager
multicasts hidden states and router logits independently of selected k.  In the
unmodified stock backend the same 6.05% AvgK reduction yields **zero dispatch
or combine byte reduction**.

## Economic bound

Even granting proportional reduction of the entire measured LLaDA MoE span,
6.05% fewer pairs maps to a 3.83% request-level upper bound on this anchor.  It
is below the 5% candidate gate before selector overhead or quality loss.  A
cost-aware marginal-expert policy was therefore not implemented.

## Conclusion

The structural answer to “do fewer pairs imply proportionally lower EP cost?”
is **no**: the relation depends on whether the backend transports assignments
or full token tensors, and fanout remains saturated.  But because the faithful
quality contract was not reproduced, this is a runtime/substrate finding, not
a negative result about REFLEX itself.
