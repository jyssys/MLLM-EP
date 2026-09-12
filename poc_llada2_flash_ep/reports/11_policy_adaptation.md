# Policy adaptation and TEAM transfer

## Static policy result

The only decisive policy is topology-specific static forward granularity:

- routed EP4: `mini_batch_size=8` among 1/2/4/8/16;
- TP4: `mini_batch_size=16` among the same choices.

This yields large deployment gains over the default but requires neither a new
runtime method nor denoising-phase adaptation.

## Denoising-adaptive policy

No phase-adaptive policy is justified. Phase changes logical liveness and
expert support, but the current runtime executes fixed physical rows, and no
clean paired per-phase topology switch was possible. TP4 and EP4 keep
incompatible expert layouts; each already consumes about 57 GiB/rank, so a
dual resident layout cannot fit in 80 GiB. Reloading tens of gigabytes between
phases overwhelms the bounded generation window.

## TEAM transfer

`TEAM-PORT-NOT-ESTABLISHED`.

TEAM's official implementation is SDAR-specific. The specification makes TEAM
secondary and permits transfer only after a strong vanilla candidate exists.
No novelty-eligible candidate reached the 8% HOLD gate, so a semantic TEAM
port would not answer the primary question and was not attempted. This is not
a TEAM method-failure result.

## Epoch complementarity

The measured physical/live gap strongly supports Epoch's target regime. The
remaining exact opportunities—packing, load balance, route metadata reuse,
and payload-only replication—are 1.80%, 5.34%, 1.57%, and at most 2.99% E2E,
respectively. None supplies a large complementary successor.
