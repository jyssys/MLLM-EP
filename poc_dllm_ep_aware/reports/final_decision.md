# Final decision

## Label

**CHARACTERIZATION-ONLY**

The PoC found a real algorithmic/physical mismatch in TEAM, but no candidate
perfect oracle reaches the 8% promotion gate.  REFLEX and DES do not supply
quality-valid cross-family confirmation in the available execution substrate.

## Final questions

1. **How do TEAM, REFLEX, and DES scale single→EP2→EP4?** TEAM's median
   speedup changes from 1.832× to 1.179× to 0.821×.  REFLEX/DES EP2/EP4
   results are structural-only because their minimal ports failed the bounded
   quality gate; no scaling speedup is claimed for them.
2. **Which proxy predicts EP latency best/worst?** TEAM NFE is the clearest
   failure: fewer calls coexist with more EP work and slower EP4 E2E.  REFLEX
   pair count maps to assignment bytes only on a ragged transport.  DES unique
   expert count is a poor fanout proxy in the captured top-8/EP4 regime.
3. **Does TEAM communication amplification worsen on EP4?** The payload
   increase is +33.6% on EP2 and +32.1% on EP4, so its *percentage* does not
   grow.  Its latency consequence worsens enough to reverse the winner as the
   topology exposes startup/synchronization cost.
4. **Do REFLEX pair reductions reduce remote bytes?** By 7.07% in the
   assignment-A2A diagnostic, but by 0% in stock naive EP because that backend
   transports full token/router tensors independently of selected k.
5. **Does DES coreset reduction reduce rank fanout?** No.  The 38/64 active
   coreset still touches all four EP4 ranks.
6. **Is there a common predictive EP cost model?** Not a decision-grade one.
   The best critical-rank model has 4.44% MAPE but R²=0.058 on held-out route
   shapes; the low error is mostly a flat startup floor.
7. **Can the model improve inference decisions training-free?** No live policy
   was justified: every action-level oracle is below 8%.
8. **Largest realistic opportunity?** Candidate F, a perfect-future joint
   decode–EP/early-commitment oracle, is 6.28% E2E and remains weak.  The most
   implementable TEAM-width bound is 4.01%.
9. **Does it survive faster communication?** No.  TEAM-width headroom falls to
   2.01% at 0.5× communication cost and 1.00% at 0.25×.
10. **Paper-level method and EP8?** No.  The evidence supports a TEAM EP scaling
    warning and backend-design lesson, not a general method.  EP8 is not
    justified without a faithful REFLEX/DES substrate or a new action trace
    whose quality-matched direct oracle exceeds 10%.

## Evidence boundary

### Measured

- true EP4 expert ownership and exact reference A2A transport;
- TEAM clean three-restart latency and structural work counts;
- LLaDA assignment-level REFLEX/DES action traces;
- 84-shape EP4 fused-expert/NCCL calibration surface.

### Invalid or structural-only

- REFLEX: 0/4 versus vanilla 3/4 on bounded GSM8K;
- DES: 1/4 versus vanilla 3/4;
- raw REFLEX/DES latency and cross-EP trajectories;
- observer-heavy stage sums as headline request performance.

### Analytical only

- Candidates A–F;
- communication sensitivity;
- the 15.60% impossible all-communication-removal ceiling.

## Recommendation

Do not build an EP-aware controller, variable-k DeepEP path, or TEAM replica
manager from these results.  Preserve the TEAM EP4 reversal as a reproducible
systems characterization.  Reopen the method question only if an official,
quality-valid REFLEX/DES implementation exposes a quality-matched request-level
oracle of at least 8%, preferably 12%.
