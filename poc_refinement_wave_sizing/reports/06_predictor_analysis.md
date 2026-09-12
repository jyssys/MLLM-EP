# Winner Predictor Analysis

No predictor is warranted. Under the strict ready-set-feasible oracle, mini32 wins
or ties on 64/65 waves. The lone exception is one early M1024 wave where mini16 is
estimated to save 40.781 ms.

Consequently:

- phase-only prediction cannot recover a systematic crossover;
- logical liveness is strongly correlated with ready-pool contraction but does not
  identify a distinct optimum;
- physical M, tiny-expert fraction, fanout, CV, or remote bytes cannot create more
  than the 0.650% perfect-oracle headroom;
- a learned controller would only add measurement and switching overhead.

The correct interpretable policy for this substrate is therefore the static rule:
aggregate all currently ready requests, up to mini32 and the HBM limit.
