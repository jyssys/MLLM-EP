# Structural weakness oracles

The structural oracles are deliberately optimistic but preserve payload and
semantics.

- O1 variable-capacity LL grants a 1% service-time ceiling despite the absence
  of a monotonic measured latency effect.
- O2 unlimited-inflight LL grants the residual Q=8→16 throughput slope, 1.17%,
  as if it were fully removable.
- O3/O4 combine both ceilings; no additional unmeasured transfer saving is
  invented.

At Poisson 82.5% load, O4 changes throughput by +0.16% and p99 by -4.53%. At
95% it changes throughput by +0.78% and p99 by -6.09%. At closed-loop Q=32 it
changes throughput by +2.19% and p99 by -2.14%. Even optimistic route-replay
tail gain stays below 8%, while throughput is far below the 10--15% kernel
gate.

The larger tail percentage near saturation is queueing amplification of a
~2.2% service ceiling, not evidence for a large EP kernel defect. Figures
15--17 and `SERVING_ORACLES.csv` contain the full results.
