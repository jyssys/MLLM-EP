# Online serving results

LL-only and the perfect existing-backend oracle are identical on every real
shape because real M never reaches the 4096--8192 crossover interval.

At Poisson 82.5% offered load, LL sustains 115.71 route-replay requests/s with
p95 83.92 ms and p99 92.10 ms. Normal sustains only 47.48 requests/s and enters
a long queue. At 95% load LL reaches 129.29 requests/s and p99 126.91 ms.

Bursty load preserves the same backend ordering. Closed-loop Q=32 reaches
138.10 requests/s under LL versus 47.90 under Normal. No concurrent-wave or
mixed-M point causes Normal to overtake LL.

This is an EP-backend result, not a claim about full LLaDA2 online serving.
Queueing magnifies small service-time changes near saturation, so tail-only
changes are interpreted together with throughput and the measured kernel cause.
See `SERVING_RESULTS.csv`, `SERVING_ORACLES.csv`, and Figures 10--13.
