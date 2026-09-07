# Rethink after 30 assumptions

**Expected:** workload and communication geometry would expose a hidden EP
axis beyond histograms.
**Observed:** fanout/traffic adds +0.001% held-out error and SMS envelope is
0.11% median/2.10% max.
**Failed assumption:** another routing scalar necessarily explains residual
latency.
**New system fact:** state/ownership, not static routing geometry, is the
remaining uncertainty.

New children: (1) communication-credit contract, (2) state-aware stream
ownership, (3) per-layer progress deadline.  All have either the 1.09% tail cap
or generic overlap prior-art collision and are not GPU-promoted.
