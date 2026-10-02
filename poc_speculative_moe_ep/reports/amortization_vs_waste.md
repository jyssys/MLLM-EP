# Amortization Vs Waste

Measured clean TRUE EP4 on GPUs2,3,6,7, output cap64/EOS, first16 frozen requests (8+8). Route features come from a separate trace pass with identical request/prefix/draft IDs.

Crossover inside tested depths: **False**. Best fixed EP depth: 8. Do not force a U-shaped result or extrapolate gamma16.

- gamma0: target calls/progress=1.0000; EP/verified row=25.9526ms; expert/row=19.5641ms; rejected logical bytes=0.00%; remote useful bytes/progress=1072280.
- gamma2: target calls/progress=0.3741; EP/verified row=8.4052ms; expert/row=6.3968ms; rejected logical bytes=9.41%; remote useful bytes/progress=1204618.
- gamma4: target calls/progress=0.2597; EP/verified row=5.4977ms; expert/row=3.8876ms; rejected logical bytes=20.56%; remote useful bytes/progress=1394112.
- gamma8: target calls/progress=0.1751; EP/verified row=2.9029ms; expert/row=2.0916ms; rejected logical bytes=33.23%; remote useful bytes/progress=1692285.

Rejected bytes are logical attribution to rejected candidate input rows; batched verification does not make all of them independently removable. Output-cap trimming is not relabeled rejection. AR emits the first token in prefill, so post-prefill committed denominators exclude that token while request E2E includes it.
