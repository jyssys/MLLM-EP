# Controlled alternative actions

Status: completed forced-action diagnostic. This is not an adaptive serving result.

Four independent model-process starts compare C512 with C2048 at G1 and G2.
Two starts use approximately1.8K inputs. Two starts use approximately8K inputs.
Each intervention starts after three background requests produce at least16 tokens.
All three background requests remain active at each recorded intervention.
The initial generated IDs, phase, hidden-state hash, and valid depth-indexed KV hashes match in 8/8 pairs.
The target prompt completes Full R4 in every case.

| Input / GPU | G | C512 target TTFT ms | C2048 target TTFT ms | C512 background ITL P99 ms | C2048 background ITL P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long /4 |1 |861.95 |911.48 |105.39 |112.13 |
| Long /4 |2 |1533.72 |175.35 |595.91 |87.24 |
| Long /6 |1 |1308.82 |415.34 |196.35 |57.02 |
| Long /6 |2 |609.75 |177.49 |75.02 |91.02 |
|8K /5 |1 |3234.24 |987.30 |48.84 |65.73 |
|8K /5 |2 |2463.78 |772.41 |76.95 |91.19 |
|8K /7 |1 |3237.13 |1435.51 |52.55 |140.43 |
|8K /7 |2 |2491.14 |794.11 |71.78 |98.52 |

Evidence: LIVE_MEASURED_FORCED_COMPONENT. Values include the event observer and any new graph preparation.
The runs stop after the target's first delivered token. They do not measure full-answer latency or accuracy.
Pre-intervention state hashing pauses the host. Background gaps that cross that pause are excluded.

Smaller C reduces background P99 in 6/8 pairs, but increases target TTFT in 7/8 pairs.
One pair improves both quantities. Two pairs worsen both quantities.
The large Long/G2/GPU4 reversal contradicts a universal smaller-C benefit.
Two starts per input regime cannot establish a population effect or a stable speedup.
Differences in cold action shapes and graph preparation remain possible contributors.

These observations establish that alternative C choices have physical effects from matched initial state.
They do not establish that an online selector can identify the beneficial alternative.
The forced choices do not enter the headline adaptive-policy result.
The C512/C2048 intervention does not reproduce each of the 61 actual online C changes.
The beneficial fraction among those 61 selected changes is NOT_IDENTIFIED by this diagnostic.

Data: `controlled_alternative_pairs.csv` and immutable `raw/dual_p2_ALTERNATIVE_*` records.
