# DES-Vote: reproduction and EP mapping

## Reproduction status

`PORT/EXECUTION-CONTRACT_FAILURE`, not `METHOD_FAILURE`.

No official DES code was found.  The minimal port follows DES-Vote's sequence
vote, constructs a beta=0.6 coreset of 38/64 experts, constrains only the active
parallel decoding block, retains top-8 per token, and renormalizes that block.
Constraining the whole no-cache sequence was rejected because future masks
dominated the vote; this correction is recorded in the experiment history.

The corrected single-GPU port scored 1/4 on GSM8K-4 versus vanilla 3/4.  Its
apparent 2.05× speedup is quality-invalid.  The optimized DES fused selection
kernel and Fast-dLLM execution contract are absent, so no DES latency
reproduction is claimed.

## Structural EP result

The active block's expert support is exactly 38 instead of up to 64, a 40.6%
algorithmic proxy reduction.  But each token still has eight branches and 38
experts necessarily span at least three EP4 ranks under contiguous ownership.
In the captured routes they span all four:

| Setting | Active-block coreset | AvgK | Mean destination fanout |
|---|---:|---:|---:|
| EP2 | 38 | 8.0 | 2.0 |
| EP4 | 38 | 8.0 | 4.0 |

Whole-forward unique-expert counts are near 64 because native context rows are
intentionally not constrained in the no-cache diagnostic.  They must not be
mistaken for the active-block coreset size.

The EP4 DES trajectory was numerically unstable relative to EP2 (214 versus 38
NFEs on the one-request trace).  That is another reason not to use its raw
latency as scaling evidence: BF16 distributed accumulation order plus a
quality-invalid approximation changed the decoder trajectory.

## Economic bound

An extremely generous topology oracle that removed one of four rank contacts
would save at most one quarter of the measured dispatch+combine share, or
3.90% request E2E.  It assumes an equally useful lower-fanout coreset and zero
selection cost.  It is below the 5% gate, so no topology-aware DES prototype was
built.

## Conclusion

Unique expert reduction does not imply rank-fanout reduction at top-8/EP4.
This is useful characterization, but the current port cannot establish a DES
method result and the most favorable physical-cost oracle is too small.
