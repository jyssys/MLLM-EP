# Baseline audit

Status: Phase 0 in progress.

The historical commits remain unchanged.
The fixed-G2 source is `c458f6a2a764f667a0da6784bcb2592b36626837`.
The SLO study source is `490c91e29e6b855cd3e911c52f8614c9980b9efb`.
That SLO study used four independent GPU workers.
The new specification calls it an eight-worker study in one heading.
This campaign does not adopt that historical description.

The new runtime checkout uses RLT commit `4a350344027960a48384626aa646f1765fda6d4f`.
It applies the preserved removable G2 patch.
It also permits G3 in the configuration validator.
G3 remains unqualified until the isolated numerical and full-work tests pass.
The original checkout remains unchanged.

The model is Base Ouro-1.4B, not the Thinking checkpoint.
The checkpoint revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
All execution uses BF16 and Static R4.
Last-exited KV storage has 96 independent layer-depth namespaces.
No adaptive model early exit is enabled.

The first frozen-trace check found 86 exact sequences among 96 comparable records.
These records include twelve historical cohorts.
They do not replace the prior report's 72/80 denominator.
The check found eight distinct mismatched request IDs.
Every compared depth trace matched.
`legacy_parity_registration.json` preserves paths, hashes, and first divergent positions.

## Ranked diagnostic hypotheses

1. Batch geometry changes BF16 arithmetic before a near-tied output logit.
2. Graph selection changes arithmetic or buffers under otherwise matched state.
3. KV or hidden-state publication depends on an incorrect stream or buffer lifetime.

The first hypothesis predicts differences under matched-state geometry replay.
The second predicts graph-dependent differences after controlling row membership and input state.
The third predicts a state inconsistency or a strict-synchronization-dependent result.
None is a confirmed cause yet.
Targeted graph, stream, hidden, and KV controls must distinguish them.

## Resource control

GPUs 0–7 are authorized for this campaign.
Workers see one physical GPU through a single-device CUDA mask.
The worker checks the CUDA device UUID against the physical inventory.
Recognized burner sessions have the same UID and an explicit `utilize.py --worker --gpu` command.
The resource helper refuses to stop any other GPU process.

Quiet interference controls reserve idle devices as part of the measurement condition.
Burners must remain stopped during those controls.
Outside reserved controls, empty owned devices receive experiment jobs or burners.

## Repository protection

There are 68 unrelated staged additions.
Their initial binary diff hash is `7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.
Only new campaign paths belong in campaign commits.
No reset, clean, or historical artifact rewrite is permitted.
