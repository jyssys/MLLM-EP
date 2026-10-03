# dLLM block-local exact expert replicas: bounded Stage-A PoC

Authoritative spec: `references/authoritative_spec.md`. Parent evidence commit:
`dc539d3222193ab0902e0a342679edc3610f54bb` (`poc_dllm_ep_comm`).

Read [report.md](report.md). Verdict **NO-GO** for cold block-local B1 replicas:
large oracle destination-closure headroom but weak early-iteration recovery,
poor expert-copy byte ROI, and instantaneous compute hotspots. Stage B was
intentionally not implemented. No accuracy or latency-speedup claim.

## Reproduce

```bash
bash poc_dllm_block_local_replica/scripts/reproduce.sh
/home/esjung/anaconda3/bin/python -m pytest -q poc_dllm_block_local_replica/tests
```

Run from this branch checkout with its inherited raw evidence. The script
creates a fresh `mktemp` directory and never replaces published results.
CUDA mask is fixed to **2,3,6,7**. A short TRUE EP4 sanity/P2P session stops only
the authorized burner and restores it with a watchdog/finally transaction.
Do not rerun GPU work if all GPU resources have not been granted to you.

Pure offline repeat: set `DLLM_REPLICA_ARTIFACT_ROOT` to a new directory, run
prepare/extract/copy_upper_bound/simulate (physical P2P and baseline validation
can be re-used by explicit copy of their published artifacts if desired).

Policies share one per-source budget across all19 sparse layers. First1/2
iteration profiling bytes are retained. Destination-aware greedy closes
complete selected-expert destination bundles. Oracle is future-truth greedy
with LP relaxation bounds, not an exact optimum. Everything is measured above
the previous **structural live-only** reference, never a measured Epoch row.

Physical measurements: loaded tensor sizes/ownership, current EP4 route/buffer
parity, directed P2P copy times. Offline accounting: post-replica residual,
copy-volume ROI, rank assignments and parallel/bulk copy estimates. Actual
replica D+C/E2E reduction, time ROI and benchmark accuracy are **NOT RUN**.
