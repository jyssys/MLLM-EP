# LLaDA2 temporal communication and approximate PP PoC

This workspace contains two independent studies on the validated LLaDA2.0-
Flash true-EP4 substrate:

- Track A: destination-qualified temporal activation-delta communication.
- Track B: AsyncDiff-like approximate timestep pipeline parallelism.

Clean request measurements and observer-heavy temporal diagnostics are kept in
separate result directories.  GPU launchers refuse to run if any process is
already present on physical GPUs 4--7 and always export
`CUDA_VISIBLE_DEVICES=4,5,6,7`.

The working contract is the user-provided
`poc_flashvep/reports/dllm_temporal_comm_and_async_pp_poc_spec.md` in the
primary checkout; its SHA-256 is recorded in `SPEC_PROVENANCE.md`.

The dInfer diagnostics are committed locally as `6cce007` and exported as a
portable patch under `patches/` so the project branch is self-contained.
