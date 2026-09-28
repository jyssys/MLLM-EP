# Q2-2 causal visual expert-path PoC

This bounded extension tests exact same-path visual causal sensitivity and
functional drop-one damage. It uses GPUs 2,3,6,7 only and makes no latency
claim. Raw instrumented runs are append-only and host-local; frozen manifests,
compact evidence, reports, and commands are committed.

The authoritative source is
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/q2_2_causal_vision_conditioned_expert_path_poc_spec.md`.

See `artifacts/reproducible_commands.json` for the measured pipeline and
`reproduce.sh` for the cleanup trap and bounded contract test.
