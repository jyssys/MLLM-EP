# Pinned official source

Retrieved 2026-09-08, separate checkouts under the study result root `refs/`.

| System | Official repository | Pinned commit | Commit date |
|---|---|---|---|
| Layered Prefill | https://github.com/scale-snu/layered-prefill | `053f80e5201a7c0ab56e468e3d578907a2ca9cc3` | 2026-03-09 |
| FastPP | https://github.com/Sys-KU/FastPP | `cb3ce5b39c085a671b2c79477ac6d673b55f8eac` | 2026-08-27 |
| NanoFlow | https://github.com/efeslab/Nanoflow | `f179a907828b87e042126e585880301ff5b2c62a` | 2025-09-17 |
| NanoFlow original main | https://github.com/efeslab/Nanoflow/tree/main | `d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57` | original C++ artifact |
| NanoFlow official dev-h100 | https://github.com/efeslab/Nanoflow/tree/dev-h100 | `915790ea862d1ddd52a8871282c1eb5be88f1391` | Qwen MoE TP/EP native source |

FastPP HEAD explicitly fixes ALP attention profiling. Use this revision, not an older uncorrected predictor benchmark.
NanoFlow current source defaults to SM90 and is not assumed identical to its original A100 paper artifact.
All README shell snippets must be adapted to the authorized physical GPU allocation 4–7; never execute `--gpus all`.
Do not run installation scripts that change global sysctls, `/usr/local`, or a validated environment without a scoped alternative.

Papers retrieved from primary sources:

- https://arxiv.org/pdf/2510.08055v2
- https://www.usenix.org/system/files/osdi26-hwang.pdf
- https://www.usenix.org/system/files/osdi25-zhu-kan.pdf

PDF checksums/page counts are saved in `papers/pdf_manifest.json`.
