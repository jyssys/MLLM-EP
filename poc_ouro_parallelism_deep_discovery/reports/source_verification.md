# Source verification checkpoint

Status: active campaign, not a final literature or novelty verdict.

## Record definitions

`analysis/ledger_checkpoint_04/index.json` indexes 98 source observations and 61 distinct arXiv work keys.
It also indexes 93 work/document keys and 73 search records.
Different versions of one arXiv work share one work key.
Different official implementation documents can retain separate URL keys.
These counts include metadata-only and partial reads.
They do not mean 61 complete paper reads.

The source registry retains each observation, locator, version, access window, and original qualifier.
Missing metadata remains unresolved.
A repository search that finds no code is not proof that no code exists.
The current generated index is separate from its preserved versioned inputs.

## Mechanism-critical primary checks

| Source | Checked material | Supported boundary | Empirical status here |
|---|---|---|---|
| Ouro paper and released Base checkpoints | Architecture/inference sections, loaded configs, official model code | Shared physical core, distinct depth state, original gate | Both sizes loaded and probed |
| Installed native vLLM Ouro | Forward, norms, Attention registration, compiler and worker paths | Actual fixed-R4 execution and TP ownership | Fresh TP1/2/4 hardware measurements |
| Pinned CDB repository | Original-gate engine, cache policy, online execution | Physical skipping differs from output selection and replay | Both-size fixed/adaptive tiny checks and natural pilots |
| [LoPA v2](https://arxiv.org/html/2512.16229v2) | Algorithm, cache commit, branch executor, selected tables/limitations | dLLM branch work does not establish independent Ouro recurrent passes | Source audit only |
| [LoopSpec v1](https://arxiv.org/html/2609.17184v1) | Branch timeline, verification, private state, CLI restrictions | Direct self-speculation prior, public single-device path | NOT_RUN |
| [WaveFront v2](https://arxiv.org/html/2609.23033v2) | Dynamic union execution, verification, KV fill, original-gate code | Mixed-depth ready unions already have direct prior | NOT_RUN |
| [FlashLoop v2](https://arxiv.org/html/2609.29812v2) | Sparse updates, attention approximation, cache switches, limits | Training-free does not imply exact or nonquantized primary results | NOT_RUN |
| [Looped Latent Attention v2](https://arxiv.org/html/2607.15456v2) | Codec equations, training, serving and capacity protocol | New trained codec is outside this campaign's proposed-method scope | NOT_RUN |
| [Parallel Loop Transformer v1](https://arxiv.org/html/2510.24824v1) | Cross-token dependency and training changes | Its parallel graph is not the released Ouro graph | NOT_RUN |
| [Enabling Dynamic Computation v1](https://arxiv.org/html/2610.09013v1) | Terminal-state alias and original-gate execution design | Terminal KV aliasing is a direct prior boundary | Public code not located in bounded checks |
| Numerical parallel-time sources | Finite correction, work, initialization, state dependencies | Asymptotic parallelism does not establish R4 break-even | Separate bounded actual-Ouro diagnostic |
| NanoFlow, AFD, and KernelFlume | Ready operators, ownership and resource organization | Generic pooling/disaggregation must remain matched baselines | Source and component controls, not a new engine |
| FlashNorm, Rammer, Welder, and Mirage | Algebra, tiles, synchronization, and numerical limits | Fresh partial dataflow alone is not recurrence-specific novelty | Source and CPU controls |
| Current ZipServ code | BF16 compressed K/V tile loader and API restrictions | Generic on-read decode exists, despite older paper future-work wording | NOT_RUN |

The detailed source files record exact author lists and repository revisions.
This table does not replace those records or claim full reads of every appendix.

## Corrections that matter

FlashLoop is training-free in the audited version.
Its main evaluated path still includes approximate updates and INT4 residual storage.
The BF16 cache switch exists but did not run here.

The CDB paper's main performance replay does not establish our online adaptive execution.
The campaign separately checks the actual original-gate path.
Native vLLM fixed R4 does not implement that adaptive path.

ZipServ's paper and current code describe different implementation stages.
The inspected code contains compressed-attention loads.
It does not establish paged-KV append support or online Ouro correctness.

Similar activation vectors do not establish exact reuse, safe state publication, or preserved free-running quality.
Source-reported single-GPU results do not establish physical multi-GPU performance in this campaign.

## Reproducible source map

- Generated registry: `analysis/ledger_checkpoint_04/source_registry.jsonl`.
- Selected-paper audit: [literature_nearest_audit.md](literature_nearest_audit.md).
- Numerical parallelism: [parallel_sampling_nearest_audit.md](parallel_sampling_nearest_audit.md).
- Ownership: [operator_ownership_nearest_audit.md](operator_ownership_nearest_audit.md).
- Dynamic state: [dynamic_runtime_prior_boundary.md](dynamic_runtime_prior_boundary.md).
- Tile execution: [tile_dataflow_prior_audit.md](tile_dataflow_prior_audit.md).
- Lossless KV: [lossless_kv_reopening_prior.md](lossless_kv_reopening_prior.md).

The campaign continues to inspect new primary leads.
No negative search proves that a new Ouro architecture is impossible.
