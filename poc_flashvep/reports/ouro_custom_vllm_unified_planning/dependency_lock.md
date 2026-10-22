# Dependency and provenance lock

## Source pins

These revisions were available during the 2026-10-22 audit.
Audit clones are read-only source references. They are not installed unified-engine environments.

| Component | Revision | Local evidence | License |
| --- | --- | --- | --- |
| vLLM v0.26.0 | `568afb3a13806beb53bb2e6bd518269357b237c0` | `/home/esjung/external/ouro-unified-planning-vllm026` | Apache-2.0 |
| vLLM-RLT current main | `4a350344027960a48384626aa646f1765fda6d4f` | `/home/esjung/external/ouro-unified-planning-rlt` | Apache-2.0 |
| RLT historical reference | `9d601ceaf9af63121c742df5f584467e23630a0b` | `/home/esjung/external/decode-first-vllm-rlt-upstream` | Inspect historical notices before copying |
| Official CDB | `abdfae0adbdd12be8c6123ccb34eee98a38afb39` | `/home/esjung/external/decode-first-cdb-upstream` | MIT root, file-specific Apache-2.0 and other notices |
| Prior eight-arm study | `2de9f955fe4eafc16bfb9086af7be07e37acfa74` | `/home/esjung/MLLM-EP-ouro-request-pp` | Preserve repository notices |
| Planning branch base | `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f` | `/home/esjung/MLLM-EP-github` | Existing repository license |
| Base model and tokenizer | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | HF cache `models--ByteDance--Ouro-1.4B/snapshots/<revision>` | Model release uses Apache-2.0 source. Archive weight/model-card terms before redistribution |

Never replace a source pin with an unqualified `main` dependency.
The official Ouro site identifies v0.26.0 as the last native-support release.
Do not upgrade vLLM during this program without a separate compatibility review.

## Preserved native environment

Path: `/home/esjung/.venvs/cp-composition-vllm026`.
Metadata inspection did not import a GPU model or allocate GPU memory.

| Item | Observed value |
| --- | --- |
| Python | 3.14.6, Anaconda build |
| vLLM | 0.26.0+cu129 |
| PyTorch | 2.11.0+cu129 |
| Transformers | 5.16.1 |
| NCCL distribution | nvidia-nccl-cu12 2.28.9 |
| NVIDIA driver | 570.211.01 |
| Standalone flash-attn distribution | Not installed under that distribution name |
| Standalone flash-attn-3 distribution | Not installed under that distribution name |

The missing standalone distributions do not prove missing vLLM bundled FlashAttention.
The actual selected attention backend, loaded CUDA libraries, custom all-reduce, and compiled extension hashes need a Phase 1 manifest.
Do not infer CUDA compatibility only from the driver number or package suffix.

Installed and upstream `vllm/model_executor/models/ouro.py` SHA256:
`cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff`.

Installed wheel metadata points to:
`file:///tmp/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl`.
Recorded wheel SHA256:
`7632856147650da3ed8d1652b1b05ffaadcc62ea8e910fdaa6f8ce055b201ebf`.
This is a metadata record. The audit does not claim that the original wheel still exists at that temporary path.
Obtain a verified wheel or a documented isolated build before claiming complete environment reproduction.

## Model configuration

| Field | Value |
| --- | ---: |
| Physical layers | 24 |
| Recurrent loops | 4 |
| Hidden dimension | 2048 |
| FFN intermediate dimension | 5632 |
| Query / KV heads | 16 / 16 |
| Head dimension | 128 |
| Vocabulary size | 49152 |
| Tied embedding/head | false |
| Configured context | 65536 |
| RoPE theta | 1000000 |
| RoPE scaling / sliding window | null / disabled |
| Precision | BF16 |

The checkpoint default early-exit threshold is 1.0. Native vLLM forward does not consume its gate.
The proposed Dynamic policy explicitly uses q=0.5, R2–R4.
Static execution explicitly runs all four loops. Do not rely on a threshold shortcut to enforce Static behavior.

## Physical GPU identity

| Physical GPU | UUID | Observed total HBM |
| --- | --- | ---: |
| 4 | GPU-6076e2f2-5b63-3761-5586-56ceb7df8139 | 81559 MiB |
| 5 | GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730 | 81559 MiB |
| 6 | GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 | 81559 MiB |
| 7 | GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b | 81559 MiB |

All four devices report NVIDIA H100 80GB HBM3.
The topology inventory reports NV18 for each permitted pair.
No bandwidth, copy/compute overlap, or free-HBM capacity benchmark ran during planning.
Burners occupied these devices. Their reserved memory is not Ouro's measured footprint.

## License and reproducibility obligations

- Preserve vLLM and RLT Apache-2.0 licenses, notices, and modified-file notices when code is reused.
- Read CDB `THIRD_PARTY_NOTICES.md`. Several cache, model, and scheduling files retain Apache-2.0 obligations.
- CDB also includes MIT evaluation examples from lm-evaluation-harness. Preserve those notices if copied.
- Record dataset revisions, licenses, prompt-template hashes, and evaluator hashes independently of the engine lock.
- Do not vendor model weights, large raw datasets, installed packages, or secrets into the planning commit.
- Record all new build flags and extension hashes in future experiment manifests.

The exact historical 400-request selection is fixed by:

| File | SHA256 |
| --- | --- |
| `evaluation_subset400.json` | `ea12d20fbb20661d8229e9d96d9198e2291b04b234bfd2bf805456076e03a509` |
| `frozen_prompt_ids.json` | `8cfd77a7cc541efbc06a77f52024519d5fe525fa512ec4110f56ad50988db09f` |

Both files remain under the prior eight-arm report directory.
Prompt IDs are frozen inputs. They do not authorize frozen outputs in accuracy or main serving results.
