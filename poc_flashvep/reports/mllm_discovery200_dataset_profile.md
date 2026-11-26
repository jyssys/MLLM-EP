# Discovery200 dataset freeze and input profile

Discovery200 contains 200 requests with images as the only non-text modality, original question text, and answer references. The frozen request mix is 100 unchanged Main100 requests, 40 MMMU, 30 MathVista, and 30 MMReason. The grouped Discovery150 and Validation50 use disjoint exact image, context, question and source-qualified ID groups. Tier1 contains 24 Discovery requests. Selection and grouping used publisher metadata and Qwen processor geometry before candidate results.

| Source | Official split | Source pool eligible | Selected | Discovery | Validation | Tier1 | Published dataset license |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original Main100 | Frozen handoff | 100 | 100 | 75 | 25 | 12 | Inherited per-source provenance |
| MMMU | validation | 896/900 | 40 | 30 | 10 | 5 | Apache-2.0 |
| MathVista | testmini | 1000/1000 | 30 | 23 | 7 | 4 | CC-BY-SA-4.0 |
| MMReason | testmini | 1000/1000 | 30 | 22 | 8 | 3 | Apache-2.0 |
| Total | | | 200 | 150 | 50 | 24 | |

Main100 came from the original handoff SHA256 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1` and the verified server-relocated SHA256 `1d7221b2391c5239ca0a85a109d3b93d46ab1a03c4a42f4caaf44d0a9a437151`. The relocated manifest supplies request fields, images, references, prompts and token metadata. The study adds a source label, split, Tier1 flag and isolation group. [Original handoff](../mllm_epd_handoff/README.md)

- MMMU uses official publisher revision `876ce5cb130f7f7e290ce4d9984357737d4db5cf`, over all 30 subject validation shards. Zero-shot text follows the official choice/question template and preserves marked image positions. Four source examples had unresolved image references for the selected explicit in-context representation and were excluded before sampling. The 40 selected rows cover 24 subjects, 37 multiple-choice and three open-response questions. [Official MMMU dataset](https://huggingface.co/datasets/MMMU/MMMU/tree/876ce5cb130f7f7e290ce4d9984357737d4db5cf) · [Prompt construction](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/data_utils.py#L122) · [Published prompt formats](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/configs/llava1.5.yaml)
- MathVista uses official publisher revision `2b6ad69445fbb5695c9b165475e8decdbeb97747` and labeled `testmini`. The supplied `query` preserves answer format hints and choices. The selected rows span five broad task groups, including 18 multiple-choice and 12 free-form examples. The published `test` split holds labels back. [Official MathVista dataset](https://huggingface.co/datasets/AI4Math/MathVista/tree/2b6ad69445fbb5695c9b165475e8decdbeb97747) · [Official split description](https://mathvista.github.io/#dataset)
- MMReason uses author revision `cc6077a553a47efa8a9d9a3c649827b7bc5c85b4` and `testmini` Parquet. The author-supplied `cot_question`, image and final answer remain intact. The 30 selected examples cover six disciplines. The author Parquet supplies final references in this split and the paper separately describes reference-based intermediate step judging. [Author dataset](https://huggingface.co/datasets/HuanjinYao/MMReason/tree/cc6077a553a47efa8a9d9a3c649827b7bc5c85b4) · [MMReason paper](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_MMReason_An_Open-Ended_Multi-Modal_Multi-Step_Reasoning_Benchmark_for_MLLMs_Toward_ICCV_2025_paper.html)

## Selection and fold isolation

The seed is `20261127`. New cohorts rank fixed metadata buckets round-robin using source subject or discipline, image-count category and original question length. Stable SHA256 ordering supplies seeded tie-breaking. Each selected row has official references and all required images. The processor checks the available context with 2,048 output tokens reserved. Candidate eligibility yielded the first selected set without context exclusions, truncation or exact request replacement. Main100 stayed intact.

The initial v1 request-level split contained five related Main100 pairs across folds, sharing 43 individual image pixel hashes. The frozen v2 solves this by grouping connected examples across raw image hash, decoded-pixel hash, ordered-image context hash, document ID or PDF byte hash, normalized original question, and source-qualified original ID. The 190 groups consist of 180 singletons and ten pairs. The minimum amendment preserving all source quotas, including sub-source quotas within Main100, moves eight Main100 fold labels. The full 200 requests and exact 24 Tier1 IDs and order are identical between v1 and v2. The v2 validation check finds zero shared isolation keys. Any full-roster baseline measured before v2 can be reaggregated by request ID. The v2 folds govern subsequent discovery and validation.

Public benchmark examples and already studied Main100 examples can appear in model training or prior local research. This validation assesses comparisons across isolated requests in the frozen study.

## Processor geometry

Qwen2.5-VL-7B-Instruct, model and processor revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. New rows use the original processor's chat template, image grids, token spans and 3D MRoPE metadata. The processor records `input_ids`, `image_grid_thw`, `positions`, `rope_delta`, and modality type IDs. These are processor measurements. The encoder shape is structural until GPU execution records it.

| Cohort | Input mean | Input median | Input min–max | Visual median | Text median | Images median / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main100 | 11029.59 | 9867 | 556–30536 | 9836 | 31.5 | 3 / 60 |
| MMMU40 | 876.40 | 608 | 197–3906 | 442 | 119.5 | 1 / 5 |
| MathVista30 | 420.67 | 300 | 119–1473 | 190 | 88.5 | 1 / 1 |
| MMReason30 | 416.23 | 410 | 115–880 | 246 | 106.5 | 1 / 1 |

Across all 200 rows there are 1,367 image occurrences, 1,163,122 input tokens, 1,143,148 visual tokens and 16,240 text tokens. The remaining input consists of special tokens. The added reasoning sources have shorter inputs than Main100 and increase task and prompt diversity. Native Natural EOS uses the measured output token counts below. References are scoring inputs and stay outside inference payloads.

## Measured Native output lengths and materialized task scores

The Native Free Natural EOS run completed 200/200 requests at offered QPS 0.75. It produced 31,835 tokens, including one MathVista request that reached the 2,048-token output cap. The observed pooled median was 41 output tokens and p95 was 807.3. The grouped v2 report joins the full-roster execution by request ID.

| Cohort | Total output tokens | Mean | Median | Min–max | Output cap hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main100 | 5,524 | 55.24 | 25.5 | 2–601 | 0 |
| MMMU40 | 83 | 2.08 | 2 | 2–4 | 0 |
| MathVista30 | 10,669 | 355.63 | 188 | 48–2048 | 1 |
| MMReason30 | 15,559 | 518.63 | 404 | 96–1341 | 0 |

MMMU uses the publisher's concise response template. MathVista and MMReason provide longer output workloads even with short input sequences. The benchmark-specific prompts and the input/output distributions belong in the interpretation of Prefill versus Decode costs.

| Task | Native point score | Scored | Scope |
| --- | ---: | ---: | --- |
| MMMU | 55.00% | 40/40 | Upstream option/open-answer scorer with stable per-request fallback |
| MathVista | 50.00% | 30/30 | Diagnostic · official normalization with local extraction |
| MMReason | 0.00% | 30/30 | Diagnostic · normalized exact final answer |
| Mantis-Eval | 48.00% | 25/25 | Pinned evaluator, unchanged response |
| DocVQA | 0.00% | 13/13 | ANLS on unchanged response |
| MP-DocVQA | 4.29% | 25/25 | ANLS on unchanged response |
| MMLongBench-Doc | 2.70% | 37/37 | Diagnostic · upstream scorer on raw response with official extraction unavailable |

The points use distinct task metrics. A local score of zero can reflect the extraction protocol and does not establish failure of the upstream judge. Figure 1 shows source counts, token ratios, input/output/image distributions and scores, with diagnostic rows hatched and marked. [Figure 1 PDF](../mllm_specific_disaggregation_discovery/results/figures/figure1_discovery200_dataset_profile.pdf) · [Figure 1 PNG](../mllm_specific_disaggregation_discovery/results/figures/figure1_discovery200_dataset_profile.png) · [Request geometry and lengths CSV](../mllm_specific_disaggregation_discovery/results/figures/figure1_dataset_profile.csv) · [Native task score CSV](../mllm_specific_disaggregation_discovery/results/figures/figure1_native_quality.csv)


## Scoring scope

MMMU's official scorer parses option letters and normalizes open answers. It uses seeded random fallback for some unparsed multiple-choice outputs, which must be identified when comparing a stricter local parser. MathVista uses answer extraction and question-type-specific normalization. Qualify any local extraction coverage and distinguish it from the complete upstream extraction pipeline. MMReason's judge-assisted final equivalence and intermediate reasoning scores require the paper's assessment protocol. Local deterministic answer metrics provide narrower diagnostics. [MMMU evaluation code](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py) · [MathVista evaluation](https://github.com/lupantech/MathVista#evaluation) · [MMReason assessment](https://arxiv.org/abs/2506.23563)

## Stored artifacts and reproduction

The runtime manifests and media are local in `h200_discovery_runs/dataset/`. Use `grouped_v2/discovery200.json`, `grouped_v2/discovery150.json`, `grouped_v2/validation50.json` and `grouped_v2/tier1_24.json`. The runner uses absolute paths to the verified local raw image bytes. The Git registry contains request IDs, split, source revision and all SHA256 references. It excludes images, answer text, tensor arrays and publisher Parquet payloads. [Registry CSV](../mllm_specific_disaggregation_discovery/dataset_registry/discovery200_registry.csv) · [Freeze receipt](../mllm_specific_disaggregation_discovery/dataset_registry/dataset_freeze_receipt.json) · [Official shard receipts](../mllm_specific_disaggregation_discovery/dataset_registry/official_source_shards.json) · [Preparation receipt](../mllm_specific_disaggregation_discovery/dataset_registry/preparation_receipt.json)

Run `acquire --output <dataset>` and `dataset prepare --output <dataset>` under the handoff download environment. Run `dataset freeze --output <dataset> --main100 <relocated-main100> --model <verified-snapshot>` under the serving environment with CUDA disabled. Run `dataset isolate --initial <dataset> --output <dataset>/grouped_v2` and optionally `dataset export --initial <dataset> --output <registry>`. Module names are `poc_flashvep.mllm_specific_disaggregation_discovery.acquire` and `poc_flashvep.mllm_specific_disaggregation_discovery.dataset`. The 32 downloaded shard bytes match publisher LFS hashes and sizes. [Acquisition code](../mllm_specific_disaggregation_discovery/acquire.py) · [Manifest code](../mllm_specific_disaggregation_discovery/dataset.py)

Render the figure with `python -m poc_flashvep.mllm_specific_disaggregation_discovery.dataset_figure --manifest <dataset>/grouped_v2/discovery200.json --native <native-analysis> --output <figure-directory>`. The renderer checks the Native confirmation manifest hash, roster and input geometry before exporting.
