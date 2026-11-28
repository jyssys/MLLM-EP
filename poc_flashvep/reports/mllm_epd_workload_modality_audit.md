# Workload modality audit

## Frozen natural assets

The Main400 manifest at research commit `07892cbe7193366ec422f1df82006c6e69f91cec` has SHA-256 `7ac7d28ae1678499bee35ec20e317760555944cac7b4ca29eb46aeb3957b9592`. It contains 400 unchanged image-and-text requests and 5,004 image occurrences. The original Natural149 subset had 149 locally materialized rows. Pinned local parquet sources restored another 149 rows. Recovery fetched the nine remaining MP-DocVQA shards and 43 MMLongBench-Doc PDFs from their pinned revisions, bringing media-qualified coverage to 400/400. Each image file matches the canonical SHA-256 and each original token metadata record remains byte-identical. The restored material includes 2,731 distinct image file hashes. [Canonical manifest](https://github.com/jyssys/MLLM-EP/blob/07892cbe7193366ec422f1df82006c6e69f91cec/poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124/data/main_manifest_template_v2.json) · [Recovery receipt](../mllm_epd_prefill_heavy_modality_discovery/results/main400_media_receipt.json)

| Source | Main400 requests | Natural149 requests | Main400 image occurrences | Median images/request | Median LM input rows | Median visual rows | Median text rows | Median vision patches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DocVQA | 50 | 13 | 50 | 1 | 4,849 | 4,819 | 25 | 19,276 |
| Mantis-Eval | 100 | 25 | 254 | 2 | 2,544 | 2,460 | 65 | 9,838 |
| MP-DocVQA | 100 | 31 | 312 | 2 | 10,532 | 10,496 | 24 | 41,984 |
| MMLongBench-Doc | 150 | 80 | 4,388 | 26 | 14,870 | 14,784 | 34 | 59,136 |
| Total | 400 | 149 | 5,004 | 3 | 9,804 | 9,770 | 32 | 39,078 |

The source commits are retained in the row audit and in every manifest entry. Original parquet images were recovered from the specified row and image columns. PDF pages were rendered at 72 DPI RGB with unchanged page order and validated against the original derivative SHA-256. All original question strings, processed prompt bytes and image lists were retained. [DocVQA revision](https://huggingface.co/datasets/lmms-lab/DocVQA/tree/539088ef8a8ada01ac8e2e6d4e372586748a265e) · [Mantis-Eval revision](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval/tree/a3d92f9eb22c31d8c2d9d5b432c5a0ff1ef6dfd5) · [MP-DocVQA revision](https://huggingface.co/datasets/lmms-lab/MP-DocVQA/tree/e776eeade5a631d9ae18230ed3ab6213cd063dac) · [MMLongBench-Doc revision](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/tree/2ff6aa9237fc777b6627dc57a486e9225ac5fb86)

Across Main400, LM prompts span 105–30,648 rows. The natural median visual share is 99.39%. In ascending count bins 1, 2–4, 5–8, 9–16, 17–32 and 33+, the request counts are 50, 178, 22, 6, 96 and 48. The maximum observed image count is 76. Thus images per request and input rows can vary independently. The first four source-specific median LM rows range from 2,544 to 14,870. The structured per-request audit includes original and processor resolutions, the actual `grid_thw`, visual spans, visual/text/structural rows and patch counts. Patch totals are sums of frame × height × width from the recorded processor grids. [Per-request geometry](../mllm_epd_prefill_heavy_modality_discovery/results/workload_modality_rows.json) · [Images](../mllm_epd_prefill_heavy_modality_discovery/results/figures/dataset_image_count.png) · [Visual rows](../mllm_epd_prefill_heavy_modality_discovery/results/figures/dataset_visual_tokens.png) · [Visual and text fractions](../mllm_epd_prefill_heavy_modality_discovery/results/figures/dataset_visual_text_ratio.png)

## Natural cohorts for serving

Main100 and Natural149 retain their original ordering and benchmark questions. The nested `natural_mix_n24/n64/n128` and `main400_mix_n24/n64/n128` select by source-stratum rank using frozen seed 2026112807. Ranked upper cohorts are selected by input length, then balanced across available strata without reading any timing results.

| Natural cohort | Requests | Input rows min | Input rows median | Median images | Median visual share |
|---|---:|---:|---:|---:|---:|
| Natural149 | 149 | 556 | 12,668 | 20 | 99.40% |
| Main400 | 400 | 105 | 9,804 | 3 | 99.39% |
| Natural149 mix N24 | 24 | 564 | 13,461 | 18 | 99.39% |
| Natural149 upper N64 | 64 | 13,893 | 20,520 | 31 | 99.49% |
| Main400 mix N128 | 128 | 414 | 9,792 | 3 | 99.39% |
| Main400 upper N128 | 128 | 14,254 | 20,072 | 26 | 99.50% |

Natural149 is the earlier media-available subset and overrepresents long documents relative to Main400, whose 150 long-document requests give a 37.5% share. The 80 long-document requests in Natural149 give 53.7%. The upper cohorts are deliberate capacity-oriented slices of natural inputs. The manifests preserve the opening system text, image order, original questions, references and every visual token span. Serving runs can select fixed outputs or natural EOS. Actual emitted output lengths come from request traces. [Frozen cohort roster and SHA-256 receipts](../mllm_epd_prefill_heavy_modality_discovery/results/workload_receipts.json)

Two additional 24-pair controls contrast original requests with nearly matched LM geometry and very different per-image resolution. Candidate pairs are selected by increasing absolute LM input length difference, with request IDs breaking ties. Each request appears at most once in a pair of a given band.

| Natural pair band | Median LM rows, few / many | Median image count, few / many | Median visual rows per image, few / many | Paired LM-row gap, median / max |
|---|---:|---:|---:|---:|
| 14K–21K | 15,316 / 15,221 | 3 / 24 | 4,819 / 616 | 128 / 309 |
| 18K–32K | 23,350 / 23,168 | 5 / 40 | 4,837 / 613 | 205 / 437 |

The requests retain their original modalities, benchmark sources and ordering within each prompt. Source mix can differ between sides. These pairs test whether larger individual images change Encoder demand at near-matched LM length and visual share. Their geometry alone gives no Encoder timing result.

## Controlled input geometry

The target input bands are 4,096, 8,192, 16,384, 24,576 and 32,512. The last target leaves 256 output rows within the 32,768 cap. Each band targets 0%, 25%, 50%, 75% and 95% visual share with at most 24 distinct image-sequence controls. The 32,512-row 95% target has three available whole-image prefixes. Each selected image is intact. We choose a contiguous whole-image prefix of a natural request and use verified processor grids to generate the corresponding placeholders. Public LongBench passages supply the comparison text. The source text is clipped in token space and re-tokenized with the Qwen2.5-VL chat template. The recorded LM input rows are allowed a residual of at most two rows from the target. Controlled prompts use a summary request and carry no corresponding original answer labels. [LongBench source cohort provenance](https://github.com/jyssys/MLLM-EP/blob/5bdd6ef5e91840e37c776c2ad937866531e6a391/poc_flashvep/reports/pd_prefill_heavy_resource_motivation.md)

| 16K control | Requests | Input rows observed | Median images | Median visual rows | Median visual share |
|---|---:|---:|---:|---:|---:|
| Target 0% | 24 | 16,383–16,384 | 0 | 0 | 0.00% |
| Target 25% | 24 | 16,383–16,384 | 7 | 4,050 | 24.72% |
| Target 50% | 24 | 16,383–16,384 | 13 | 8,100 | 49.44% |
| Target 75% | 24 | 16,383–16,384 | 20 | 12,320 | 75.20% |
| Target 95% | 24 | 16,383–16,384 | 25 | 15,390 | 93.93% |
| Three larger images, target 50% | 24 | 16,383–16,384 | 3 | 8,193 | 50.01% |
| Eight or more images, target 50% | 24 | 16,383–16,384 | 13 | 8,100 | 49.44% |

The triplet control combines complete images from three distinct source requests and remains in the controlled cohort. The matched-count comparison pairs it with a whole-document image prefix using 8–200 images and nearly the same total visual rows. This is a testable geometry contrast. Actual Encoder Service Time and any performance difference require native stage measurements.

The mixed-traffic roster combines 24 unchanged Natural149 multimodal requests with 24 original LongBench user inputs under the pinned Qwen2.5-VL text-only template. Its text-side system wrapper and its arrival mix form a separate workload.

## Independent processor parity

A current pinned CPU processor rerun matched complete expanded input token IDs and the full image-grid list in 10/10 witnesses. It covered four natural strata, five 16K visual fractions and the triplet control. A 16K high-visual witness had 15,309 post-merger rows, 27 images and a 288,054,144-byte CPU pixel tensor. Its computed logical BF16 LM-ready embedding payload is 109,734,912 B, and its logical 28-layer BF16 prompt KV is 939,524,096 B. The resulting 8.56× is a logical full-prompt size ratio. Physical CUDA IPC transfer and image-complete prefix traffic have separate measurement boundaries. The 16K triplet witness had 8,193 visual rows and a 58,727,424-byte logical embedding payload at the same full KV size. [Processor witnesses](../mllm_epd_prefill_heavy_modality_discovery/results/processor_witness.json)

## Reproduction

Run the commands from the research checkout. Use the existing download environment for PyArrow and PyMuPDF and the pinned runtime environment for the native processor. Public downloads are optional when cache coverage is complete. All source assets and fresh manifests stay in the external run root.

```bash
pkg=poc_flashvep.mllm_epd_prefill_heavy_modality_discovery
ns=poc_flashvep/mllm_epd_prefill_heavy_modality_discovery
out=../h200_epd_modality_runs/dataset
cache=../mllm_epd_autosetup/hf_cache
main=../mllm_epd_autosetup/manifests_1795600931728841878/main100_manifest_template_v2.json
natural=../h200_memory_asymmetry_runs/natural149_manifest.json
model="$cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
HF_HUB_DISABLE_XET=1 ../mllm_epd_download_env/bin/python -m "$pkg.recover_media" \
  --repository ../MLLM-EP-mllm-epd --assets ../MLLM-EP-mllm-epd.filtered \
  --cache "$cache" --out "$out/media_recovery" --allow-download
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  ../mllm_epd_runtime_env/bin/python -m "$pkg.datasets" --main100 "$main" --natural149 "$natural" \
  --recovered "$out/media_recovery/main400_media_n400.json" --out "$out" --results "$ns/results" \
  --model-dir "$model" --passages ../h200_eada_runs/longbench/qwen_128.json --verify-hashes
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ../mllm_epd_runtime_env/bin/python -m "$pkg.processor_witness" \
  --model-dir "$model" --out "$ns/results/processor_witness.json" --natural "$out/main400_mix_n24.json" \
  --controlled "$out"/controlled_16384_v{00,25,50,75,95}_n24.json "$out/controlled_16384_v50_triplet_n24.json"
python -m "$pkg.workload_figures" --audit "$ns/results/workload_modality_rows.json" \
  --receipts "$ns/results/workload_receipts.json" --out "$ns/results/figures"
```
