# InternVL3 native representation and APC control

This control uses the published InternVL3-8B snapshot at `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` and its original tile processor, image positions, and chat template. The four cases were selected in the prior EADA campaign from the unchanged Main100 manifest at projected image counts 1, 3, 8 and 16. Their actual counts are 1, 3, 6 and 16. The vLLM 0.26.0 V2 runner on one H200 used BF16, eager, B8192, 128-row APC blocks, 8 GiB KV and 16-token fixed greedy generation.

The loaded vision tower has 0.566 GiB of parameters, the projector has 0.051 GiB and the LM has 14.180 GiB. The LM has 28 layers, 4 KV heads and head dimension 128. Each LM-ready BF16 image row occupies 7,168 bytes and each full-layer KV row occupies 57,344 bytes.

## Representation witness

The raw image path used a new APC salt for each draw and cleared both scheduler and worker encoder caches before each run. The first native draw captured the actual post-projector tensors, which were copied through CPU storage and supplied unchanged to the stock vLLM embedding input path. The second raw image draw served as the independent native A/A control. The injected requests retained the original template and image positions.

| Main100 request | Images | LM-ready embedding (MiB) | Full prompt KV (MiB) | Native image embedding GPU enclosure (ms) | Injected greedy and embedding parity |
|---|---:|---:|---:|---:|---|
| DocVQA 15382 | 1 | 22.75 | 185.12 | 23.02 | 16/16 tokens, byte-exact |
| Mantis m_17 | 3 | 40.25 | 325.50 | 41.64 | Stock injection rejected ragged rows |
| MP-DocVQA 44820 | 6 | 136.50 | 1,095.94 | 144.38 | 16/16 tokens, byte-exact |
| MMLongBench 984 | 16 | 196.00 | 1,574.13 | 223.40 | 16/16 tokens, byte-exact |

All four native A/A cases matched prompt IDs, post-projector BF16 bytes and greedy output. All three accepted injections matched those same quantities. The measured KV-to-embedding payload ratio was 8.03–8.14×. The actual maximum case had 28,784 prompt rows and 28,672 visual rows.

The first GPU attempt failed in vLLM's `InternVLImageEmbeddingInputs` validation on the three-image Mantis example, with visual tensor shapes 1,792×3,584 and 768×3,584 drawing the rejection. The earlier CPU test had identical raw/injected renderer token IDs and placeholder offsets on the very same input. The failure therefore names a model-side input boundary. The follow-up GPU run gathered native A/A and representation data for Mantis and restricted injection to the uniform image tensor shapes accepted by the stock implementation. Full ragged injection remains unsupported on this tested path.

## Whole-prefix APC control

Two distinct original MP-DocVQA requests 55287 and 55279 share two images in the same order. The model's processor produced 6,732 and 6,713 prompt rows, and the first differing token is at index 6,702. The warm native request retained the first request's APC salt while the cold comparison used a fresh salt. Both draws reset the independent worker encoder cache.

The cold path had zero cached prompt rows. The warm path reused exactly 6,656 rows, equivalent to 364 MiB of logical KV, and produced the same 16 greedy tokens. The observed pixel-processing call fell from 44.23 to 22.97 ms and the instrumented serial generation wall time went from 516.03 to 361.26 ms. These are one case and one pair in a local cache. The language-model hooks observed zero V2 runner calls, so separate CUDA P-only cost is `NOT_MEASURED`. The APC counter and observed stage enclosures remain direct evidence. Prefix-local APC compatibility, image embedding bypass and model weight asymmetry are supported on this second architecture. InternVL EPD, passive peer cache, natural EOS task quality and online generality remain `NOT_RUN`.

The separate Qwen text-only control can match the InternVL and Qwen2.5-VL storage geometry of 57,344 bytes per Full28 prompt row. It has no corresponding image-transform stage or image-identity reuse scope. The one-GPU InternVL control gives mechanism evidence. A resource-allocation advantage over 2P/1D, and a two-architecture online claim, require paired native-strength baselines and matched arrival, warmup and GPU-budget contracts.

[Raw InternVL model and call records](results/internvl3_control_raw.json) · [Reduced result and stock ragged failure log receipt](results/internvl3_control_summary.json)

## Reproduction

Use the pinned runtime and release a verified GPU with the existing lease helper. On this server the model snapshots and media stay outside the Git branch.

```bash
runtime=/home1/irteam/MLLM_PD/mllm_epd_runtime_env
run_root=/home1/irteam/MLLM_PD/h200_memory_asymmetry_runs
pkg=poc_flashvep.mllm_epd_memory_asymmetry_discovery
export PATH="$runtime/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VLLM_USE_FLASHINFER_SAMPLER=0
export HF_MODULES_CACHE="$run_root/internvl3_hf_modules"

"$runtime/bin/python" -m poc_flashvep.mllm_specific_disaggregation_discovery.gpu_epoch \
  --run-dir "$run_root/internvl3_reproduction_lease" --devices 4 --wall-seconds 185 -- \
  "$runtime/bin/python" -u -m "$pkg.internvl_control" \
  --model ../h200_eada_runs/models/internvl3_8b \
  --main100 ../mllm_epd_autosetup/manifests_1795600931728841878/main100_manifest_template_v2.json \
  --paired-manifest "$run_root/natural149_manifest.json" \
  --out "$run_root/internvl3_reproduction.json"
"$runtime/bin/python" -m "$pkg.internvl_summary" "$run_root/internvl3_reproduction.json" \
  --out "$run_root/internvl3_reproduction_summary.json"
```

For the original pre-retry ragged failure, the reducer can include `--ragged-failure-log "$run_root/internvl3_physical_lease/logs/command.log"`. The linked compact result preserves the exact failure text and the log's SHA-256.
