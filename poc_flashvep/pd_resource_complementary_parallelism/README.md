# Resource-complementary PD discovery

Reproduce from the repository root using the package `poc_flashvep.pd_resource_complementary_parallelism`. The campaign ran on `henn-0`, driver 580.126.16, H200 NV18 and vLLM 0.26.0+cu129 with BF16, FA3, paged KV, CUDA Graph and NIXL 1.4.1. Driver and dependency details are in [environment.json](results/environment.json). The frozen weight and tokenizer checksums are in [model_receipt.json](results/model_receipt.json). The study uses its original model weights.

## Workload

Obtain Qwen/Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218` and place the local checkpoint at `$PD_RUN_ROOT/models/qwen3-8b`. The secondary Qwen2.5-7B-Instruct snapshot reuses the checkpoint in the EADA environment. Its individual file receipts are preserved. Direction C uses Qwen/Qwen3-30B-A3B at `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. [Qwen3-8B pinned checkpoint](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) · [MoE pinned checkpoint](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39)

The input `qwen_128.json` artifact is the earlier EADA cohort, SHA-256 `c3472d660ffa61988e04205a1dcbb02b0ff7737ece729a1fdd001cb5ba85b904`. The pinned EADA cohort builder selects 32 official records each from Qasper, HotpotQA, NarrativeQA and GovReport. Use that prior receipt and builder to recover the exact cohort from the original LongBench v1 ZIP and its pinned prompt templates. The present workload applies Qwen3 non-thinking formatting and controlled length. It truncates records and appends additional document text when needed. These are performance traces. [EADA frozen cohort and builder](https://github.com/jyssys/MLLM-EP/tree/07892cbe7193366ec422f1df82006c6e69f91cec/poc_flashvep/encoder_assisted_decode_attention) · [LongBench dataset revision](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672)

```bash
ns=poc_flashvep/pd_resource_complementary_parallelism
pkg=poc_flashvep.pd_resource_complementary_parallelism
export PD_RUN_ROOT=/path/to/research-data
export PD_PYTHON=/path/to/isolated/runtime/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VLLM_USE_FLASHINFER_SAMPLER=0
"$PD_PYTHON" -m "$pkg.workload" --source /path/to/qwen_128.json \
  --model "$PD_RUN_ROOT/models/qwen3-8b" --out "$PD_RUN_ROOT/frozen" --requests 24 64 128
```

Compare the output receipts to [Qwen3 workload receipt](results/qwen3_workload_receipt.json). The open-loop arrival offsets use a fixed Poisson seed. All tested fixed model completions have at least 16 tokens. The independent P-only capacity witness requests one output token to isolate prefill service.

## Dense serving and profiling

On this host the existing project GPU epoch handled selected burner transitions, identified owners and restored leased GPUs. Use the equivalent authorized lifecycle on a different host. The chosen H200s were 4 and 5 for two-GPU dense trials, 4–6 for the same-three-GPU placement study and 4–7 for true MoE PD. The same-architecture physical comparisons retained the same topology. Keep other work off the test GPUs.

```bash
bash "$ns/run_dense.sh" native
bash "$ns/run_dense.sh" dp2
bash "$ns/run_dense.sh" dp2-small-chunks 2048
bash "$ns/run_dense.sh" dp2-no-chunk
bash "$ns/run_dense.sh" pd-cross
bash "$ns/run_dense.sh" pd-layout-probe
bash "$ns/run_dense.sh" pd-deflect
bash "$ns/run_dense.sh" p-reference 8192
bash "$ns/run_dense.sh" p-reference 16384
bash "$ns/run_dense.sh" p-reference 32768
bash "$ns/run_dense.sh" p_tp2_d1
bash "$ns/run_dense.sh" p2_d1
bash "$ns/run_dense.sh" p1_d2
```

Use the environment's supported C++ compiler for vLLM compilation. The light observer uses the CUDA activity API and writes concurrent-kernel intervals, coverage and calibration receipts. A clean/profile/clean comparison judges its perturbation. The worker extension exposes its RPC only on the localhost serving endpoints in this measurement.

```bash
site_packages=$("$PD_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
nvidia="$site_packages/nvidia"
export PD_ACTIVITY_LIBRARY="$PD_RUN_ROOT/libpd_kernel_activity.so"
"${CXX:-g++}" -std=c++17 -O3 -shared -fPIC "$ns/kernel_activity.cc" \
  -I"$nvidia/cuda_cupti/include" -I"$nvidia/cuda_runtime/include" -I"$nvidia/cuda_nvcc/include" \
  -Wl,-rpath,"$nvidia/cuda_cupti/lib" "$nvidia/cuda_cupti/lib/libcupti.so.12" \
  -o "$PD_ACTIVITY_LIBRARY"
bash "$ns/run_dense.sh" p-light
bash "$ns/run_dense.sh" steady 8192
bash "$ns/run_dense.sh" steady-p-only
```

Native NIXL cross-layer options and deflection are local baseline controls. The deflection driver sends requests to local D prefill when a queued P arrival sees no current D request. The original published elastic-serving artifacts have a separate availability record. [Prior work audit](../reports/pd_resource_complementary_prior_work.md)

## Physical operator and resource tests

```bash
model="$PD_RUN_ROOT/models/qwen3-8b"
trace="$PD_RUN_ROOT/frozen/16384x16-n24.json"
"$PD_PYTHON" -m "$pkg.operators" --model "$model" --trace "$trace" \
  --rows 1024 4096 8192 --contexts 8192 16384 --batch 1 4 8 --repeats 15 --spatial \
  --out "$PD_RUN_ROOT/primary.jsonl"
"$PD_PYTHON" -m "$pkg.operators" --model "$model" --trace "$trace" \
  --rows 4096 8192 --contexts 8192 16384 --batch 1 4 --repeats 31 --targeted \
  --out "$PD_RUN_ROOT/targeted.jsonl"
"$PD_PYTHON" -m "$pkg.operators" --model "$model" --trace "$trace" \
  --rows 8192 --contexts 16384 --batch 1 --repeats 31 --overlap-quanta 256 512 1024 \
  --out "$PD_RUN_ROOT/quanta.jsonl"
"$PD_PYTHON" -m "$pkg.spatial_pairs" --model "$model" --trace "$trace" \
  --repeats 9 --out "$PD_RUN_ROOT/spatial.jsonl"
```

The native server used no Green SM partition. The spatial physical runs validate SM assignments and CUDA driver event timing. The D study and C's expert-sharded physical build, routing and replay have separate reproduction contracts. [A](../reports/pd_operator_level_parallelism_discovery.md) · [B](../reports/pd_destination_aware_computation_discovery.md) · [C](../reports/pd_moe_resource_sharing_discovery.md) · [D](../reports/pd_heterogeneous_prefill_parallelism_discovery.md)

## Raw results and figures

```bash
mkdir -p /tmp/pd-native-replay
tar -xzf "$ns/results/raw_native.tar.gz" -C /tmp/pd-native-replay
python -m "$pkg.analysis_native" --raw-root /tmp/pd-native-replay --out /tmp/pd-native-reduced --figures
python -m "$pkg.serving_figures" --summary /tmp/pd-native-reduced/native_summary.json \
  --out /tmp/pd-native-reduced/figures
python -m "$pkg.reduce_physical" --in "$ns/results/operator_primary.jsonl.gz" \
  --out /tmp/pd-a.json --figures /tmp/pd-a-figures
python -m "$pkg.reduce_physical" --in "$ns/results/operator_targeted.jsonl.gz" --out /tmp/pd-a-target.json \
  --tiled "$ns/results/operator_quanta.jsonl.gz" --figures /tmp/pd-a-figures
python -m "$pkg.reduce_physical" --in "$ns/results/operator_quanta.jsonl.gz" --out /tmp/pd-a-quanta.json
python -m "$pkg.verdict_figure"
```

Matplotlib is needed for the physical and verdict plots. Figures use actual observed, isolated or analytical values where identified. The archive manifest contains hashes for raw result receipts. GPU-side model, input-text and LLM tensor files remain external. The experiment spec froze gates before the physical reruns. [Spec](../reports/pd_resource_complementary_parallelism_poc_spec.md) · [Final report](../reports/pd_resource_complementary_parallelism_final_summary.md)
