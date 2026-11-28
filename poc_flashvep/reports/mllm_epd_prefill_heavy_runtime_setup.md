# H200 EPD runtime and hardware

## Test environment

The experiment runs on `henn-0`, eight NVIDIA H200 GPUs, 143,771 MiB per device from NVML, driver 580.126.16 and NV18 links. Native E/P/D use indices 4/5/6 on NUMA 1 in one NVSwitch domain. The owner guard records GPU UUIDs and reserves the physical group for each launch. GPU 0–3 and 7 keep the approved utilization jobs while this group is leased.

The pre-existing isolated runtime supplies Python 3.14.6, PyTorch 2.11.0+cu129, vLLM 0.26.0+cu129, NIXL 1.4.1 and CUDA 12.9. The PyTorch runtime reports NCCL 2.28.9. Executed P/D logs report FlashAttention 3. Native EC selects the V1 model runner because installed V2 lacks the EC path. The startup completes compilation and CUDA Graph capture before warmup measurements. No package upgrade is involved.

Primary workers use the original Qwen2.5-VL-7B-Instruct BF16 checkpoint at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. The asset proof verifies the registered model and original Main100+Calibration32 manifest from `449c3e1873c65206435a6bb9411a889a62e55be7`. The research worktree is based on the native AR campaign at `5bdd6ef5e91840e37c776c2ad937866531e6a391`. Each launch freezes executable Python worker source in its external run directory. The first bootstrap smoke stopped before model serving because the sparse checkout lacked a legacy shared module. The recovered start includes the needed runtime packages.

## Executed data paths

- E performs original local media processing and compiled vision-only execution, with physical LM-ready GPU embeddings. The E worker receipt records 1,353,100,288 vision weight bytes, zero unique KV bytes, and 147,103,678,464 CUDA free bytes at startup before request work.
- P and D retain the original language model and configure 42 GiB KV reservations in the primary run. The original P token budget is 8,192, D is 8,192, S is 128. Both use TP1, Paged KV, FlashAttention and original MRoPE. The respective launch receipts and GPU-owned processes establish the executed values.
- P and D each load 15,231,233,024 language-weight bytes and retain the 1,353,100,288-byte vision module in this vLLM checkpoint load path. The encoder process holds zero LM weight bytes. P and D receive original GPU embeddings through the consumer connector during multimodal requests.
- A separate native Upper N64 epoch reserves 48 GiB per P/D GPU and records 51,534,364,672 actual KV storage bytes per model worker. The corresponding 42 GiB setting records 45,097,156,608 bytes. The same physical GPU group, APC and model settings carry over.
- Grid-only handoff carries the original processor metadata. A native request reaches P only after E returns. P imports an acknowledged CUDA IPC clone. D receives full prompt KV through NIXL and continues the original generation. Role isolation and online outcomes have separate raw receipts.
- APC is off, renderer cache is zero, and the client uses a different media occurrence identity per request. The inline CUDA IPC descriptor plus live same-request E embedding has a finite lease and is released after safe retirement. This is required handoff memory.

## Observation support

NVML GPM samples SM, Tensor Core, DRAM activity, PCIe and NVLink counters on the actual physical devices. Low-rate Prometheus reads report engine waiting, deferred requests, capacity, KV occupancy and counters. Kernel/copy recording compiles the verified CUDA 12.9 CUPTI collector. Nsight Systems and Compute were absent in PATH, and the host restricts privileged hardware profiling through RmProfilingAdminOnly. Results use observed intervals and GPM device counters.

Full raw contexts and image assets live under the external run root. The analysis distinguishes kernel union, engine state, host transfer enclosures, hardware counter windows and request-lifetime observations. [Observer and source contract](../mllm_epd_prefill_heavy_modality_discovery/README.md)

The first colocated DP3 start failed before serving because the shared API observation wrapper looked for a processor-metadata directory even for requests without EC handoff headers. The retry supplies an unused per-replica metadata directory to that wrapper. DP3 runs the original model and preprocessing in each independent replica and uses round-robin request assignment. No EC/KV disaggregation connector participates in this control.

## Architecture scope

The exercised GPU IPC contract asserts a 3,584-wide BF16 LM-ready Qwen2.5-VL embedding. The installed Qwen3-VL model returns concatenated DeepStack vision representations and injects their additional per-layer features into early LM layers. A second-model EPD baseline requires changing and validating the transport and consumer state contract. Pinned Qwen2.5-VL-7B-Instruct results carry a single-architecture scope. [Audited implementation boundaries](mllm_epd_modality_specific_opportunity.md)
