# Reproducing the memory-asymmetry campaign

Base is EADA commit `07892cbe7193366ec422f1df82006c6e69f91cec`. The native overlay uses vLLM `0.26.0+cu129`, PyTorch `2.11.0+cu129`, Python `3.14.6`, CUDA/NIXL and the existing BF16 Qwen revision. The offline data inventory and protocol are described in [workload_audit.md](workload_audit.md). Use a fresh external run directory. The code lazily reuses original model and image paths after digest verification.

```bash
python -m poc_flashvep.mllm_epd_memory_asymmetry_discovery.workload --help
python -m poc_flashvep.mllm_epd_memory_asymmetry_discovery.stress_workload --help
poc_flashvep/mllm_epd_memory_asymmetry_discovery/run_online_case.sh combined-natural CASE_A --sample-period 1
poc_flashvep/mllm_epd_memory_asymmetry_discovery/run_online_case.sh pressure-peer CASE_B --sample-period 1
```

`MLLM_MEMORY_RUNS`, `MLLM_MEMORY_ASSET`, `MLLM_MEMORY_PYTHON` and `MLLM_MEMORY_RECEIPT` override only the default local filesystem paths. The online study uses E/P/D on the verified GPUs 4/5/6. It holds the campaign lease, prechecks ports and process ownership, freezes the runtime overlay, and restores verified burners after shutdown. The runner never stops a user process that does not match the known burner fingerprint. Execute one lease at a time. `native_study --help` lists all options.

`--cycles 2` runs identical cold and then warm sequences after a separate runtime warmup. Timed outputs are Natural EOS unless `--output` is set, which makes a controlled fixed-length cohort. The stress case preserves original prompts but controls the request order and explicitly repeats prompts. The JSON contract records the independent start, offered load, UUID scope, allocator capacities and source digests. A separate environment flag `MLLM_EPD_STABLE_EC_AUDIT=1` captures CPU copies and native BF16 digests for numerical diagnosis. Leave it disabled for all performance runs.

Use the raw response and scheduler journal to regenerate reduced timing and figures. The stock baseline and cached arms use the same client timer. Read the study-specific fields under `workload` in each measurement configuration for cache-mode truth. Legacy top-level client configuration fields are inherited from the original no-cache runner. vLLM labels the installed offloading metrics `CPU_to_GPU` and `GPU_to_CPU`; under the peer spec these counters refer to the peer-GPU tier. `peer_storage.json` records actual device ordinals and byte capacity.

```bash
python -m poc_flashvep.mllm_epd_memory_asymmetry_discovery.online_summary \
  --run /path/to/CASE_A --run /path/to/CASE_B --out /path/to/online_summary.json
python -m poc_flashvep.mllm_epd_memory_asymmetry_discovery.online_figures \
  --summary /path/to/online_summary.json --out /path/to/figures
```

All raw HTTP request streams, renderer and transport logs, host inventory, scheduler samples and per-process memory records remain under the recorded external run root. Only reduced receipts, shareable scientific figures, and small physical measurements are versioned. Local task scorers preserve raw answers. Official extraction for MMLongBench-Doc remains unavailable without a separately authorized external evaluation contract.

For repeated server starts, `--shared-compiler-cache` reuses the compiled kernel artifacts under `epd_runtime_cache`. It does not retain model weights, vLLM state, prefix blocks, encoder cache entries or request ownership. Each paired start launches distinct processes, verifies their PIDs, warms the same calibration inputs, and begins the first trace with a new request scope. `paired_blocks` resamples complete server-start pairs. It emits no independent-start CI below three pairs. The other five GPU burners continue on their original GPUs throughout the three-GPU leases.

The [placement confirmation contract](results/placement_confirmation_contract.json) lists the three pair names, model budget and inference boundaries. In the driver, `combined-natural` is the E-VRAM arm and `combined-host-embedding-natural` the optimized pinned-host arm. Invoke `paired_blocks` with `--pair HOST GPU` three times and run it separately for `--cohort measured_1` and `measured_2`. The [run index](results/run_index.json) records source versions, failure markers and SHA-256 values for the external request and engine ledgers.
