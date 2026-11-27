# Encoder-assisted decode attention on three H200 GPUs

[Experiment contract](../reports/mllm_encoder_assisted_decode_attention_poc_spec.md) · [Primary-source audit](prior_work.md)

The research namespace contains the BF16 FlashAttention-3 operator, physical peer-copy and graph controls, original Qwen stage adapters, page ownership, open-loop serving, and receipt analysis. Publisher weights, original images, token tensors, and the substantial raw traces are external to Git. The local run root for this campaign is `../h200_eada_runs`.

Run commands from the research worktree with `../mllm_epd_runtime_env/bin/python`. Each measured GPU job uses the previously audited GPU lease in `poc_flashvep.mllm_specific_disaggregation_discovery.gpu_epoch` or the original NativeStrong launcher, which takes its own lease. Only the authorized burner processes on leased GPUs are paused. The per-run receipts specify GPU UUIDs, role mapping, toolchain, arguments and exits.

## Boundaries of the comparison

- The native NIXL/HTTP control and the in-process PyTorch engine have different timing boundaries. A0 within the prototype uses the same P/D stages and the same paged FlashAttention-3 integration as the EADA modes. Both are actual open-loop workloads.
- Hardware counters and CUPTI traces describe their profiled run. An unqualified observer pair stays diagnostic. The exact causal merge uses FA3 partition log-normalizers, and the local comparison recomputes full attention using an optimized native FA3 kernel.
- Paged capacity depends on actual occupied pages, admission, and the GPU stage outputs. A large touched arena present during Vision Encoding is a measured co-residency bound.

## Reproduce the paired serving study

The NativeStrong commands and expected raw arrival files are in the [resource report](../reports/mllm_encoder_assisted_decode_attention_motivation.md). Use the pinned manifest and snapshots listed in `run_study.sh`. Run from a worktree containing the tracked dependencies and use a fresh run directory for each lease. The original H200 worktree stores the model and dataset adjacent to the repository. Copy the study driver before starting a lease so later source edits cannot change the shell commands of a running experiment.

```bash
run_root=../h200_eada_runs
runtime=../mllm_epd_runtime_env
study=poc_flashvep/encoder_assisted_decode_attention
lease=poc_flashvep.mllm_specific_disaggregation_discovery.gpu_epoch
mkdir -p "$run_root/drivers"
cp "$study/run_study.sh" "$run_root/drivers/reproduction.sh"
sha256sum "$run_root/drivers/reproduction.sh"

env PYTHONDONTWRITEBYTECODE=1 EADA_ATTENTION_ENGINE=graph \
  "$runtime/bin/python" -m "$lease" --run-dir "$run_root/repro_graph_witness_lease" \
  --devices 4,5,6 --wall-seconds 420 -- \
  bash "$run_root/drivers/reproduction.sh" "$runtime" "$run_root" graph-witness _repro

env PYTHONDONTWRITEBYTECODE=1 EADA_ATTENTION_ENGINE=graph \
  "$runtime/bin/python" -m "$lease" --run-dir "$run_root/repro_graph_online_lease" \
  --devices 4,5,6 --wall-seconds 3000 -- \
  bash "$run_root/drivers/reproduction.sh" "$runtime" "$run_root" online _repro
```

The `text`, `capacity`, `tight-capacity`, `stock-dp`, `profile-stages` and `repeat-medium-*` driver sections use the same role mapping. Use a separate lease for each section. `run_operator_sweep.sh` and `run_internvl.sh` contain the physical comparison and second-model diagnostics. These tests use real H200 peer access. The raw model fixtures remain on the experiment host.

[The online manifest](results/raw_online/manifest.json) records exact SHA-256 data, immutable shell snapshots, failed launches, and complete compressed timing traces. To recreate the Python reductions from the archive, unpack each `compressed_file` into its relative `file` path under a scratch root, point `analyze_online.py --runs` at the resulting run directories, and pass its `summary.json` to `plot_online.py`. [The frozen arrivals](results/arrivals/) store the two Primary stock traces used for every matched image run. [Stock resource](results/native_extended_resource_ledger.json), [quality](results/online/quality.json), and [physical operator](results/operator/) receipts capture their corresponding boundaries separately.
