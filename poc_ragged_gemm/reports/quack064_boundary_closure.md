# QuACK 0.6.4 tile-boundary closure

## Question and status

Does crossing only a grouped-GEMM M-tile boundary cause a reproducible >=5%
Qwen3 BF16 expert-kernel latency increase under QuACK 0.6.4?

`QUACK064_BOUNDARY_STATUS: NO-GO`

`REOPEN_RAGGED_GEMM: NO`

The aggregate aligned/boundary-heavy difference is +1.090%, and neither that
comparison nor the 1x/2x/3x boundary sweeps has a stable sign across three
independent rounds. The version change does not recover the missing staircase.

## Isolated environment

The existing vLLM environment was not modified. The successful environment is
`/home/esjung/.venvs/quack064-boundary-closure-cu128-clean`, created from
Python 3.12 without `--system-site-packages`.

| Component | Version |
|---|---|
| PyTorch | 2.11.0+cu128 |
| Torch CUDA | 12.8 |
| QuACK | 0.6.4 |
| CUTLASS DSL | 4.6.2 |
| apache-tvm-ffi | 0.1.13.post3 |
| cuda-python | 12.9.7 |
| GPU | H100 80GB, physical GPU 4 only |
| SonicMoE | `7396f3e604827d8186c2e16e64b28ee33d3defd0` |

A CUDA 13 isolated environment was also attempted first, but the host's 12.8
driver rejected PyTorch CUDA initialization. It was not used for measurement.
The driver-compatible CUDA 12.8 environment uses Sonic's required QuACK 0.6.4
and compatible CUTLASS DSL 4.6.2 and compiles/runs successfully.

## Runtime tile verification

The benchmark imports SonicMoE, then queries
`quack.gemm_config.default_config(cuda)` and
`cta_tile_shape_m(tile_m, cluster_m, device_capacity)` at runtime. The returned
H100 configuration is:

```text
GemmConfig(tile_m=128, tile_n=192, pingpong=True,
           cluster_m=2, cluster_n=1, device_capacity=9,
           is_dynamic_persistent=False, ...)
```

The actual CTA `BLOCK_M` is therefore **128**. Histogram construction uses this
queried value; 128 is not assumed by the closure script.

## Method

Primary shape: BF16, H=2048, I=768, E=128, K=8 context, G=32, primary N=4096.
Within each comparison N, G, weights, dtype, GPU, and stock up/SwiGLU/down
kernel path are unchanged. Only the per-expert histogram crosses a boundary.

Measured cases are aligned, boundary-heavy, and paired-expert offsets -1/0/+1
around 1x, 2x, and 3x `BLOCK_M`. Each case uses 20 warmups and 100 CUDA-event
measurements in each of three independently shuffled rounds. The preregistered
closure rule uses the largest *minimum* positive penalty shared by all three
rounds: GO >=5%, HOLD >=2%, otherwise NO-GO. This prevents selecting a favorable
single round post hoc.

## Results

### Aligned versus boundary-heavy

| Metric | QuACK 0.6.4 |
|---|---:|
| Aligned median | 0.211344 ms |
| Boundary-heavy median | 0.213648 ms |
| Aggregate relative difference | +1.090% |
| Per-round relative differences | -1.100%, +1.582%, +1.513% |

The aggregate difference is below 2%, and its sign is not stable across rounds.

### Boundary sweep

| Boundary | Round penalties | Median | Sign-consistent? |
|---|---:|---:|---:|
| 1x BLOCK_M | -1.336%, -1.221%, +0.041% | -1.221% | No |
| 2x BLOCK_M | +0.025%, -0.262%, +0.110% | +0.025% | No |
| 3x BLOCK_M | +3.935%, -2.405%, +0.311% | +0.311% | No |

The largest single-round magnitude is 3.935%, but it reverses to -2.405% in
another round. The largest reproducible positive boundary jump is therefore
**0.000%**. The sweep is non-monotonic/noisy and meets the NO-GO definition.

## Direct QuACK 0.5 comparison

| Version | Aligned | Boundary-heavy | Relative difference | Staircase gate |
|---|---:|---:|---:|---:|
| QuACK 0.5.0 | 0.264608 ms | 0.263456 ms | -0.435% | NO-GO |
| QuACK 0.6.4 | 0.211344 ms | 0.213648 ms | +1.090% | NO-GO |

QuACK 0.6.4 is faster in absolute time in this separately installed stack, but
the absolute difference also includes PyTorch/CUDA/CUTLASS environment changes
and is not attributed to QuACK alone. The causal conclusion is unchanged:
neither version produces a repeatable boundary penalty, and 0.6.4 remains below
the 2% HOLD threshold.

## Closure

Because the result is NO-GO, no QuACK 0.5/0.6.4 source-diff investigation,
additional shapes, threshold changes, or custom kernel were performed. The
Ragged GEMM direction remains closed for this Qwen3 BF16 H100 regime.

Artifacts:

* `raw.json`: all 3 x 100 timing samples and exact histograms
* `summary.json`: fixed-gate aggregation and QuACK 0.5 comparison
* `plot_quack064_boundary_closure.png`: aligned/boundary timing, actual boundary
  sweep latency, and round-wise penalty
