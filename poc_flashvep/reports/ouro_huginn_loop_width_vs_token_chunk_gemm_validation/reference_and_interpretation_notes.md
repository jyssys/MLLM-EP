# Reference and interpretation notes

## Matrix geometry

NVIDIA describes GEMM cost as `2 × M × N × K` operations.
Its arithmetic-intensity model includes matrix reads and output writes.
The guide also explains tile efficiency and resource limits.
These principles motivate the experiment. They do not prove the measured bottleneck.

Source: [NVIDIA matrix-multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

This campaign calculates that matrix model from the actual checkpoint dimensions.
The calculation is structural accounting. It is not an HBM counter.
Cache reuse and fused implementations can change physical traffic.
No result claims an optimal-kernel lower bound.

The separate cold-matrix roofline uses NVIDIA's H100 SXM specification.
The page lists 1,979 BF16 TFLOP/s with sparsity and 3.35 TB/s memory bandwidth.
The dense model uses 989.5 TFLOP/s.
These values are theoretical specifications, not measured sustained rates.
Source: [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

The model assumes one HBM read for each GEMM matrix and one output write.
Cache reuse, fusion, Attention, and launch costs remain outside this model.
The report must not call its predictions measured runtime or a universal lower bound.

## Kernel selection

cuBLASLt supports reusable algorithm choices for specified matrix geometries.
This campaign tests a finite set of valid choices.
It uses the same 64 MiB workspace limit and 32-candidate limit for each shape.
The candidate set includes heuristic choices, valid no-split-K variants, and canonical-shape algorithm reuse.
Canonical reuse executes the actual unpadded row count.

Source: [NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgogetheuristic).

The current public documentation describes a newer library version.
Each raw experiment records the library that it actually loaded.
The observed cuBLASLt version is 120901. The build uses CUDA 12.8 headers.
The Python environment uses Torch 2.11.0+cu129.
The library file hashes and algorithm attributes remain in the evidence.

## What the geometry can explain

A 64-token suffix in Huginn R32 has 32 dependent recurrent passes.
A full 2048-token Loop has one recurrent pass.
Both cases contain 2048 recurrent token rows.
The Token case has more tail Attention work.
The Loop case also transfers much more prompt Coda work to D.
These differences require separate accounting.

A faster GEMM does not remove the recurrent Hidden State dependency.
It can reduce padding, launch overhead, and inefficient matrix execution.
Compatible-request batching can also increase the Token matrix width.
Such batching requires ready requests and valid KV allocations.
Projection-only batch geometry does not establish either requirement.

## Scientific limits

The best tested kernel is not the globally optimal GEMM kernel.
Projection equality covers the recorded input tests.
Model qualification covers the recorded N8 and N32 cohorts.
Neither test proves equality for every possible input.

CUDA Event enclosures can contain stream waits and host launch gaps.
They are not physical kernel-union measurements.
Marker-free serving does not provide an exact Service TTFT timestamp.
The report must retain `NOT_MEASURED` for that field.

A generic logical-layer implementation can reproduce an identical Loop frontier.
That case is an equivalent mechanism.
A measured Loop-versus-Token geometry benefit does not prove exclusive Loop-specific novelty.
