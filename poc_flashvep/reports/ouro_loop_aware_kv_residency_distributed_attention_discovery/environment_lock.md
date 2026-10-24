# Environment lock

The checkpoint is ByteDance/Ouro-1.4B Base.
The revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The runtime revision is `4a350344027960a48384626aa646f1765fda6d4f`.
`source_lock.json` stores the actual model configuration and tracked runtime diff.
Each worker stores its process identity and experiment-source snapshot.

The Python executable is `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python`.
The environment supplies PyTorch 2.11.0+cu129 and flash-attn-4 4.0.0b30.
Nsight Systems is available.
Nsight Compute 2025.1.1 is installed outside the executable search path.
Its path is `/opt/nvidia/nsight-compute/2025.1.1/ncu`.
The study attempts hardware counters with that executable.
`counters/summary.json` records the outcome.
Unavailable counters remain missing evidence.
Logical bytes are not measured HBM traffic.

All eight authorized devices are H100 80GB HBM3 GPUs.
`gpu_identity.txt` records their physical IDs, UUIDs, PCI addresses, and usable memory.
Every worker checks its CUDA UUID against the registered physical GPU.
Distributed ranks check the same identity mapping.

Only verified owned burner sessions can be stopped.
An unrelated GPU process causes the lease check to fail.
Independent jobs restore a burner when their GPU has no remaining job.
Quiet comparison groups reserve all eight GPUs without burners.
The reservation excludes burner interference from those measurements.

The initial unrelated staged-diff SHA256 is `7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.
The study preserves that index state.
Previous reports and raw results remain unchanged.
