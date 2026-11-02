# Memory denominators

NVML total HBM and CUDA-visible managed capacity are different measurements.
Do not substitute the nominal 80 GiB model name for either value.

| Measurement | Bytes | Binary GiB |
| --- | ---: | ---: |
| NVML total | 85,520,809,984 | 79.6474609375 |
| CUDA-visible total in owner snapshots | 85,028,372,480 | 79.1888427734375 |
| Difference | 492,437,504 | 0.4586181640625 |

The read-only NVML check found the same total on all eight devices.
The owner records obtain CUDA capacity from the native CUDA memory query.
The difference is outside that CUDA-managed capacity.
It is not free space that the KV allocator can reserve.

Earlier generated `percent_HBM` columns divide by the recorded CUDA-visible total.
They do not divide by the larger NVML total.
`physical_hbm_accounting.csv` supplies both explicit denominators.
It retains bytes, GiB, percentage of NVML total, and percentage of CUDA-visible capacity.
It does not change any original memory snapshot.

Allocator totals contain their tensor-storage components.
Active KV occupies part of the preallocated KV pool.
Do not add either contained quantity to its containing allocation.
The opaque allocator residual is not an independently measured CUDA Graph pool.
Per-worker initial and close snapshots are not simultaneous cross-worker peaks.
