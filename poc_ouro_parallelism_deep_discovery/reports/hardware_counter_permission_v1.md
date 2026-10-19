# Hardware-counter access: source and permission screen

The installed Nsight Compute executable is `/usr/local/cuda-12.8/bin/ncu`.
Its reported version is 2025.1.1.0.
Its absence from `PATH` did not mean the tool was absent.

The [read-only screen](../artifacts/counter_permission_screen_v1.json) records `RmProfilingAdminOnly=1` and effective UID1002.
The process has neither `CAP_SYS_ADMIN` nor `CAP_PERFMON`.
The resolved executable has no file capabilities or setuid bit.
The loaded driver reports 570.211.01.

[NVIDIA's permission guide](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters) describes restricted counter access and the required privileges.
Its R565-and-later rule includes `CAP_PERFMON` as an alternative privilege.
Neither applicable privilege exists in the observed process.
The newer R610 capability-node procedure does not describe this loaded driver.

These observations predict restricted counter access under the documented rule.
They are not an observed `ERR_NVGPUCTRPERM` experiment.
No target profiling run, CUDA context, privileged command, or driver change occurs.
The campaign continues without claiming actual HBM counters.

The local help also reports default cache control `all` and clock control `base`.
A future authorized residency experiment must account for those profiler changes.
Otherwise, the profiler could change the cache behavior that the experiment seeks to measure.
This is a design constraint, not a measured observer effect here.

Weight payload, allocated KV capacity, encoded storage size, and modeled read volume remain separate from measured HBM traffic.
No bandwidth or cache-miss conclusion follows from this permission screen.

Reproduce the read-only inventory from the campaign worktree:

```bash
/home/esjung/.venvs/ouro-deep-discovery/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.counter_permission_screen --output artifacts/NEW_COUNTER_PERMISSION_SCREEN.json
```

The script refuses an existing output file.
It starts no profiler target and changes no permission.
