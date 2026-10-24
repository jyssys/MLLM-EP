# Physical distributed scaling

| Visible keys | GPUs | Cache condition | Process starts | Local graph wall, ms | Distributed graph wall, ms | Operator speedup, 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| 2049 | 2 | repeated KV | 3 | 0.0470 | 0.0781 | 0.619× [0.479,0.693] |
| 2049 | 2 | pressure | 3 | 0.0530 | 0.0833 | 0.644× [0.547,0.702] |
| 8193 | 2 | repeated KV | 3 | 0.1276 | 0.1178 | 1.097× [0.932,1.188] |
| 8193 | 2 | pressure | 3 | 0.1358 | 0.1268 | 1.078× [0.959,1.141] |
| 16385 | 2 | repeated KV | 3 | 0.2317 | 0.1668 | 1.389× [1.372,1.408] |
| 32769 | 2 | repeated KV | 3 | 0.4413 | 0.2811 | 1.574× [1.457,1.633] |
| 32769 | 2 | pressure | 3 | 0.4475 | 0.2917 | 1.539× [1.423,1.599] |
| 65409 | 2 | repeated KV | 3 | 0.8569 | 0.4891 | 1.753× [1.685,1.789] |
| 65409 | 2 | pressure | 3 | 0.8621 | 0.4978 | 1.733× [1.661,1.774] |
| 2049 | 4 | repeated KV | 3 | 0.0472 | 0.0753 | 0.631× [0.584,0.706] |
| 2049 | 4 | pressure | 3 | 0.0518 | 0.0984 | 0.561× [0.383,0.682] |
| 8193 | 4 | repeated KV | 3 | 0.1277 | 0.0943 | 1.358× [1.305,1.462] |
| 8193 | 4 | pressure | 3 | 0.1357 | 0.1004 | 1.353× [1.283,1.398] |
| 16385 | 4 | repeated KV | 3 | 0.2319 | 0.1142 | 2.031× [2.023,2.044] |
| 32769 | 4 | repeated KV | 3 | 0.4402 | 0.1773 | 2.485× [2.402,2.527] |
| 32769 | 4 | pressure | 3 | 0.4484 | 0.1847 | 2.428× [2.405,2.454] |
| 65409 | 4 | repeated KV | 3 | 0.8531 | 0.2806 | 3.041× [2.975,3.086] |
| 65409 | 4 | pressure | 3 | 0.8619 | 0.2873 | 3.000× [2.974,3.021] |
| 2049 | 8 | repeated KV | 3 | 0.0480 | 0.0882 | 0.555× [0.457,0.621] |
| 2049 | 8 | pressure | 3 | 0.0525 | 0.0869 | 0.605× [0.585,0.627] |
| 8193 | 8 | repeated KV | 3 | 0.1283 | 0.0990 | 1.316× [1.112,1.499] |
| 8193 | 8 | pressure | 3 | 0.1359 | 0.1005 | 1.356× [1.253,1.413] |
| 16385 | 8 | repeated KV | 3 | 0.2319 | 0.1156 | 2.022× [1.917,2.113] |
| 32769 | 8 | repeated KV | 3 | 0.4418 | 0.1373 | 3.241× [2.894,3.551] |
| 32769 | 8 | pressure | 3 | 0.4475 | 0.1396 | 3.206× [3.178,3.225] |
| 65409 | 8 | repeated KV | 3 | 0.8578 | 0.1983 | 4.349× [3.940,4.682] |
| 65409 | 8 | pressure | 3 | 0.8598 | 0.1953 | 4.404× [4.340,4.439] |


Evidence: `LIVE_MEASURED_OPERATOR`.
The speedup compares local FA4 with the full distributed graph wall interval.
NCCL, stable merge, query transfer, and completion are included.
The interval uses the slowest measured rank for each paired repetition.
The 95% interval resamples process-start means.
It does not treat loop, layer, or repeated kernel calls as independent requests.

Repeated-KV tests can favor a shard that fits in L2.
The pressure control touches a separate 128 MiB buffer before each timed operation.
That buffer operation is outside the timing interval for both arms.
The test is cache-pressure sensitivity, not counter-certified cold-cache behavior.

`capacity_by_gpu_count.csv` reports actual live operator KV payload and page padding.
Full-model memory equivalents multiply by 96 and remain `MODEL_UPPER_BOUND`.
They do not prove that an integrated 96-namespace distributed server fits or runs.
Reserved allocator memory can exceed valid shard payload.
Later confirmatory records also retain pre-timing allocated and reserved CUDA bytes.

The primary balanced split and 75%-on-rank0 split use the same original state.
They test generic shard imbalance.
No added loop-aware planner improves that generic implementation here.

The pinned RLT model has no matched TP2/4/8 path.
Those baselines are `UNSUPPORTED_TP_BASELINE_IN_PINNED_RLT`.
This is not a claim that Ouro lacks TP in every runtime.
`DP_replica_scaling.csv` records actual independent replica cohorts.
Their throughput is scale-out hardware throughput, not B1 memory extension or SLO capacity.

There are two collectives per tested attention operator.
Repeating them over 24 layers and four loops would require 192 collectives per token.
That multiplication is structural accounting, not a measured full-model collective count.
Application payload is not measured NVLink wire traffic.
The full serving integration remains `NOT_RUN_FULL_SERVING`.

## Strong local baseline

| Visible keys | GPUs | Best local S | Local wall, ms | CP wall, ms | CP speedup over strong local, 95% CI |
| --- | --- | --- | --- | --- | --- |
| 2049 | 2 | 4 | 0.0502 | 0.0833 | 0.603× [0.520,0.659] |
| 2049 | 4 | 4 | 0.0502 | 0.0984 | 0.510× [0.370,0.664] |
| 2049 | 8 | 4 | 0.0502 | 0.0869 | 0.578× [0.561,0.598] |
| 8193 | 2 | 8 | 0.0724 | 0.1268 | 0.571× [0.510,0.608] |
| 8193 | 4 | 8 | 0.0724 | 0.1004 | 0.720× [0.684,0.743] |
| 8193 | 8 | 8 | 0.0724 | 0.1005 | 0.720× [0.670,0.751] |
| 32769 | 2 | 8 | 0.1370 | 0.2917 | 0.470× [0.436,0.491] |
| 32769 | 4 | 8 | 0.1370 | 0.1847 | 0.742× [0.735,0.750] |
| 32769 | 8 | 8 | 0.1370 | 0.1396 | 0.982× [0.975,0.988] |
| 65409 | 2 | 8 | 0.2224 | 0.4978 | 0.447× [0.429,0.458] |
| 65409 | 4 | 8 | 0.2224 | 0.2873 | 0.774× [0.768,0.780] |
| 65409 | 8 | 8 | 0.2224 | 0.1953 | 1.139× [1.121,1.151] |


This table uses the cache-pressure diagnostic and one representative real model state per context.
It compares three independent local processes with three independent distributed group restarts.
The confidence interval uses independent-process resampling.
It does not pair arbitrary restart ordinals as if they were request pairs.
Best S is an offline measured tuning envelope, not a new online policy.
The comparison does not establish full-model E2E speedup.
