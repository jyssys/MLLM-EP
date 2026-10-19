# Inter-pass weight residency: a conditional capacity screen

## Result and scope

Shared weights do not fit in the measured H100 L2 cache as one complete Ouro core.
That statement alone does not prove a cache miss or reject a different ownership design.
This screen instead bounds retained bytes under an explicit repeated-stream model.

The result is an `ANALYTIC_BOUND` inside that model.
It is not measured HBM traffic, an alias/copies timing bound, or an E2E speedup estimate.
It supplies no new architecture or quality result.

## Assumptions

Let W be one rank's distinct BF16 linear weights for one complete physical core.
Let C be available cache capacity and D the executed pass count.
The same weight stream repeats D times.
Each pass requires one fetch per distinct weight byte without caching.
The reference therefore transfers `D * W` logical bytes into this cache.

This model grants all nominal L2 capacity to those weights.
It permits optimal replacement and free bypass of uncached reads.
It ignores KV, activations, other weights, metadata, and code-cache competition.
These choices favor the residency proposal.
The model does not grant free prefetch traffic.

It excludes persistent registers, shared-memory residency, compression, operator reordering, and extra ready rows.
Those mechanisms need separate cost and dependency tests.
The model does not assert the native GEMM's actual fetch count or cache policy.

## Derivation

At a pass boundary, at most `min(C,W)` distinct weight bytes can survive in this cache.
Every other byte needs a counted fetch before its next use.
Moving that fetch into a prefetch kernel does not remove its bytes.

With an initially empty cache:

```text
minimum fetch bytes = D*W - (D-1)*min(C,W)
maximum reduction fraction = (D-1)*min(C,W) / (D*W)
```

With an optimally preloaded initial cache:

```text
minimum fetch bytes = D*(W-min(C,W))
maximum reduction fraction = min(C,W) / W
```

The second view favors a warm invocation.
Do not apply the cold first-pass expression to every cached decode token.
Neither expression identifies the incremental benefit over a real physical-copies run.
That run can also retain some weights between tokens.

The [CPU checker](../scripts/interpass_cache_screen.py) searches every replacement choice in 168 bounded toy cases.
Its search also grants bypass.
The exact minima match both expressions.
Five CPU tests pass.
This agreement concerns the stated model, not an H100 replacement policy.

## Pinned dimensions and numerical screen

Both pinned Base checkpoints have H=2048, I=5632, and 16 Q/KV heads.
Their physical layer counts are 24 and 48.
One physical layer contains 98 MiB of these BF16 projection matrices:

```text
Q/K/V/O: 4 * H * H elements
MLP gate/up/down: 3 * H * I elements
```

The screen divides those matrices by TP degree.
It excludes replicated norm vectors, gates, embedding, and LM-head weights from W.
Those exclusions give the retained-weight fraction a favorable denominator.
They do not define total model memory.

The existing hardware audit records 52,428,800 L2 bytes on physical GPUs4 and6.
This screen uses 50 MiB per rank as its stated H100-family assumption.
It does not assert that all 50 MiB can become a persisting set-aside.

| Model | TP | Matrix W/rank, MiB | Cold R4 maximum logical-fetch reduction | Warm R4 maximum logical-fetch reduction |
|---|---:|---:|---:|---:|
| 1.4B | 1 | 2352 | 1.594% | 2.126% |
| 1.4B | 2 | 1176 | 3.189% | 4.252% |
| 1.4B | 4 | 588 | 6.378% | 8.503% |
| 2.6B | 1 | 4704 | 0.797% | 1.063% |
| 2.6B | 2 | 2352 | 1.594% | 2.126% |
| 2.6B | 4 | 1176 | 3.189% | 4.252% |

The [machine artifact](../artifacts/interpass_cache_capacity_screen_v1.json) also retains D=2/3 and TP8 capacity projections.
No TP8 device or performance measurement exists.
An adaptive B1 trace selects different D cases.
This does not reduce its allocated depth-indexed KV slots automatically.

## Relation to native measurements

The v4 storage controls retain valid saved-graph provenance and observed prefill bindings.
They still differ numerically.
A fresh same-alias repeat also differs numerically.
This screen does not resolve either numerical cause.

It also does not convert those live times into causal cache evidence.
HBM counters remain unavailable.
No measured cache-hit rate enters this report.

The capacity result weakens a proposal based only on retaining the entire core across passes.
It does not reject selective residency with a measured benefit.
It does not reject a different distributed memory optimum merely because an aliased unroll can represent its graph.
Any surviving proposal must show its actual removed cost against a competent same-resource baseline.

## Primary hardware semantics

CUDA supports persistence hints and a bounded persisting L2 set-aside.
Concurrent kernels share that resource.
The supported maximum and access-policy window are device properties.
This campaign changes none of those settings.
See [CUDA12.9.1 Programming Guide, sections6.2.3.1–6.2.3.8](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#device-memory-l2-access-management).

## Reproduction

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest \
  poc_ouro_parallelism_deep_discovery.tests.test_interpass_cache_cpu -v

/home/esjung/.venvs/llada2-mini-reference/bin/python -m \
  poc_ouro_parallelism_deep_discovery.scripts.interpass_cache_screen \
  --output poc_ouro_parallelism_deep_discovery/artifacts/NEW_CACHE_SCREEN.json
```

Use a new output path.
The script refuses to replace an earlier artifact.
