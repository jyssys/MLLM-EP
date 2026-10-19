# SPT_NOENV source review and CPU regression

## Outcome and scope

`SPT_NOENV=1` is a documented control that preserves the environment area exposed through `/proc/PID/environ`.
It limits title storage to the command-line area.
It does not waive a process-lease requirement.
See the [immutable 1.3.7 documentation, lines 86–94](https://github.com/dvarrazzo/py-setproctitle/blob/389ed6f4e3ccfebb34a38893553992ea10cf5dc8/README.rst#L86-L94).

The new CPU regression reproduces the documented mismatch with installed setproctitle 1.3.7.
The job-scoped option prevents that mismatch in the same regression.
The unchanged strict lease rejects the default case and accepts the preserved case.
No model, native engine, Torch, vLLM, or GPU runs in either CPU child.

Base1.4 GPU v4 now completes with the strict leases unchanged.
Its runner and guard status files both report success.
The two worker lease files contain successful frontend and TP-rank checks.
This review reads those completion records only.
Raw cohort result analysis remains pending in this review.
These status checks establish no latency gain, candidate gain, or task quality result.

## Primary source mechanism

The official `version-1.3.7` tag resolves to commit `389ed6f4e3ccfebb34a38893553992ea10cf5dc8`.
The installed package reports version 1.3.7 and uses its C extension.
The structured record pins the installed package, metadata, and extension hashes.
This review does not prove a reproducible binary build from the official commit.

On Linux, the source selects the argv-clobber strategy.
Its default setup can extend the writable title area into contiguous environment strings.
It copies the logical C environment away from that area.
A nonempty `SPT_NOENV` skips that extension and copy.
See [spt_status.c, lines 186–218](https://github.com/dvarrazzo/py-setproctitle/blob/389ed6f4e3ccfebb34a38893553992ea10cf5dc8/src/spt_status.c#L186-L218).
The title function writes the buffer and clears the unused old region.
See [the same source, lines 321–356](https://github.com/dvarrazzo/py-setproctitle/blob/389ed6f4e3ccfebb34a38893553992ea10cf5dc8/src/spt_status.c#L321-L356).

The first `setproctitle` or `getproctitle` call invokes setup.
See [setproctitle.c, lines 31–67](https://github.com/dvarrazzo/py-setproctitle/blob/389ed6f4e3ccfebb34a38893553992ea10cf5dc8/src/setproctitle.c#L31-L67).
Setup caches its result once per process.
See [spt_setup.c, lines 414–438](https://github.com/dvarrazzo/py-setproctitle/blob/389ed6f4e3ccfebb34a38893553992ea10cf5dc8/src/spt_setup.c#L414-L438).
The option must precede that first setup.
An exec environment that contains the option meets this timing requirement.
A late assignment does not restore bytes that setup already overwrote.

Installed `WorkerProc` sets the title before `init_device`, then again after distributed groups exist.
See `vllm/v1/executor/multiproc_executor.py:624–633`.
Installed `set_process_title` forwards to the setproctitle package.
See `vllm/utils/system_utils.py:184–200`.
The first CPU title call already produces the mismatch without the option.
The second native title call is therefore not a necessary cause of the CPU failure.

## Reproducible CPU result

[The script](../scripts/setproctitle_noenv_red_green_cpu_v1.py) starts two fresh native Python subprocesses.
Each child imports the actual installed extension before runtime environment edits.
Neither child calls `configure_native_imports` before title setup.
The script checks the package version and the actual C function identity.
It checks all three guard keys before the first title call.
It then requests `VLLM::Worker` and `VLLM::Worker_TP0` in that order.

Each title must match both `getproctitle()` and `/proc/self/cmdline`.
The script exports guard presence and equality as booleans only.
It exports no environment values.
Each observation checks `LOOPED_KV_OUTPUT`, `CUDA_VISIBLE_DEVICES`, and `VLLM_WORKER_MULTIPROC_METHOD`.

| Exec setting | Logical keys after each title | Initial-area keys after each title | Strict self lease |
|---|---|---|---|
| Default, option absent | 3/3 present and matching | 0/3 present | Expected rejection |
| `SPT_NOENV=1` | 3/3 present and matching | 3/3 present and matching | Pass |

Both requested titles applied in both children.
Here, “initial-area” describes `/proc/PID/environ`, not a live view of Python's logical dictionary.
The [CPU artifact](../artifacts/setproctitle_noenv_red_green_cpu_v1.json) contains timestamps, booleans, assertions, and source hashes.
It also checks that the retained earlier fixtures and artifacts remain byte-identical.

Run this command from `/home/esjung/MLLM-EP-ouro-deep-discovery`:

```bash
/home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.setproctitle_noenv_red_green_cpu_v1 --output /home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/artifacts/setproctitle_noenv_red_green_cpu_v1.json
```

Use a new artifact path for a rerun.

## Limits and retained history

The documentation warns that earlier environment edits may interfere with initialization.
See [README, lines 51–55](https://github.com/dvarrazzo/py-setproctitle/blob/389ed6f4e3ccfebb34a38893553992ea10cf5dc8/README.rst#L51-L55).
The earlier fixtures called `configure_native_imports` before title use.
They did not assert that the requested title actually applied.
Their passing leases did not reproduce the full native failure.
This review does not identify which earlier operation caused the different fixture result.

The earlier failed hypothesis and nested-chain artifacts remain unchanged.
Their results are not rewritten as failures.
The new regression uses a different import sequence and stronger title assertions.
The option reduces available title space, so longer titles may truncate.
The short fixture titles fit this command line.
This regression does not measure a truncation limit or all native startup operations.

The v4 completion records belong to Base1.4 fixed-R4 TP2×DP2 only.
They are not evidence for Base2.6, adaptive execution, natural quality, or a candidate advantage.
[The structured review](../artifacts/setproctitle_noenv_source_review_v1.json) pins those records without analysis of cohort values.
The separate time ledger excludes the user pause from 17:07:51 to 17:11:01 UTC.
Only listed active segments count as this agent's work.
Those segments overlap coordinator work and do not add campaign elapsed time.
