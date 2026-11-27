"""Qualify EPD GPU counters against matched clean cohorts using the pinned runner."""
import argparse
import asyncio
from contextlib import contextmanager
import math
import random
import sys
import threading
import time

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_wide_decode_batch_token_split import online


@contextmanager
def sample_gpm(output, roles, period_s):
    """Query read-only device-level GPU Performance Monitoring and memory counters."""
    import pynvml as nvml

    counter_names = (
        "SM_UTIL",
        "SM_OCCUPANCY",
        "ANY_TENSOR_UTIL",
        "DRAM_BW_UTIL",
        "PCIE_TX_PER_SEC",
        "PCIE_RX_PER_SEC",
        "NVLINK_TOTAL_RX_PER_SEC",
        "NVLINK_TOTAL_TX_PER_SEC",
    )
    counter_ids = {name: getattr(nvml, "NVML_GPM_METRIC_" + name) for name in counter_names}
    stop = threading.Event()
    samples = []
    errors = []
    units = {}

    def capture():
        resources = []
        try:
            nvml.nvmlInit()
            try:
                devices = {role: nvml.nvmlDeviceGetHandleByIndex(index) for role, index in roles.items()}
                snapshots = {}
                for role, device in devices.items():
                    if not nvml.nvmlGpmQueryDeviceSupport(device).isSupportedDevice:
                        raise RuntimeError(f"GPM unsupported for {role}")
                    a = nvml.nvmlGpmSampleAlloc()
                    resources.append(a)
                    b = nvml.nvmlGpmSampleAlloc()
                    resources.append(b)
                    nvml.nvmlGpmSampleGet(device, a)
                    snapshots[role] = (a, b, time.monotonic())

                while not stop.wait(period_s):
                    wave = {"begin_monotonic": time.monotonic(), "roles": {}}
                    for role, device in devices.items():
                        previous, current, begin = snapshots[role]
                        nvml.nvmlGpmSampleGet(device, current)
                        end = time.monotonic()
                        settings = nvml.c_nvmlGpmMetricsGet_t()
                        settings.version = nvml.NVML_GPM_METRICS_GET_VERSION
                        settings.numMetrics = len(counter_ids)
                        settings.sample1 = previous
                        settings.sample2 = current
                        for i, metric_id in enumerate(counter_ids.values()):
                            settings.metrics[i].metricId = metric_id

                        report = nvml.nvmlGpmMetricsGet(settings)
                        values = {}
                        metric_errors = {}
                        for name, metric in zip(counter_ids, report.metrics[:report.numMetrics], strict=True):
                            if metric.nvmlReturn or not math.isfinite(metric.value):
                                values[name] = None
                                metric_errors[name] = int(metric.nvmlReturn)
                            else:
                                values[name] = metric.value
                                units[name] = metric.metricInfo.unit.decode("utf-8")
                        memory = nvml.nvmlDeviceGetMemoryInfo(device)
                        wave["roles"][role] = {
                            "sample_window": [begin, end], "gpu_memory_used_bytes": memory.used,
                            "gpu_memory_total_bytes": memory.total, "values": values,
                            "metric_error_codes": metric_errors,
                        }
                        snapshots[role] = (current, previous, end)
                    wave["end_monotonic"] = time.monotonic()
                    samples.append(wave)
            finally:
                for sample in resources:
                    nvml.nvmlGpmSampleFree(sample)
                nvml.nvmlShutdown()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")

    thread = threading.Thread(target=capture, name="eada-gpm-sampler", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
        rt.write(output, {
            "scope": "NVML GPM device-wide counter windows. DRAM bandwidth utilization is reported in percent. "
                     "This does not identify one operator's HBM traffic or infer future slack.",
            "roles": roles, "counter_ids": counter_ids, "units": units, "samples": samples,
            "period_seconds_lower_bound": period_s, "errors": errors,
            "joined": not thread.is_alive(),
        })


def capacity_probe(args, run, url, manifests, execute, group, gpm_period, role, reserve_gib):
    from ..mllm_epd_prefill_heavy_token_split import metrics
    from ..mllm_epd_prefill_heavy_token_split.ipc_ec_control import ownership_empty

    (_, device, port, _), = (entry for entry in rt.ROLES if entry[0] == role)
    label = f"{role}_arena_{reserve_gib}gib"
    receipt = run / "state" / f"{label}_ready.json"
    program = __package__ + ".reserve_encoder_memory"
    uuid = dict(rt.nvidia("--query-gpu=index,uuid"))[str(device)]
    reserve = group.start(label, [sys.executable, "-m", program, "--reserve-gib", str(reserve_gib),
                                  "--role", role, "--receipt", str(receipt)], CUDA_VISIBLE_DEVICES=uuid)
    try:
        deadline = time.monotonic() + 90
        while not receipt.exists() and time.monotonic() < deadline:
            if reserve.poll() is not None:
                break
            time.sleep(.1)
        if not receipt.exists() or reserve.poll() is not None:
            rt.write(run / "state" / f"{label}_failure.json", {
                "status": "INITIALIZATION_FAILED", "requested_GiB": reserve_gib, "returncode": reserve.poll(),
                "process_log": f"logs/{label}.log",
            })
            return False
        rows = rt.read(manifests[rt.MAIN100])["requests"]
        random.Random(online.ORDER_SEED).shuffle(rows)
        rows *= args.repeat_main100
        with sample_gpm(run / "raw" / f"{label}_gpm.json", {role: device}, gpm_period):
            asyncio.run(execute(rows, [url], label, args.qps[0], args.seed, None, defer_raw=True,
                                unique_media_occurrence=args.repeat_main100 > 1,
                                metadata={"method": "NativeStrong", f"same_{role}_resident_GiB": reserve_gib,
                                          "scope": "Physical capacity lower-bound probe with stock EPD serving"}))
        ownership_empty(label, "cuda_ipc")
        result = metrics.summarize(label)
        rt.write(run / "state" / f"{label}_{role}_memory.json",
                 rt.http_json(port, "campaign/worker_info", timeout=30)[0])
        print("CAPACITY_PROBE", result, flush=True)
        rt.require(result["primary_execution_valid"] and reserve.poll() is None, f"{role} arena probe incomplete")
        return True
    finally:
        group.stop(label, reserve)
        group.processes.remove((label, reserve))


def run_with_observer(gpm_period, main100_cycles, arena_sizes, capacity_only, fallback_gib, capacity_role="encoder"):
    """Keep the historical worker setup, arrival generation and clean/profiler gate."""
    parse_legacy = online.parse_args
    run_legacy = online.measured_cohorts
    launch_legacy = rt.launch_models
    owned = {}

    def choose_args():
        args = parse_legacy()
        if (args.profile == capacity_only or len(args.qps) != 1 or args.subset != "all" or args.repeat_main100 != 1
                or args.screen or args.replay or args.mode != "native-strong"):
            raise ValueError("Select one full-Main100, Natural NativeStrong QPS and either --profile or --capacity-only")
        args.repeat_main100 = main100_cycles
        return args

    def launch(group, *args, **kwargs):
        owned["group"] = group
        return launch_legacy(group, *args, **kwargs)

    def measure(args, run, url, manifests):
        from ..mllm_specific_disaggregation_discovery.native_epd import nixl_metrics

        client = rt.module("client")
        execute = client.execute
        roles = {role: gpu for role, gpu, _, _ in rt.ROLES}
        rt.write(run / "state" / "native_observer.json", {
            "sampler_source_sha256": rt.sha(__file__), "period_seconds": gpm_period,
            "main100_cycles": main100_cycles, "roles": roles, "arena_sizes_GiB": arena_sizes,
            "capacity_role": capacity_role, f"fallback_{capacity_role}_GiB": fallback_gib,
            "scope": "GPM runs during matched capacity arms, or in the profiling arm of a clean triplet.",
        })

        async def observe(rows, urls, tag, *positional, **keywords):
            if capacity_only and tag != "warmup":
                with sample_gpm(run / "raw" / f"{tag}_gpm.json", {capacity_role: roles[capacity_role]}, gpm_period):
                    return await execute(rows, urls, tag, *positional, **keywords)
            if not tag.endswith("_profile"):
                return await execute(rows, urls, tag, *positional, **keywords)
            rt.write(run / "raw" / f"{tag}_nixl_before.json", nixl_metrics())
            with sample_gpm(run / "raw" / f"{tag}_gpm.json", roles, gpm_period):
                result = await execute(rows, urls, tag, *positional, **keywords)
            rt.write(run / "raw" / f"{tag}_nixl_after.json", nixl_metrics())
            return result

        client.execute = observe
        try:
            run_legacy(args, run, url, manifests)
            rt.write(run / "state" / "worker_after_full_load.json", {
                role: rt.http_json(port, "campaign/worker_info", timeout=30)[0]
                for role, _, port, _ in rt.ROLES
            })
            for size in arena_sizes:
                ready = capacity_probe(args, run, url, manifests, execute, owned["group"], gpm_period,
                                       capacity_role, size)
                if not ready and fallback_gib:
                    ready = capacity_probe(args, run, url, manifests, execute, owned["group"], gpm_period,
                                           capacity_role, fallback_gib)
                rt.require(ready, f"{capacity_role} memory arena initialization failed. See the per-size receipts.")
        finally:
            client.execute = execute

    online.parse_args = choose_args
    online.measured_cohorts = measure
    rt.launch_models = launch
    try:
        online.main()
    finally:
        online.parse_args = parse_legacy
        online.measured_cohorts = run_legacy
        rt.launch_models = launch_legacy


def main():
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--gpm-period", type=float, default=.4)
    options.add_argument("--main100-cycles", type=int, default=2, choices=(1, 2, 3))
    options.add_argument("--reserve-arena-gib", "--reserve-encoder-gib", type=int, action="append", default=[])
    options.add_argument("--fallback-arena-gib", "--fallback-encoder-gib", type=int)
    options.add_argument("--capacity-role", choices=("encoder", "prefill"), default="encoder")
    options.add_argument("--capacity-only", action="store_true")
    config, other = options.parse_known_args()
    if not .1 <= config.gpm_period <= 5:
        options.error("--gpm-period must be in [.1, 5] seconds")
    if (any(size <= 0 for size in config.reserve_arena_gib)
            or len(set(config.reserve_arena_gib)) != len(config.reserve_arena_gib)
            or config.capacity_only and not config.reserve_arena_gib):
        options.error("Choose positive distinct arena sizes, and at least one for --capacity-only")
    if config.capacity_role == "prefill" and not config.capacity_only:
        options.error("Prefill arena probes require --capacity-only")
    if config.fallback_arena_gib is not None and (len(config.reserve_arena_gib) != 1
            or not 0 < config.fallback_arena_gib < config.reserve_arena_gib[0]):
        options.error("Choose one primary arena and a smaller positive fallback")
    command = sys.argv.copy()
    try:
        sys.argv[1:] = other
        run_with_observer(config.gpm_period, config.main100_cycles, config.reserve_arena_gib, config.capacity_only,
                          config.fallback_arena_gib, config.capacity_role)
    finally:
        sys.argv[:] = command


if __name__ == "__main__":
    main()
