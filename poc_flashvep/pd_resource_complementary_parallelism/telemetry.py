"""Same-window read-only NVML-GPM samples for Hopper devices."""
import asyncio
import json
import math
import os
import threading
import time

import pynvml as nv


METRICS = {
    "graphics_or_compute_pct": nv.NVML_GPM_METRIC_GRAPHICS_UTIL,
    "sm_util_pct": nv.NVML_GPM_METRIC_SM_UTIL,
    "sm_occupancy_pct": nv.NVML_GPM_METRIC_SM_OCCUPANCY,
    "tensor_util_pct": nv.NVML_GPM_METRIC_ANY_TENSOR_UTIL,
    "hmma_util_pct": nv.NVML_GPM_METRIC_HMMA_TENSOR_UTIL,
    "dram_bw_pct": nv.NVML_GPM_METRIC_DRAM_BW_UTIL,
    "nvlink_rx_mib_s": nv.NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC,
    "nvlink_tx_mib_s": nv.NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC,
}


def get(device, previous, current):
    sample = nv.c_nvmlGpmMetricsGet_t()
    sample.version = nv.NVML_GPM_METRICS_GET_VERSION
    sample.sample1, sample.sample2 = previous, current
    sample.numMetrics = len(METRICS)
    for idx, statistic in enumerate(METRICS.values()):
        sample.metrics[idx].metricId = statistic
    nv.nvmlGpmMetricsGet(sample)
    result = {}
    for idx, name in enumerate(METRICS):
        record = sample.metrics[idx]
        if record.nvmlReturn == nv.NVML_SUCCESS and math.isfinite(record.value):
            result[name] = record.value
        else:
            result[f"{name}_status"] = int(record.nvmlReturn)
    result["vram_B"] = nv.nvmlDeviceGetMemoryInfo(device).used
    return result


def sample_worker(output, stop, interval):
    nv.nvmlInit()
    handles = {}
    current = []
    with output.open("w", buffering=1) as stream:
        try:
            for uuid in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
                device = nv.nvmlDeviceGetHandleByUUID(uuid)
                previous, next_value = nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()
                current.extend((previous, next_value))
                nv.nvmlGpmSampleGet(device, previous)
                handles[uuid] = (device, previous, next_value, time.monotonic())
            while not stop.wait(interval):
                for uuid, (device, previous, next_value, earlier) in handles.items():
                    nv.nvmlGpmSampleGet(device, next_value)
                    now = time.monotonic()
                    record = get(device, previous, next_value)
                    stream.write(json.dumps({"uuid": uuid, "begin_s": earlier, "end_s": now, **record}) + "\n")
                    handles[uuid] = device, next_value, previous, now
        except nv.NVMLError as error:
            stream.write(json.dumps({"error": str(error), "selected": list(handles)}) + "\n")
        finally:
            for item in current:
                nv.nvmlGpmSampleFree(item)
            nv.nvmlShutdown()


async def poll_gpu(output, interval=0.25):
    stopped = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(sample_worker, output, stopped, interval))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        stopped.set()
        await worker
        raise
