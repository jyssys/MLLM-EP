"""Verify profiler-created sessions by process ancestry, not session ID alone."""
import json
import os
from pathlib import Path
import time

from .common import write


def lineage(pid, ancestor):
    result = []
    while pid > 1 and len(result) < 30:
        try:
            raw = Path(f'/proc/{pid}/stat').read_text()
        except FileNotFoundError:
            return None
        fields = raw[raw.rfind(')') + 2:].split()
        result.append(dict(pid=pid, parent=int(fields[1]), session=int(fields[3]),
                           start_clock_ticks=int(fields[19])))
        if pid == ancestor:
            return result
        pid = int(fields[1])
    return None


def install():
    from ..dense_ar_slo_layer_pd import watch
    original = watch.GPUWatch
    class AncestryWatch(original):
        def sample(self):
            import pynvml as n
            peaks, samples, foreign = {g: 0 for g in self.gpus}, 0, []
            owner = os.getpid()
            try:
                n.nvmlInit()
                all_handles = {g: n.nvmlDeviceGetHandleByIndex(g) for g in range(8)}
                handles = {g: all_handles[g] for g in self.gpus}
                ownership = {}
                with (self.folder / f'NVML-{self.attempt}.jsonl').open('a') as stream:
                    while not self.done.is_set():
                        start = time.perf_counter()
                        if samples % 5 == 0:
                            for gpu, handle in all_handles.items():
                                ownership[gpu] = []
                                for process in n.nvmlDeviceGetComputeRunningProcesses(handle):
                                    try:
                                        sid = os.getsid(process.pid)
                                        proof = lineage(process.pid, owner)
                                    except ProcessLookupError:
                                        continue
                                    row = dict(pid=process.pid, session=sid,
                                        owned_session=sid == os.getsid(0),
                                        owned_descendant=proof is not None, lineage=proof)
                                    ownership[gpu].append(row)
                                    if proof is None:
                                        foreign.append(dict(time=time.monotonic(), GPU=gpu, **row))
                        for gpu, handle in handles.items():
                            memory = n.nvmlDeviceGetMemoryInfo(handle)
                            peaks[gpu] = max(peaks[gpu], memory.used)
                            stream.write(json.dumps(dict(time=time.monotonic(), physical_GPU=gpu,
                                uuid=n.nvmlDeviceGetUUID(handle), used_HBM_bytes=memory.used,
                                sampled_utilization_percent=n.nvmlDeviceGetUtilizationRates(handle).gpu,
                                SM_clock_MHz=n.nvmlDeviceGetClockInfo(handle, n.NVML_CLOCK_SM),
                                memory_clock_MHz=n.nvmlDeviceGetClockInfo(handle, n.NVML_CLOCK_MEM),
                                clocks_event_reasons_mask=n.nvmlDeviceGetCurrentClocksEventReasons(handle),
                                temperature_C=n.nvmlDeviceGetTemperature(handle, n.NVML_TEMPERATURE_GPU),
                                power_W=n.nvmlDeviceGetPowerUsage(handle) / 1000,
                                processes=ownership[gpu], sampling_cpu_ms=1000 * (time.perf_counter() - start),
                                evidence='MEASURED_NVML_NOT_KERNEL_BUSY')) + '\n')
                        stream.flush()
                        samples += 1
                        self.done.wait(.2)
                n.nvmlShutdown()
                write(self.folder / f'NVML-summary-{self.attempt}.json', dict(peak_sampled_HBM_bytes=peaks,
                    samples=samples, sampling_period_s=.2, foreign_process_observations=foreign,
                    isolation_passed=not foreign, owner_coordinator_pid=owner,
                    scope='Only live /proc ancestry to this experiment coordinator proves ownership. Nsight may create a new session for its actual worker. All8 GPUs remain covered. No physical idle is inferred.'))
            except BaseException as error:
                write(self.folder / f'NVML-summary-{self.attempt}.json', dict(status='NOT_MEASURED', error=repr(error)))
    watch.GPUWatch = AncestryWatch
