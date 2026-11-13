"""Record memory, clocks, and ownership without inferring physical idle."""
import os,threading,time
from .common import *


class GPUWatch:
    def __init__(self,folder,gpus,attempt):
        self.folder=folder;self.gpus=gpus;self.attempt=attempt;self.done=threading.Event();self.thread=None
    def start(self):
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()
    def sample(self):
        import pynvml as n
        peaks={g:0 for g in self.gpus};samples=0;foreign=[]
        try:
            n.nvmlInit();all_handles={g:n.nvmlDeviceGetHandleByIndex(g) for g in range(8)}
            handles={g:all_handles[g] for g in self.gpus};ownership={}
            with (self.folder/f'NVML-{self.attempt}.jsonl').open('a') as log:
                while not self.done.is_set():
                    start=time.perf_counter()
                    if samples%5==0:
                        for gpu,h in all_handles.items():
                            ownership[gpu]=[]
                            for p in n.nvmlDeviceGetComputeRunningProcesses(h):
                                try:sid=os.getsid(p.pid)
                                except ProcessLookupError:continue
                                row=dict(pid=p.pid,session=sid,owned_session=sid==os.getsid(0))
                                ownership[gpu].append(row)
                                if not row['owned_session']:foreign.append(dict(time=time.monotonic(),GPU=gpu,**row))
                    for gpu,h in handles.items():
                        memory=n.nvmlDeviceGetMemoryInfo(h);peaks[gpu]=max(peaks[gpu],memory.used)
                        row=dict(time=time.monotonic(),physical_GPU=gpu,uuid=n.nvmlDeviceGetUUID(h),
                            used_HBM_bytes=memory.used,sampled_utilization_percent=n.nvmlDeviceGetUtilizationRates(h).gpu,
                            SM_clock_MHz=n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_SM),
                            memory_clock_MHz=n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_MEM),
                            clocks_event_reasons_mask=n.nvmlDeviceGetCurrentClocksEventReasons(h),
                            temperature_C=n.nvmlDeviceGetTemperature(h,n.NVML_TEMPERATURE_GPU),
                            power_W=n.nvmlDeviceGetPowerUsage(h)/1000,processes=ownership[gpu],
                            sampling_cpu_ms=1000*(time.perf_counter()-start),evidence='MEASURED_NVML_NOT_KERNEL_BUSY')
                        log.write(json.dumps(row)+'\n')
                    log.flush();samples+=1;self.done.wait(.2)
            n.nvmlShutdown()
            write(self.folder/f'NVML-summary-{self.attempt}.json',dict(peak_sampled_HBM_bytes=peaks,
                samples=samples,sampling_period_s=.2,foreign_process_observations=foreign,
                isolation_passed=not foreign,scope='NVML utilization is not physical kernel activity or a Decode-safe budget. Ownership polls cover all8 GPUs once each second. Telemetry covers the experiment GPUs.'))
        except BaseException as error:
            write(self.folder/f'NVML-summary-{self.attempt}.json',dict(status='NOT_MEASURED',error=repr(error)))
    def stop(self):
        self.done.set()
        if self.thread:self.thread.join(10)
