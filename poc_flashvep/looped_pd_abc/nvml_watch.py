"""Sample physical GPU memory. NVML utilization is not exact kernel busy time."""
import json
import os
import threading
import time
from .common import UUIDS,write

class GPUWatch:
 def __init__(self,folder,gpus,attempt):
  self.folder=folder;self.gpus=gpus;self.attempt=attempt;self.done=threading.Event();self.thread=None
 def start(self):
  self.thread=threading.Thread(target=self.sample,name='abc-nvml',daemon=True);self.thread.start()
 def sample(self):
  peaks={g:0 for g in self.gpus};measured={g:0 for g in self.gpus};samples=0
  try:
   import pynvml as n
   n.nvmlInit();handles={g:n.nvmlDeviceGetHandleByUUID(UUIDS[g]) for g in self.gpus}
   with (self.folder/f'NVML-{self.attempt}.jsonl').open('a') as log:
    while not self.done.is_set():
     start=time.perf_counter();records=[];active=os.environ.get('ABC_MEASURED_ACTIVE')=='1'
     for gpu,handle in handles.items():
      memory=n.nvmlDeviceGetMemoryInfo(handle);util=n.nvmlDeviceGetUtilizationRates(handle)
      peaks[gpu]=max(peaks[gpu],memory.used)
      if active:measured[gpu]=max(measured[gpu],memory.used)
      records.append(dict(time=time.monotonic(),physical_GPU=gpu,uuid=UUIDS[gpu],
       used_HBM_bytes=memory.used,sampled_utilization_percent=util.gpu,measured_cohort=active,
       evidence='MEASURED_NVML_SAMPLE_NOT_KERNEL_BUSY_TIME'))
     elapsed=1000*(time.perf_counter()-start)
     for row in records:row['sampling_cpu_ms']=elapsed;log.write(json.dumps(row)+'\n')
     log.flush();samples+=1;self.done.wait(.2)
   n.nvmlShutdown()
   write(self.folder/f'NVML-summary-{self.attempt}.json',dict(peak_sampled_HBM_bytes=peaks,
    measured_cohort_peak_sampled_HBM_bytes=measured,sampling_period_s=.2,samples=samples,
    scope='Sampled physical-device memory, not a continuous exact peak. Utilization uses the NVML averaging window.'))
  except BaseException as error:
   write(self.folder/f'NVML-summary-{self.attempt}.json',dict(status='NOT_MEASURED',error=repr(error)))
 def stop(self):
  self.done.set()
  if self.thread:self.thread.join(10)
