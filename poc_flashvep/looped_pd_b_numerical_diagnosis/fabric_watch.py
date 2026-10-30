"""Low-rate, read-only NVML process samples for quiet-lane validation."""
import json
import threading
import time


class FabricWatch:
    def __init__(self,folder,assigned):
        self.folder=folder;self.assigned=set(assigned)
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)

    def run(self):
        import pynvml as nvml
        nvml.nvmlInit()
        handles=[nvml.nvmlDeviceGetHandleByIndex(i) for i in range(8)]
        try:
            with (self.folder/'fabric_process_samples.jsonl').open('a') as stream:
                while not self.stop_event.is_set():
                    processes=[]
                    for gpu,handle in enumerate(handles):
                        for process in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                            processes.append(dict(physical_GPU=gpu,pid=process.pid,
                                                  outside_assigned_pair=gpu not in self.assigned))
                    stream.write(json.dumps(dict(time=time.monotonic(),processes=processes))+'\n');stream.flush()
                    self.stop_event.wait(1.)
        finally:nvml.nvmlShutdown()

    def start(self):self.thread.start()

    def close(self):
        self.stop_event.set();self.thread.join(timeout=3)
