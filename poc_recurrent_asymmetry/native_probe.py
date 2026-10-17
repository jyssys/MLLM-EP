"""Existing native worker diagnostics with dynamic 24-layer ownership."""
from poc_looped_multigpu_kv_placement.worker_probe import KVWorkerProbe
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe


class NativeProbe(KVWorkerProbe):
    teacher=FourModelProbe.teacher
    start_parity_capture=FourModelProbe.start_parity_capture
    finish_parity_capture=FourModelProbe.finish_parity_capture
    runtime_counts_start=FourModelProbe.runtime_counts_start
    runtime_counts_finish=FourModelProbe.runtime_counts_finish
