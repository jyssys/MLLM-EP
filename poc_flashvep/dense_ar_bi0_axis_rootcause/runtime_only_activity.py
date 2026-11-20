"""Record CUDA APIs and kernels without graph-enclosure activity records."""
import ctypes
import json
from .common import ROOT
from .correlated_activity import Activity as BaseActivity


class Activity(BaseActivity):
    def __init__(self,folder,include_copies=False,witness=False):
        if not witness:
            gate=json.loads((ROOT/'analysis/runtime_only_correlation_witness.json').read_text())
            assert gate['eager_correlation_passed'] and gate['runtime_correlation_passed']
        build=json.loads((ROOT/'references/correlated_collector/build.json').read_text())
        library=ctypes.CDLL(build['library']);library.activity_graph_mode.argtypes=[ctypes.c_int]
        assert library.activity_graph_mode(0)==0
        super().__init__(folder,include_copies=include_copies,witness=True)
        self.sdk=ctypes.CDLL(build['CUPTI_library'])
        self.sdk.cuptiActivityEnable.argtypes=[ctypes.c_int];self.sdk.cuptiActivityDisable.argtypes=[ctypes.c_int]
        # CUPTI12.9 cupti_activity.h defines RUNTIME=5 and DRIVER=4.
        self.runtime_kind=5
        assert self.sdk.cuptiActivityEnable(self.runtime_kind)==0

    def pause(self):
        assert self.sdk.cuptiActivityDisable(self.runtime_kind)==0
        super().pause()

    def close(self):
        if self.enabled:assert self.sdk.cuptiActivityDisable(self.runtime_kind)==0
        super().close()
