"""Enable both CUDA API domains without changing the collector library."""
import ctypes
import json
from .common import ROOT
from .correlated_activity import Activity as DriverActivity


class Activity(DriverActivity):
    def __init__(self,folder,include_copies=False,witness=False):
        self.include_copies=include_copies
        if not witness:
            gate=json.loads((ROOT/'analysis/runtime_api_correlation_witness.json').read_text())
            assert gate['eager_correlation_passed'] and gate['runtime_correlation_passed']
        build=json.loads((ROOT/'references/correlated_collector/build.json').read_text())
        library=ctypes.CDLL(build['library'])
        library.activity_graph_mode.argtypes=[ctypes.c_int]
        assert library.activity_graph_mode(1)==0
        # The existing graph mode enables both RUNTIME and DRIVER activities.
        # Its graph records do not qualify graph-kernel attribution.
        super().__init__(folder,include_copies=include_copies,witness=True)

    def resume(self):
        assert not self.enabled
        assert self.lib.activity_enable(int(self.include_copies))==0
        self.enabled=True
