"""Associate GPU records with engine steps. Require a separate witness gate."""
import ctypes
import json
import time
from pathlib import Path
from .common import ROOT,write,digest

CURRENT=None


class Activity:
    def __init__(self,folder,include_copies=False,witness=False):
        global CURRENT
        build=json.loads((ROOT/'references/correlated_collector/build.json').read_text())
        for key in ('source','library','CUPTI_library'):
            assert digest(Path(build[key]))==build[key+'_SHA256']
        if not witness:
            assert json.loads((ROOT/'analysis/correlated_collector_witness.json').read_text())['eager_correlation_passed']
        self.folder=folder;self.lib=ctypes.CDLL(build['library']);self.enabled=True
        self.lib.activity_begin.argtypes=[ctypes.c_size_t,ctypes.c_int]
        self.lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        self.lib.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
        self.lib.activity_push_id.argtypes=[ctypes.c_uint64]
        self.lib.activity_pop_id.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        for key in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes'):
            getattr(self.lib,'activity_'+key).restype=ctypes.c_uint64
        assert self.lib.activity_begin(1024*1024**2,int(include_copies))==0
        write(folder/'activity_provenance.json',dict(build=build,external_correlation=True,
            include_copies=include_copies,scope='Witness and clean/profile/clean overhead gates remain mandatory.'))
        CURRENT=self

    def push(self,value):
        assert self.lib.activity_push_id(value)==0

    def pop(self,expected):
        value=ctypes.c_uint64()
        assert self.lib.activity_pop_id(ctypes.byref(value))==0 and value.value==expected

    def stamp(self):
        value=ctypes.c_uint64();before=time.monotonic_ns()
        assert self.lib.activity_timestamp(ctypes.byref(value))==0
        return dict(cupti_ns=value.value,host_lower_ns=before,host_upper_ns=time.monotonic_ns())

    def pause(self):
        assert self.lib.activity_disable()==0
        self.enabled=False

    def close(self):
        global CURRENT
        code=int(self.lib.activity_end(str(self.folder/'kernels.csv').encode(),0,2**64-1))
        counters={key:int(getattr(self.lib,'activity_'+key)()) for key in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes')}
        write(self.folder/'activity_close_result.json',dict(returncode=code,counters=counters,
            scope='A nonzero close result fails the trace. Do not infer the missing error code of an earlier failed process from these later diagnostics.'))
        write(self.folder/'activity_counters.json',dict(counters=counters,
            passed=code==0 and not counters['dropped'] and not counters['invalid']))
        assert code==0,('CUPTI activity_end failed',code,counters)
        self.enabled=False;CURRENT=None
