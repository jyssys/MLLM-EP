"""Use the previously qualified, unchanged CUPTI API28 collector."""
import ctypes
import json
import time
from .common import ROOT,PREVIOUS,digest,write


class Activity:
    def __init__(self,folder,include_copies=False):
        local=ROOT/'references/CUPTI_collector'
        build=json.loads(((local/'activity_build.json') if local.exists() else (PREVIOUS/'tools/activity_build.json')).read_text())
        if local.exists():
            build['source']=str(local/'preallocated_activity.cc')
            build['library']=str(local/'activity_api28_kernel_only.so')
        for key in ('source','library','CUPTI_library'):
            assert digest(__import__('pathlib').Path(build[key]))==build[key+'_SHA256']
        gate=json.loads(((local/'cupti28_witness_gate.json') if local.exists() else (PREVIOUS/'analysis/cupti28_witness_gate.json')).read_text())
        assert gate['passed']
        self.folder=folder;self.lib=ctypes.CDLL(build['library'])
        self.lib.activity_begin.argtypes=[ctypes.c_size_t,ctypes.c_int]
        self.lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        self.lib.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
        for key in ('dropped','invalid','count','all_kernels'):
            getattr(self.lib,'activity_'+key).restype=ctypes.c_uint64
        assert self.lib.activity_begin(1024*1024**2,int(include_copies))==0
        write(folder/'activity_provenance.json',dict(build=build,witness=gate,unchanged_collector=True,
            include_copies=bool(include_copies)))

    def stamp(self):
        value=ctypes.c_uint64();before=time.monotonic_ns()
        assert self.lib.activity_timestamp(ctypes.byref(value))==0
        return dict(cupti_ns=value.value,host_lower_ns=before,host_upper_ns=time.monotonic_ns())

    def close(self):
        assert self.lib.activity_end(str(self.folder/'kernels.csv').encode(),0,2**64-1)==0
        counters={key:int(getattr(self.lib,'activity_'+key)()) for key in ('dropped','invalid','count','all_kernels')}
        write(self.folder/'activity_counters.json',dict(counters=counters,passed=not counters['dropped'] and not counters['invalid']))

    def pause(self):
        assert self.lib.activity_disable()==0
