"""Build the activity collector and retain graph/kernel evidence boundaries."""
import ctypes
import subprocess
import time
from .common import *


def build():
    import shutil
    source=SOURCE/'graph_activity.cc';target=ROOT/'tools';target.mkdir(exist_ok=True)
    library=target/('graph_activity_'+digest(source)[:16]+'.so')
    command=['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',str(source),
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(library)]
    if not library.exists():
        result=subprocess.run(command,capture_output=True,text=True)
        write(target/'graph_build_log.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        assert result.returncode==0,result.stderr
    frozen=target/('graph_activity_'+digest(source)+'.cc')
    if not frozen.exists():shutil.copyfile(source,frozen)
    assert digest(frozen)==digest(source)
    write(target/'graph_activity_build.json',dict(command=command,source_SHA256=digest(source),
        library=str(library),library_SHA256=digest(library)))


def library():
    value=json.loads((ROOT/'tools/graph_activity_build.json').read_text())
    assert digest(value['library'])==value['library_SHA256']
    assert digest(SOURCE/'graph_activity.cc')==value['source_SHA256'],'Rebuild the collector after a source change'
    handle=ctypes.CDLL(value['library']);handle._build_record=value;handle.activity_begin.argtypes=[ctypes.c_int]
    handle.activity_end.argtypes=[ctypes.c_char_p];handle.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    if hasattr(handle,'activity_end_forced'):handle.activity_end_forced.argtypes=[ctypes.c_char_p]
    if hasattr(handle,'activity_end_drain_first'):handle.activity_end_drain_first.argtypes=[ctypes.c_char_p]
    for key in ('activity_dropped','activity_invalid','activity_requests','activity_completions','activity_bytes','activity_parsed'):
        if hasattr(handle,key):getattr(handle,key).restype=ctypes.c_uint64
    return handle


def clock(handle):
    value=ctypes.c_uint64();before=time.monotonic();assert handle.activity_timestamp(ctypes.byref(value))==0
    after=time.monotonic()
    return dict(device_clock_ns=value.value,monotonic_s=(before+after)/2,host_uncertainty_s=after-before)


if __name__=='__main__':initialize();build()
