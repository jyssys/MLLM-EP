"""Set documented CUPTI buffer preallocation before CUDA context creation."""
import ctypes
import os
from .common import *


def set_size_attribute(setter, attribute, number):
    setter.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    value = ctypes.c_size_t(number)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    status = setter(attribute, ctypes.byref(size), ctypes.byref(value))
    if status:
        raise RuntimeError(f'CUPTI attribute {attribute} failed with code {status}')


def prepare(cfg):
    count = cfg.get('cupti_preallocated_buffers')
    if count is None:
        return
    assert 3 <= count <= 32
    import torch
    assert not torch.cuda.is_initialized(), 'Buffer control must precede CUDA initialization'
    path = Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12')
    assert digest(path) == '2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a'
    handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    setter = handle.cuptiActivitySetAttribute
    getter = handle.cuptiActivityGetAttribute
    getter.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    value = ctypes.c_size_t(count)
    set_size_attribute(setter, 6, count)
    actual = {}
    for key, attribute in (('device_buffer_size_bytes', 0), ('pool_limit', 2), ('preallocated_buffers', 6)):
        size, number = ctypes.c_size_t(ctypes.sizeof(value)), ctypes.c_size_t()
        assert getter(attribute, ctypes.byref(size), ctypes.byref(number)) == 0
        actual[key] = number.value
    assert actual['preallocated_buffers'] == count
    write(ROOT / 'raw' / cfg['name'] / f'cupti_buffer_control_{os.getpid()}.json', dict(
        **actual, library=str(path), library_SHA256=digest(path),
        torch_cuda_initialized_before_setting=False,
        requested_storage_bytes=actual['device_buffer_size_bytes'] * count,
        scope='Profiler allocation setting. Actual HBM change requires memory inventory. No model allocation or arithmetic is modified.'))
