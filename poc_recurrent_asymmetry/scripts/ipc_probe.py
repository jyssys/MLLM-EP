"""Minimal CUDA IPC compatibility diagnostic, not a model benchmark."""
import ctypes
import json
from pathlib import Path
import torch


def main():
    tensor=torch.empty(16*1024**2,device='cuda',dtype=torch.uint8)
    libraries=[line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line]
    assert libraries
    runtime=ctypes.CDLL(libraries[0])
    runtime.cudaIpcGetMemHandle.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    runtime.cudaIpcGetMemHandle.restype=ctypes.c_int
    handle=ctypes.create_string_buffer(64)
    code=runtime.cudaIpcGetMemHandle(ctypes.byref(handle),ctypes.c_void_p(tensor.data_ptr()))
    runtime.cudaGetLastError()
    print(json.dumps(dict(cuda_IPC_return_code=code,success=code==0,payload_bytes=tensor.numel(),
        runtime=libraries[0],scope='Allocator interoperability probe only. No inference speed or quality result.')))


if __name__=='__main__':main()
