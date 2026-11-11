import ctypes,subprocess
from .common import *


def build():
    initialize();source=SOURCE/'lt_bridge.cc';target=ROOT/'tools'/('lt_'+digest(source)[:16]+'.so')
    command=['g++','-O3','-std=c++17','-fPIC','-shared',str(source),
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcublasLt','-o',str(target)]
    if not target.exists():subprocess.run(command,check=True)
    record=dict(command=command,source_SHA256=digest(source),library=str(target),library_SHA256=digest(target))
    write(ROOT/'tools/lt_build.json',record)
    write(ROOT/'tools'/('lt_build_'+digest(source)[:16]+'.json'),record)
    return record


def library():
    record=json.loads((ROOT/'tools/lt_build.json').read_text())
    assert digest(record['library'])==record['library_SHA256']
    lib=ctypes.CDLL(record['library'])
    lib.lt_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_size_t,ctypes.c_int]
    lib.lt_create.restype=ctypes.c_void_p
    lib.lt_count.argtypes=[ctypes.c_void_p]
    lib.lt_origin.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.lt_attribute.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    lib.lt_workspace.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.lt_workspace.restype=ctypes.c_size_t
    lib.lt_run.argtypes=[ctypes.c_void_p,ctypes.c_int,*([ctypes.c_void_p]*4),ctypes.c_size_t,ctypes.c_void_p]
    lib.lt_destroy.argtypes=[ctypes.c_void_p];lib.lt_version.restype=ctypes.c_size_t
    return lib


class Plan:
    def __init__(self,m,n,k,workspace=64*1024**2,maximum=32,workspace_buffer=None):
        import torch
        self.lib=library();self.bytes=workspace;self.geometry=(m,n,k)
        self.pointer=self.lib.lt_create(m,n,k,workspace,maximum)
        assert self.pointer,('cuBLASLt plan failed',self.lib.lt_last_error())
        self.workspace=workspace_buffer if workspace_buffer is not None else torch.empty(workspace,device='cuda',dtype=torch.uint8)
        assert self.workspace.numel()>=workspace

    def metadata(self,index):
        return dict(index=index,origin=self.lib.lt_origin(self.pointer,index),**{name:self.lib.lt_attribute(self.pointer,index,key) for name,key in
            (('id',0),('tile',1),('splitK',2),('reduction',3),('swizzle',4),('stages',6))},
            workspace_bytes=self.lib.lt_workspace(self.pointer,index))

    def run(self,index,x,w,y):
        import torch
        assert x.is_contiguous() and w.is_contiguous() and y.is_contiguous()
        assert (len(x),len(w),x.shape[1])==self.geometry
        code=self.lib.lt_run(self.pointer,index,x.data_ptr(),w.data_ptr(),y.data_ptr(),
            self.workspace.data_ptr(),self.bytes,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(f'cuBLASLt execution status {code}')
        return y

    def __len__(self):return self.lib.lt_count(self.pointer)
    def close(self):
        if self.pointer:self.lib.lt_destroy(self.pointer);self.pointer=None


if __name__=='__main__':print(json.dumps(build()))
