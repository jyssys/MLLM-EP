"""Compute a domain-separated SHA256 leaf-tree over every tensor byte.

The result is not the SHA256 of the raw tensor byte stream. The schema names
that difference explicitly. CPU leaf comparisons must pass before GPU use.
This read-only kernel is a correctness observer,not a model correction.
"""
import ctypes
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from .common import ROOT,write,digest

ALGORITHM='SHA256_OF_SHA256_4096B_LEAVES_V1'
CHUNK=4096
DOMAIN=b'MLLM_EP_TENSOR_SHA256_LEAVES_V1\0'
_LIB=None


def build():
    source=Path(__file__).with_name('sha256_leaves.cu');sha=digest(source)
    folder=ROOT/'tools'/('gpu_digest_'+sha[:16]);folder.mkdir(parents=True,exist_ok=True)
    output=folder/'digest.so'
    with (folder/'build.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not output.exists():
            command=['/usr/local/cuda/bin/nvcc','-O3','--shared','-Xcompiler','-fPIC',
                '-gencode','arch=compute_90,code=sm_90',str(source),'-o',str(output),
                '-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64']
            result=subprocess.run(command,capture_output=True,text=True)
            write(folder/'build.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                source_SHA256=sha,compiler_version=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True)))
            assert result.returncode==0,result.stderr
            value=json.loads((folder/'build.json').read_text());value['library_SHA256']=digest(output);write(folder/'build.json',value)
        proof=json.loads((folder/'build.json').read_text())
        assert proof['source_SHA256']==sha and digest(output)==proof['library_SHA256']
    return output


def library():
    global _LIB
    if _LIB is None:
        _LIB=ctypes.CDLL(str(build()))
        _LIB.sha256_leaves.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
        _LIB.sha256_leaves.restype=ctypes.c_int
    return _LIB


def root(leaves,byte_count):
    return hashlib.sha256(DOMAIN+struct.pack('<QQ',byte_count,CHUNK)+leaves).hexdigest()


def cpu(raw):
    assert raw
    leaves=b''.join(hashlib.sha256(raw[i:i+CHUNK]).digest() for i in range(0,len(raw),CHUNK))
    return dict(digest_algorithm=ALGORITHM,digest=root(leaves,len(raw)),bytes=len(raw),chunk_bytes=CHUNK)


def gpu(tensor,verify=False):
    import torch
    assert tensor.is_cuda
    data=tensor.detach().contiguous().view(torch.uint8).flatten();n=data.numel();assert n
    output=torch.empty(((n+CHUNK-1)//CHUNK,32),device=data.device,dtype=torch.uint8)
    with torch.cuda.device(data.device):
        code=library().sha256_leaves(data.data_ptr(),n,CHUNK,output.data_ptr(),torch.cuda.current_stream().cuda_stream)
    assert code==0,('SHA256 observer kernel launch failed',code)
    leaves=output.cpu().numpy().tobytes()
    result=dict(digest_algorithm=ALGORITHM,digest=root(leaves,n),bytes=n,chunk_bytes=CHUNK)
    if verify:
        raw=data.cpu().numpy().tobytes();reference=cpu(raw)
        assert result==reference,'GPU digest differs from hashlib over all original bytes'
    return result


def selftest():
    import numpy as np
    import torch
    torch.set_num_threads(1);rows=[];rng=np.random.default_rng(2026112241)
    for n in (1,2,7,55,56,63,64,65,127,128,255,4095,4096,4097,8192,8193,1048576,67108864):
        raw=rng.integers(0,256,n,dtype=np.uint8);data=torch.from_numpy(raw).cuda();value=gpu(data,verify=True)
        before=value['digest'];data[-1]^=1;assert gpu(data)['digest']!=before
        data[-1]^=1;assert gpu(data)['digest']==before
        rows.append(dict(bytes=n,CPU_leaf_tree_match=True,last_byte_change_detected=True,restore_exact=True))
    for dtype in (torch.bfloat16,torch.float32,torch.int64):
        data=torch.arange(65536,dtype=torch.float32,device='cuda').to(dtype).reshape(256,256).t()
        value=gpu(data,verify=True);rows.append(dict(dtype=str(dtype),noncontiguous=True,bytes=value['bytes'],CPU_leaf_tree_match=True))
    value=dict(passed=True,rows=rows,digest_algorithm=ALGORITHM,chunk_bytes=CHUNK,evidence='GPU_DIGEST_CORRECTNESS_ONLY',
        scope='Every tested byte is included. Each leaf matches Python hashlib. This is not raw-byte-stream SHA256 or a model-output proof. '
            'The observer reads model tensors and never changes them.')
    write(ROOT/'analysis/gpu_digest_selftest_v1.json',value);print(value,flush=True)


if __name__=='__main__':
    import sys
    selftest() if len(sys.argv)>1 and sys.argv[1]=='selftest' else print(build())
