"""Hash unchanged CPU tensor bytes in a bounded correctness-only worker pool."""
import atexit
from concurrent.futures import ThreadPoolExecutor
import hashlib

_POOLS={}


def value(tensor):
    import torch
    assert tensor.device.type=='cpu' and tensor.is_contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def submit(tensor,workers):
    assert 1<=workers<=8
    if workers not in _POOLS:
        _POOLS[workers]=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='correctness_sha256')
    return _POOLS[workers].submit(value,tensor)


def close():
    for pool in _POOLS.values():pool.shutdown(wait=True)


atexit.register(close)
