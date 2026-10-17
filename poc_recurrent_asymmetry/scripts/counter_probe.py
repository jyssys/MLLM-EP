"""Bounded hardware-counter access probe with one pretrained MLP."""
import os
import torch
from poc_recurrent_asymmetry.contract import *
from poc_async_recurrent_pp.scripts.baseline import load


@torch.inference_mode()
def main():
    model,tokenizer=load()
    x=torch.ones((128,2048),device='cuda',dtype=torch.bfloat16)
    for _ in range(3):model.model.layers[0].mlp(x)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    result=model.model.layers[0].mlp(x)
    torch.cuda.synchronize();torch.cuda.profiler.stop()
    assert bool(torch.isfinite(result).all())
    write(Path(os.environ['LOOPED_KV_OUTPUT'])/'status.json',dict(success=True,
        scope='Hardware-counter probe. Does not represent whole-request HBM traffic or E2E latency.'))


if __name__=='__main__':main()
