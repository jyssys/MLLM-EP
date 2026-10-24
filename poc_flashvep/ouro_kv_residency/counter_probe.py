"""One original BF16 FA4 operation for hardware counters, if permission allows."""
import argparse
import os
import torch
from .common import FIXTURES,UUIDS
from .distributed import paged,attention

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--P',type=int,required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[0].removeprefix('GPU-')
    packet=torch.load(FIXTURES/f'P{a.P}_d1_l12.pt',weights_only=True)
    q=packet['q'].to('cuda');cache=paged(packet['k'],packet['v'],1)
    for _ in range(3):attention(q,cache,return_lse=False)
    pressure=torch.zeros(32*1024*1024,device='cuda');pressure.add_(1);torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    attention(q,cache,return_lse=False)
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()

if __name__=='__main__':main()
