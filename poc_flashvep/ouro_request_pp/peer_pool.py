"""Common preallocated-receive transport control, not a placement policy."""
import torch


class PeerPool:
    def __init__(self,rank):
        self.device=torch.device('cuda',rank)
        self.buffers={}

    def prepare(self,requests):
        self.buffers.clear()
        for r in requests:
            for shape,dtype in [((1,len(r['input_ids']),2048),torch.bfloat16),
                                ((1,1,2048),torch.bfloat16),((1,1),torch.long)]:
                self.buffers[(r['id'],shape,dtype)]=torch.empty(shape,dtype=dtype,device=self.device)

    def transfer(self,rid,payload):
        out=self.buffers[(rid,tuple(payload.shape),payload.dtype)]
        out.copy_(payload,non_blocking=True)
        return out
