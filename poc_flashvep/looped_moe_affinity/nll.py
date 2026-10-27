"""Pure LM NLL. Reuse the official packed-data reader, not the auxiliary loss."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from .common import DATA,append,write,digest

def evaluate(a,ep,target,config):
    sys.path.insert(0,'/home/esjung/external/how-to-loop-moe-poc')
    from src.data.packed_dataset import PackedSeqDataset
    directory=Path(config.get('nll_data',DATA/'foil-heldout/fineweb_edu_heldout_v1'))
    data=PackedSeqDataset(str(directory),seq_len=4096,seed=0)
    assert data.is_complete
    # The official reader exposes deterministic canonical packed windows.
    rank=ep.rank if ep else 0
    output=target/f'nll_sequences.rank{rank}.jsonl'
    total_loss=0.;total_tokens=0
    for i in range(min(data.num_sequences,config.get('nll_sequences',data.num_sequences))):
        tokens=data.get_sequence(i)
        if isinstance(tokens,dict):tokens=tokens['input_ids']
        ids=torch.as_tensor(tokens,dtype=torch.long,device='cuda').reshape(1,-1)
        logits=a.forward(ids)
        # Sum cross entropy in FP32. Only the NLL reduction changes precision.
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='sum')
        value=float(loss);n=ids.shape[-1]-1
        total_loss+=value;total_tokens+=n
        append(output,dict(sequence=i,tokens=n,nll_sum=value,nll=value/n))
        del logits,loss
        if i%25==0:print('NLL',i,data.num_sequences,total_loss/total_tokens,flush=True)
    write(target/f'nll_summary.rank{rank}.json',dict(model=a.name,ep=ep.world if ep else 1,
        sequences=i+1,tokens=total_tokens,nll=total_loss/total_tokens,full_sequences=data.num_sequences,
        data_manifest=str(directory/'manifest.json'),data_sha256=digest(directory/'manifest.json'),
        scoring='Pure shifted-token cross entropy. No router auxiliary loss. Packed windows do not score boundary transitions.',
        evidence='LIVE_FULL_NATIVE_MODEL_NLL'))
