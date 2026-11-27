"""Compare dynamic long-context FA3 graph replay with eager attention after dummy warmup."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .operator_bench import compare
from .paged_graph import PagedGraphPool
from .peer_attention import AssistedAttention, KVShard, PeerAccess


def make_arena(device, blocks,layers):
    memory = torch.randn((layers,2,blocks,128,4,128),device=device,dtype=torch.bfloat16)
    return SimpleNamespace(device=device,cache=memory)


def page_args(lengths, device, table_start):
    counts = [(n+127)//128 for n in lengths]
    table = torch.zeros((len(lengths),max(counts)),device=device,dtype=torch.int32)
    offset = table_start
    for row,width in enumerate(counts):
        table[row,:width] = torch.arange(offset,offset+width,device=device,dtype=torch.int32)
        offset += width
    return table,torch.tensor(lengths,device=device,dtype=torch.int32)


def test_pool(pool, peer, batch, count, d, e, queued):
    result = []
    runtime = AssistedAttention((batch,28,128),decode_device=d,helper_device=e,peer=peer)
    slots = []
    for context in (180,4096,16400,30537,16391,1033):
        for split in (False,True):
            lengths = [max(1,context-min(113,context//(3*batch))*rank) for rank in range(batch)]
            e_lens = [n//2 for n in lengths] if split else [0]*batch
            d_lens = [n-offloaded for n,offloaded in zip(lengths,e_lens,strict=True)]
            offset = 7 if context in (16400,16391,1033) else 0
            d_table,d_used = page_args(d_lens,d,offset)
            e_table,e_used = page_args(e_lens,e,offset)
            layout = SimpleNamespace(d_lengths=d_lens,h_lengths=e_lens,
                                     d_table=d_table,d_used=d_used,h_table=e_table,h_used=e_used)
            slot = pool.prepare(layout)
            if slot is None:
                raise RuntimeError(f"Unexpected native-shape fallback {pool.last_fallback}")
            queries = [torch.randn((batch,28,128),device=d,dtype=torch.bfloat16)
                       for _ in range(pool.d.cache.shape[0])]
            candidates = [slot.attend(layer,query).clone() for layer,query in enumerate(queries)] if queued else []
            for layer,query in enumerate(queries):
                candidate = candidates[layer] if queued else slot.attend(layer,query).clone()
                dc,ec = pool.d.cache[layer],pool.helper.cache[layer]
                local = KVShard.paged(dc[0],dc[1],d_lens,d_table,d_used)
                remote = KVShard.paged(ec[0],ec[1],e_lens,e_table,e_used)
                reference = runtime.split(query,local,remote)
                result.append({
                    "context":context,"mode":"split" if split else "local",
                    "layer":layer,"compare":compare(reference,candidate),
                })
            slots.append(id(slot))
    if len(set(slots))!=2:
        raise RuntimeError("Dynamic long witness unexpectedly created additional graph slots")
    return {"batch":batch,"graph_num_splits":count,"queued":queued,"tests":result,
            "all_operator_pass":all(x["compare"]["relative_l2"] <= .008 for x in result),
            "max_relative_l2":max(x["compare"]["relative_l2"] for x in result),
            "pool":pool.receipt()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--decode-device",type=int,default=1)
    parser.add_argument("--helper-device",type=int,default=0)
    parser.add_argument("--splits",type=int,nargs="+",default=[0,1,4,8])
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--helper-context",type=int,default=16384)
    parser.add_argument("--queued",action="store_true")
    args = parser.parse_args()
    d,e = torch.device("cuda",args.decode_device),torch.device("cuda",args.helper_device)
    with torch.cuda.device(d):
        peer = PeerAccess(d.index,e.index)
        dd,hh = make_arena(d,1040,args.layers),make_arena(e,560,args.layers)
        with args.out.open("x") as stream:
            for n in args.splits:
                for batch in (1,4):
                    pool=PagedGraphPool(dd,hh,heads=28,dim=128,max_batch=4,
                                        fixed_d_context=32768,fixed_helper_context=args.helper_context,num_splits=n)
                    pool.prewarm((batch,),include_local=True,include_split=True)
                    row=test_pool(pool,peer,batch,n,d,e,args.queued)
                    pool.close()
                    stream.write(json.dumps(row)+"\n")
                    stream.flush()
                    print(n,batch,row["max_relative_l2"],row["all_operator_pass"],flush=True)
            stream.write('{"type":"complete"}\n')


if __name__=="__main__":
    main()
