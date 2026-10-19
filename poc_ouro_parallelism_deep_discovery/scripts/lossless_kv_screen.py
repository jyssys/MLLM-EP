"""CPU coded-size and explicitly modeled reconstruction-read screen."""
import argparse
import datetime as dt
import numpy as np
import zlib
from poc_ouro_parallelism_deep_discovery.contract import *

def frame(words):
    words=np.ascontiguousarray(words,dtype='<u2')
    plane=words.view(np.uint8).reshape(-1,2).T.copy().tobytes()
    encoded=zlib.compress(plane,level=1)
    restored=np.frombuffer(zlib.decompress(encoded),dtype=np.uint8).reshape(2,-1).T.copy()
    restored=restored.reshape(-1).view('<u2').reshape(words.shape)
    assert np.array_equal(restored,words)
    return len(encoded)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    rows=[];provenance=[]
    for size,short in [('1.4B','14'),('2.6B','26')]:
        run=ROOT/f'outputs/key_rank_base{short}_P1_v1'
        status=read(run/'status.json');assert status['success'] and read(run/'guard_status.json')['success']
        path=run/'actual_key_words.npz';assert sha(path)==status['captures'][0]['sha256']
        arrays=np.load(path,allow_pickle=False);words=arrays['K_words'];meta=status['captures'][0]['layers']
        for layer in sorted({r['physical_layer'] for r in meta}):
            indices=[next(i for i,r in enumerate(meta) if r['physical_layer']==layer and r['depth']==d) for d in [1,2,3,4]]
            for head in range(3):
                states=[words[i,head] for i in indices]
                amount=states[0].nbytes
                independent=[frame(x) for x in states]
                adjacent=[frame(np.bitwise_xor(states[i],states[i-1])) for i in [1,2,3]]
                base=[frame(np.bitwise_xor(states[i],states[0])) for i in [1,2,3]]
                for i in [1,2,3]:
                    assert np.array_equal(np.bitwise_xor(np.bitwise_xor(states[i],states[i-1]),states[i-1]),states[i])
                native=4*amount
                chain_capacity=independent[0]+sum(adjacent)
                base_capacity=independent[0]+sum(base)
                # These expressions count requested encoded bytes only. They
                # exclude decoder work, extra buffers, headers beyond zlib,
                # transaction granularity, and any partial cache residency.
                chain_uncached=sum(independent[0]+sum(adjacent[:r]) for r in range(4))
                base_uncached=4*independent[0]+sum(base)
                rows.append(dict(size=size,physical_layer=layer,head=int(arrays['heads'][head]),
                    rows=states[0].shape[0],dimensions=128,raw_bytes_per_depth=amount,
                    independent_frame_bytes=independent,adjacent_XOR_frame_bytes=adjacent,base_XOR_frame_bytes=base,
                    independent_capacity_ratio=sum(independent)/native,
                    adjacent_chain_capacity_ratio=chain_capacity/native,
                    common_base_capacity_ratio=base_capacity/native,
                    uncached_chain_reconstruction_read_ratio=chain_uncached/native,
                    uncached_common_base_reconstruction_read_ratio=base_uncached/native,
                    optimistic_hot_base_chain_read_ratio=chain_capacity/native,
                    exact_CPU_reconstruction=True))
        provenance.append(dict(size=size,path=str(path.relative_to(ROOT)),sha256=sha(path),
            model=status['model'],revision=status['revision'],selected_keys_only=True))
    summary=[]
    fields=['independent_capacity_ratio','adjacent_chain_capacity_ratio','common_base_capacity_ratio',
            'uncached_chain_reconstruction_read_ratio','uncached_common_base_reconstruction_read_ratio']
    for size in ['1.4B','2.6B']:
        selected=[r for r in rows if r['size']==size]
        summary.append(dict(size=size,observed_layer_head_groups=len(selected),
            **{field:float(np.mean([r[field] for r in selected])) for field in fields}))
    write(args.output,dict(timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        records=rows,summaries=summary,provenance=provenance,codec='CPU zlib level1 over reversible BF16 byte planes',
        evidence_class='ANALYTIC_BOUND',measured='Encoded frame lengths and exact CPU word reconstruction for selected actual keys.',
        modeled='Uncached encoded-byte demand under two stated reconstruction layouts. Not hardware traffic.',
        hot_base='An optimistic residency envelope only. It requires a separately proved or measured base-retention mechanism.',
        limitations=['One controlled request per model','Keys only, no value compression result',
            '257 selected positions, not a full long-context distribution','No GPU encoder, decoder, or compressed attention kernel',
            'No physical link or HBM byte saving','No E2E or capacity-admission measurement',
            'No free cross-depth query readiness','No speedup inferred from encoded size']))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
