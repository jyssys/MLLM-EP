"""Bounded exact-real attention-pattern test on selected observed BF16 keys."""
import argparse
import datetime as dt
import numpy as np
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.scripts.key_difference_rank import PRIME,residue,rank_and_rows,determinant


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    start=dt.datetime.now(dt.timezone.utc).isoformat()
    source=read(args.run/'status.json');assert source['success']
    assert read(args.run/'guard_status.json')['success']
    capture=source['captures'][0];path=args.run/'actual_key_words.npz'
    assert sha(path)==capture['sha256']
    arrays=np.load(path,allow_pickle=False);words=arrays['K_words']
    assert words.shape[2:]==(257,128)
    assert not np.any(((words>>7)&255)==255)
    lookup=np.zeros(65536,dtype=np.int64)
    for word in range(65536):
        if (word>>7)&255!=255:lookup[word]=residue(word)
    grouped={}
    for index,metadata in enumerate(capture['layers']):
        physical=metadata['physical_layer']
        grouped.setdefault(physical,[]).append((metadata,index))
    records=[]
    for layer,members in sorted(grouped.items()):
        members=sorted(members,key=lambda row:row[0]['depth'])
        assert len(members)==4
        for a,b in [(0,1),(1,2),(2,3),(0,3)]:
            for h,head in enumerate(arrays['heads']):
                left=lookup[words[members[a][1],h]];right=lookup[words[members[b][1],h]]
                Da=(left[1:]+PRIME-left[:1])%PRIME
                Db=(right[1:]+PRIME-right[:1])%PRIME
                combined=np.concatenate([Da,(-Db)%PRIME],axis=1)
                rank,rows=rank_and_rows(combined)
                certificate=determinant(combined) if rank==256 else None
                if rank==256:assert certificate!=0
                records.append(dict(physical_layer=layer,head=int(head),left_depth=members[a][0]['depth'],
                                    right_depth=members[b][0]['depth'],matrix_shape=list(combined.shape),
                                    field_prime=PRIME,rank_over_field=rank,determinant_mod_prime=certificate,
                                    full_rank_certificate=rank==256))
    write(args.output,dict(start_utc=start,end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
          model=source['model'],revision=source['revision'],input_path=str(path),input_sha256=sha(path),
          evidence_class='ANALYTIC_BOUND',records=records,full_rank_pairs=sum(r['full_rank_certificate'] for r in records),
          total_pairs=len(records),
          equation='[K_a(j)-K_a(0), -(K_b(j)-K_b(0))] [q_a; q_b] = 0 for every selected j.',
          conclusion='For a full-rank pair, identical exact-real softmax distributions on these selected positions require both query vectors to be zero.',
          limitations=['Selected observed keys from one controlled request per model.',
                       'Not a theorem about GPU-rounded scores or BF16 softmax equality.',
                       'Not a theorem about equal attention outputs with different patterns or values.',
                       'No measured actual-query hit rate, quality change, HBM saving, or latency benefit.',
                       'No rejection of approximate pattern reuse.']))
    print('CROSS_DEPTH_RANK',source['size'],sum(r['full_rank_certificate'] for r in records),len(records))


if __name__=='__main__':main()
