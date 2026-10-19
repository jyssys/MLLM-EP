"""Exact finite-field rank certificates for selected actual BF16 keys."""
import argparse
import datetime as dt
from fractions import Fraction
import struct
import numpy as np
from poc_ouro_parallelism_deep_discovery.contract import *
PRIME=65521

def residue(word):
    sign=-1 if word&0x8000 else 1
    exponent=(word>>7)&255;mantissa=word&127
    assert exponent!=255
    coefficient=mantissa if exponent==0 else mantissa+128
    power=-133 if exponent==0 else exponent-134
    return sign*coefficient*pow(2,power,PRIME)%PRIME

def rank_and_rows(matrix):
    a=np.array(matrix,dtype=np.int64,copy=True)%PRIME
    identifiers=list(range(len(a)));rank=0
    for column in range(a.shape[1]):
        possible=np.flatnonzero(a[rank:,column])
        if not len(possible):continue
        row=rank+int(possible[0])
        a[[rank,row]]=a[[row,rank]]
        identifiers[rank],identifiers[row]=identifiers[row],identifiers[rank]
        inverse=pow(int(a[rank,column]),-1,PRIME)
        a[rank]=a[rank]*inverse%PRIME
        factors=a[rank+1:,column].copy()
        a[rank+1:]=(a[rank+1:]+PRIME-(factors[:,None]*a[rank][None,:])%PRIME)%PRIME
        rank+=1
        if rank==min(a.shape):break
    return rank,identifiers[:rank]

def determinant(matrix):
    a=np.array(matrix,dtype=np.int64,copy=True)%PRIME
    assert a.shape[0]==a.shape[1]
    product=1
    for i in range(len(a)):
        available=np.flatnonzero(a[i:,i])
        if not len(available):return 0
        row=i+int(available[0])
        if row!=i:a[[i,row]]=a[[row,i]];product=-product
        pivot=int(a[i,i]);product=product*pivot%PRIME
        factors=a[i+1:,i]*pow(pivot,-1,PRIME)%PRIME
        a[i+1:]=(a[i+1:]+PRIME-(factors[:,None]*a[i][None,:])%PRIME)%PRIME
    return product%PRIME

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    source=read(args.run/'status.json');assert source['success']
    assert read(args.run/'guard_status.json')['success']
    path=args.run/'actual_key_words.npz';capture=source['captures'][0]
    assert sha(path)==capture['sha256']
    # Independently check each exponent class and representative mantissas.
    fixtures=[0,0x8000]
    fixtures += [(sign<<15)|(exp<<7)|mant for sign in [0,1] for exp in range(255) for mant in [0,1,63,127]]
    for word in fixtures:
        value=struct.unpack('>f',struct.pack('>I',word<<16))[0]
        rational=Fraction.from_float(value)
        assert residue(word)==rational.numerator*pow(rational.denominator,-1,PRIME)%PRIME
    lookup=np.zeros(65536,dtype=np.int64)
    for word in range(65536):
        if (word>>7)&255!=255:lookup[word]=residue(word)
    arrays=np.load(path,allow_pickle=False);words=arrays['K_words']
    assert not np.any(((words>>7)&255)==255)
    records=[]
    for layer in range(len(words)):
        for head in range(words.shape[1]):
            keys=lookup[words[layer,head]]
            differences=(keys[1:]+PRIME-keys[:1])%PRIME
            rank,rows=rank_and_rows(differences)
            minor=None
            if rank==128:
                minor=determinant(differences[rows]);assert minor!=0
            records.append(dict(**capture['layers'][layer],head=int(arrays['heads'][head]),
                n_keys=len(keys),features=128,field_prime=PRIME,rank_over_field=rank,
                selected_difference_rows=[i+1 for i in rows],minor_determinant_mod_prime=minor,
                exact_rational_full_column_rank_certified=rank==128))
    write(args.output,dict(timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        model=source['model'],revision=source['revision'],source_capture=str(path),source_sha256=sha(path),
        evidence_class='ANALYTIC_BOUND',empirical_input='Actual native post-RoPE BF16 keys from one frozen controlled request.',
        arithmetic='Map exact dyadic BF16 values into an odd-prime field. A nonzero minor proves full rank over the rationals.',
        residue_validation_cases=len(fixtures),records=records,
        full_rank_heads=sum(r['exact_rational_full_column_rank_certified'] for r in records),total_heads=len(records),
        conclusion='For a certified head, no nonzero real query change preserves all exact real-arithmetic key-score differences.',
        limitations=['Not a GPU rounded-score equality theorem','Not a complete attention-output reuse impossibility result',
            'No query-change distribution or population-level hit rate','No task accuracy or runtime speedup',
            'Only three physical layers and three heads per recurrent depth from one controlled request']))
    print('KEY_RANK_CERTIFICATES',source['size'],sum(r['exact_rational_full_column_rank_certified'] for r in records),len(records))

if __name__=='__main__':main()
