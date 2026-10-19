"""Exact rational counterexamples for a fixed-size linear attention summary."""
from fractions import Fraction
import argparse
import datetime as dt
from poc_ouro_parallelism_deep_discovery.contract import *

def eliminate(matrix):
    a=[[Fraction(v) for v in row] for row in matrix];N=len(a);rank=0;det=Fraction(1)
    for column in range(N):
        pivot=next((r for r in range(rank,N) if a[r][column]),None)
        if pivot is None:continue
        if pivot!=rank:a[pivot],a[rank]=a[rank],a[pivot];det=-det
        value=a[rank][column];det*=value
        a[rank]=[x/value for x in a[rank]]
        for r in range(rank+1,N):
            factor=a[r][column]
            a[r]=[x-factor*y for x,y in zip(a[r],a[rank])]
        rank+=1
    return rank,det if rank==N else Fraction(0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=[]
    for N in range(2,9):
        # Real queries q_i=i*log(2), real keys k_j=j give exp(q_i*k_j)=2**(i*j).
        # Evaluate that known kernel exactly as integers, not rounded exp().
        matrix=[[2**(i*j) for j in range(N)] for i in range(N)]
        normalized=[[Fraction(v,sum(row)) for v in row] for row in matrix]
        rank,det=eliminate(matrix);normalized_rank,_=eliminate(normalized)
        vandermonde=1
        for i in range(N):
            for j in range(i+1,N):vandermonde*=2**j-2**i
        assert rank==normalized_rank==N and det==vandermonde
        records.append(dict(N=N,rank=rank,normalized_rank=normalized_rank,
            determinant_exact=str(det),positive_determinant=True,
            consequence='A factorization through fewer than N fixed linear features cannot reproduce this kernel for all listed queries and arbitrary values.'))
    # Equal unweighted value sums do not imply equal query-weighted attention.
    values_a=[Fraction(1),Fraction(-1)];values_b=[Fraction(0),Fraction(0)]
    weights=[Fraction(1,3),Fraction(2,3)]
    assert sum(values_a)==sum(values_b)==0
    first=sum(a*b for a,b in zip(weights,values_a));second=sum(a*b for a,b in zip(weights,values_b))
    assert first!=second
    target=args.output;assert not target.exists()
    write(target,dict(records=records,moment_collision=dict(shared_sum='0',attention_A=str(first),attention_B=str(second)),
        UTC=dt.datetime.now(dt.timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        evidence_class='ANALYTIC_BOUND',subtype='EXECUTED_EXACT_RATIONAL_CONSTRUCTION',
        GPU_execution=False,pretrained_Ouro_data=False,native_BF16_kernel_equivalence='NOT_RUN',
        scope='Universal fixed-dimensional linear feature-summary claim only. No lower bound for every nonlinear data structure or approximate method.',
        source=dict(url='https://arxiv.org/html/2009.14794v4',locators=['Section2.2 equation4','Section2.3 equations6-7'],
            read_utc_window=['2026-10-19T11:24:49Z',dt.datetime.now(dt.timezone.utc).isoformat()],
            timestamp_note='Observed sequence window. Exact individual HTTP timestamps are not claimed.',
            note='Primary source distinguishes approximate random-feature attention. The rational rank proof is an explicit local derivation, not an attributed theorem.')))
    print('ATTENTION_SUMMARY_RANK_PASS',len(records),'exact rational cases and one collision')

if __name__=='__main__':main()
