"""Exact token-layer ownership and causal work for the two-dimensional DAG."""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Partition:
    length: int
    upper: int = 0
    suffix: int = 0

    def __post_init__(self):
        assert 0 < self.length <= 38912
        assert 0 <= self.upper < 36
        assert 0 <= self.suffix < self.length

    @property
    def prefix(self): return self.length-self.suffix

    @property
    def lower(self): return 36-self.upper

    @property
    def P_rows(self): return self.prefix*self.lower

    @property
    def D_rows(self): return self.prefix*self.upper+self.suffix*36

    @property
    def P_attention_pairs(self): return self.lower*self.prefix*(self.prefix+1)//2

    @property
    def D_attention_pairs(self): return 36*self.length*(self.length+1)//2-self.P_attention_pairs

    def owners(self,token,layer):
        assert 0 <= token < self.length and 0 <= layer < 36
        return 'P' if token < self.prefix and layer < self.lower else 'D'

    def record(self):
        return dict(length=self.length,upper=self.upper,suffix=self.suffix,
            prefix=self.prefix,lower=self.lower,P_rows=self.P_rows,D_rows=self.D_rows,
            P_attention_pairs=self.P_attention_pairs,D_attention_pairs=self.D_attention_pairs,
            valid_KV_bytes=self.prefix*self.lower*4096,
            hidden_residual_bytes=self.prefix*16384 if self.upper else 0,
            scope='STRUCTURAL_ACCOUNTING. Block padding, descriptors and metadata add bytes. '
                'The official Native proxy can recompute its last prompt query for sampling. '
                'That query is recorded separately and is not hidden in this conserved partition.')


def aligned_suffix(length,cut):
    target=Fraction(length*cut,36)
    nearest=int((target+8)//16)*16
    return max(16,min((length-1)//16*16,nearest))


def matched_suffixes(lengths,cut):
    """Minimize aggregate row error on a16-token lattice without changing prompts."""
    assert all(length>=32 for length in lengths)
    total_target=Fraction(sum(lengths)*cut,36)
    feasible_total=int((total_target+8)//16)*16
    values=[max(16,min((length-1)//16*16,int(Fraction(length*cut,36)//16)*16)) for length in lengths]
    residual=feasible_total-sum(values)
    order=sorted(range(len(lengths)),key=lambda i:(-(Fraction(lengths[i]*cut,36)-values[i]),i))
    while residual>0:
        changed=False
        for i in order:
            if values[i]+16<lengths[i]:
                values[i]+=16;residual-=16;changed=True
                if not residual:break
        assert changed,'No feasible aligned suffix allocation'
    while residual<0:
        changed=False
        for i in reversed(order):
            if values[i]>16:
                values[i]-=16;residual+=16;changed=True
                if not residual:break
        assert changed,'No feasible aligned suffix allocation'
    actual=sum(values)*36;target=sum(lengths)*cut
    return values,dict(target_D_rows=target,actual_D_rows=actual,
        signed_error_rows=actual-target,absolute_error_fraction=abs(actual-target)/target,
        exact_match=actual==target,alignment_tokens=16,
        scope='Integer-lattice optimum for the aggregate token-row sum. '
            'Do not call a nonzero residual an exact work match.')


def active_spans(query_starts,computed,prefixes,cuts,layer,role):
    """Return nonempty request-local query tails that this stage must execute."""
    spans=[]
    for i,(left,right) in enumerate(zip(query_starts,query_starts[1:])):
        start=computed[i];count=right-left;prefix=prefixes[i];cut=cuts[i]
        if role=='P':
            if layer<cut:spans.append((i,left,right))
        elif layer>=cut:
            spans.append((i,left,right))
        else:
            skip=max(0,min(count,prefix-start))
            if skip<count:spans.append((i,left+skip,right))
    return spans


def tests():
    for length in (32,257,8000,8192,16384,32768):
        for upper in (0,2,4,8,12):
            for suffix in (0,length//16,length//4):
                p=Partition(length,upper,suffix)
                assert p.P_rows+p.D_rows==36*length
                assert p.P_attention_pairs+p.D_attention_pairs==36*length*(length+1)//2
                if length<=257:
                    counts={'P':0,'D':0}
                    for token in range(length):
                        for layer in range(36):
                            owner=p.owners(token,layer);counts[owner]+=1
                            # Every causal predecessor is available on P,or imported before its D consumer.
                            if owner=='P':assert token<p.prefix and layer<p.lower
                    assert counts=={'P':p.P_rows,'D':p.D_rows}
    spans=active_spans([0,8,9],[12,128],[16,0],[32,0],0,'D')
    assert spans==[(0,4,8),(1,8,9)],spans
    assert active_spans([0,8,9],[12,128],[16,0],[32,0],32,'D')==[(0,0,8),(1,8,9)]
    for cut in (2,4,8,12):
        values,audit=matched_suffixes([8000]*144,cut)
        assert audit['exact_match'] and all(v%16==0 for v in values)
    return dict(passed=True,scope='CPU ownership and metadata-span tests. No GPU or model correctness claim.')


if __name__=='__main__':print(tests())
