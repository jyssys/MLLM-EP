"""CPU contract tests. These tests are not model or GPU performance evidence."""
from .boundary import valid_segments
from .common import MODEL_PINS
from .worker import prompts

def info():
 return dict(num_layers=4,block_size=16,key_ptr=0,value_ptr=1000000,
  block_bytes=4*16*8,layer_bytes=16*8,token_bytes=8,device=0)

def test_boundary_does_not_ship_unused_depth_planes():
 tables=((0,1),(2,3),(4,5))
 rows=list(valid_segments(info(),tables,0,19,(1,2),(3,),19))
 assert sum(size for ptr,size,device in rows)==(19*(1+2*3)+18)*8*2

def test_unaligned_range_has_no_missing_tokens():
 rows=list(valid_segments(info(),((0,1),),3,21,(1,2),(3,),22))
 assert sum(size for ptr,size,device in rows)==18*4*8*2

def test_coalescing_preserves_exact_address_coverage():
 from .segments import coalesce
 a,b=coalesce([(8,8,0),(0,8,0),(32,8,0)],[(108,8,1),(100,8,1),(132,8,1)])
 assert a==[(0,16,0),(32,8,0)] and b==[(100,16,1),(132,8,1)]

def test_generic_boundary_control_preserves_full_kv_address_coverage():
 from vllm_rlt.pd.transport import kv_segments
 config=info();config['num_blocks']=6;tables=((0,1),(2,3),(4,5))
 actual=list(valid_segments(config,tables,0,32,range(4),(),32))
 expected=list(kv_segments(config,tables,0,32))
 def addresses(segments):return {n for pointer,size,device in segments for n in range(pointer,pointer+size)}
 assert addresses(actual)==addresses(expected)

def test_dory_prompt_contains_integer_token_ids():
 rows=prompts('dory',2,0)
 assert len(rows)==2 and all(type(x) is int and 0<=x<131072 for row in rows for x in row)

def test_declared_fixed_depths_preserve_checkpoint_defaults():
 assert [MODEL_PINS[x][2] for x in ('ouro','huginn','dory')]==[4,32,2]

def test_split_frontier_is_a_real_prefix():
 from vllm_rlt.request import Request
 for R,cut in ((4,2),(32,16),(2,1)):
  ranges=Request.prefill_boundaries(128,R)
  for depth in range(cut):
   ranges[depth][0]=128;ranges[depth+1][1]=128
  assert all(left==128 for left,right in ranges[:cut])
  assert ranges[cut]==[0,128]

def test_fidelity_comparison_preserves_first_divergence():
 from .compare import agreement
 a=[dict(sample_id=0,input_ids=[1],output_ids=[2,3],exit_depths=[4,4])]
 b=[dict(sample_id=0,input_ids=[1],output_ids=[2,5,6],exit_depths=[4,4,4])]
 r=agreement(a,b)
 assert r['N']==1 and r['exact']==0 and r['input_equal']==1
 assert r['first_divergences'][0]['first_mismatch']==1
 assert r['length_equal']==0

def test_bootstrap_uses_paired_observations():
 from .compare import bootstrap
 blocks=[[(dict(e2e_ms=x),dict(e2e_ms=x/2)) for x in (2.,4.,8.)] for _ in range(3)]
 r=bootstrap(blocks,samples=30)
 assert r['speedup']==2. and r['ci95']==[2.,2.]
